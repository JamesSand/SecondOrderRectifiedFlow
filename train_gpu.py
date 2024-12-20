import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os
import argparse
import logging

############# hyper parameters ##########
wandb_enable = False
wandb_log_name = "second_order_v6"
ckpt_dir = "checkpoints"
# first_order_loss_scale = 1
# second_order_loss_scale = 1e-8

os.makedirs(ckpt_dir, exist_ok=True)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = 10.
M = D+5
VAR = 0.3
DOT_SIZE = 4
COMP = 3

initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
initial_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., D / 2.], [-D * np.sqrt(3) / 2., D / 2.], [0.0, - D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
initial_model = MixtureSameFamily(initial_mix, initial_comp)
samples_0 = initial_model.sample([10000])

target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([10000])

# move samples to GPU
samples_0 = samples_0.to(device)
samples_1 = samples_1.to(device)

print('Shape of the samples:', samples_0.shape, samples_1.shape)

plt.figure(figsize=(4,4))
plt.xlim(-M,M)
plt.ylim(-M,M)
plt.title(r'Samples from $\pi_0$ and $\pi_1$')
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_1$')
plt.legend()

plt.tight_layout()

# check if there is nan in torch tensor
def has_nan(tensor):
    nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
    if len(nan_indices[0]) > 0:
        return True, nan_indices
    else:
        return False, []

# def get_gradient_norm(model):
#   grad_norm_dict = {}
#   grad_norm_sum = 0.0
#   # Iterate through model parameters and compute gradient norm
#   for name, param in model.named_parameters():
#       if param.grad is not None:  # Check if gradients exist for the parameter
#           grad_norm = param.grad.data.norm(2)  # L2 norm of the gradient
#           grad_norm_dict[name] = grad_norm.item()
#           # print(f"Gradient norm for {name}: {grad_norm.item()}")
#           grad_norm_sum += grad_norm.item()

  # return grad_norm_dict, grad_norm_sum

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x
    
class MLP_2nd_order(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, first_order_input, x_input, t):
        inputs = torch.cat([first_order_input, x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x
    

def debug_logger(index, t, alpha_t, beta_t, first_order_alpha, first_order_beta, second_order_alpha, second_order_beta):
    print(f"t: {t[index]}")
    print(f"alpha_t: {alpha_t[index]}")
    print(f"beta_t: {beta_t[index]}")
    print(f"first_order_alpha: {first_order_alpha[index]}")
    print(f"first_order_beta: {first_order_beta[index]}")
    print(f"second_order_alpha: {second_order_alpha[index]}")
    print(f"second_order_beta: {second_order_beta[index]}")

class RectifiedFlow():
  def __init__(self, first_order_model=None, second_order_model=None, num_steps=1000):
    # self.model = model
    self.first_order_model = first_order_model
    self.second_order_model = second_order_model
    self.N = num_steps

  def get_train_tuple(self, z0=None, z1=None):

    # assert False, "Zhizhou: Modified this function, let it return second order before you call it"

    # we need to exclude 1, since 1 will make first order beta to be inf
    t = torch.rand((z1.shape[0], 1)) / (1+ 1e-6)
    a = 19.9
    b = 0.1

    # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
    alpha_t = torch.exp(- (1/4) * a * (1-t)**2 - (1/2) * b * (1-t))

    # first order alpha: 
    # d alpha_t / dt = alpha_t * 1/2 * (a (1-t) + b)
    first_order_alpha = alpha_t * (1/2) * (a * (1-t) + b)

    # second order alpha:
    # d^2 alpha_t / dt^2 = 1/2 * (alpha_t * (a(1-x)+b)^2 - a alpha_t)
    second_order_alpha = (1/2) * (alpha_t * (a * (1-t) + b)**2 - a * alpha_t)

    # beta_t = sqrt{1-alpha^2}
    beta_t = torch.sqrt(1 - alpha_t**2)
    # first order beta
    # d beta_t / dt = (- alpha  / sqrt{1 - alpha^2}) * (d alpha / dt)
    first_order_beta = (- alpha_t / torch.sqrt(1 - alpha_t**2)) * first_order_alpha
    # second order beta
    # d^2 beta_t / dt^2 = (- 1  / (1 - alpha^2) sqrt (1 - x^2)) * (d alpha / dt) + (- alpha  / sqrt{1 - alpha^2}) * (d^2 alpha / dt^2)
    second_order_beta = (- 1 / ((1 - alpha_t**2) * torch.sqrt(1 - alpha_t**2))) * first_order_alpha + first_order_beta * second_order_alpha

    # to device
    alpha_t = alpha_t.to(device)
    beta_t = beta_t.to(device)
    first_order_alpha = first_order_alpha.to(device)
    first_order_beta = first_order_beta.to(device)
    second_order_alpha = second_order_alpha.to(device)
    second_order_beta = second_order_beta.to(device)
    t = t.to(device)

    z_t = alpha_t * z1 + beta_t * z0
    first_order_gt = first_order_alpha * z1 + first_order_beta * z0
    second_order_gt = second_order_alpha * z1 + second_order_beta * z0

    return z_t, t, first_order_gt, second_order_gt
  
  def frist_and_second_order_predict(self, z_t, t):
    first_order_pred = self.first_order_model(z_t, t)
    second_order_pred = self.second_order_model(first_order_pred, z_t, t)
    return first_order_pred, second_order_pred

  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):

    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      first_order_pred, second_order_pred = self.frist_and_second_order_predict(z, t)
      z = z.detach().clone() + first_order_pred * dt + 0.5 * second_order_pred * dt**2

      traj.append(z.detach().clone())

    return traj
  
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters, first_order_loss_scale, second_order_loss_scale):
  loss_curve = []

  first_order_loss_list = []
  second_order_loss_list = []

  for i in tqdm(range(inner_iters+1)):
    optimizer.zero_grad()
    indices = torch.randperm(len(pairs))[:batchsize]
    batch = pairs[indices]
    z0 = batch[:, 0].detach().clone().to(device)
    z1 = batch[:, 1].detach().clone().to(device)

    z_t, t, first_order_gt, second_order_gt = rectified_flow.get_train_tuple(z0=z0, z1=z1)

    # zt shape: [bs, 2]
    # t shape: [bs, 1]

    first_order_pred, second_order_pred = rectified_flow.frist_and_second_order_predict(z_t, t)

    first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
    second_order_loss = (second_order_gt - second_order_pred).abs().pow(2).sum(dim=1)
    first_order_loss_mean = first_order_loss.mean()
    second_order_loss_mean = second_order_loss.mean()

    first_order_loss_mean = first_order_loss_scale * first_order_loss_mean
    second_order_loss_mean = second_order_loss_scale * second_order_loss_mean 

    loss = first_order_loss_mean + second_order_loss_mean

    

    # for debug
    first_order_loss_list.append(first_order_loss_mean.item())
    second_order_loss_list.append(second_order_loss_mean.item())

    loss.backward()

    if wandb_enable:
        wandb.log({
            "first_order_loss": first_order_loss_mean.item(),
            "second_order_loss": second_order_loss_mean.item(),
            "total_loss": loss.item(),
            # "grad_norm/first_order_grad_norm_sum" : first_order_grad_norm_sum,
            # "grad_norm/second_order_grad_norm_sum" : second_order_grad_norm_sum,
            # "grad_norm/first_order_grad_norm_max" : max(first_order_grad_norm_dict.values()),
            # "grad_norm/second_order_grad_norm_max" : max(second_order_grad_norm_dict.values())
        })

    # print(second_order_grad_norm_dict)
    # print(second_order_grad_norm_sum)
    # breakpoint()
    # second_order_grad_norm_list.append(second_order_grad_norm_sum)

    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  # # calcualte the mean of first and second order loss
  # first_order_loss_avg = np.mean(first_order_loss_list)
  # second_order_loss_avg = np.mean(second_order_loss_list)
  # print("First order loss avg: ", first_order_loss_avg)
  # print("Second order loss avg: ", second_order_loss_avg)

  return rectified_flow, loss_curve

if __name__ == '__main__':

  # set up logging
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  parser = argparse.ArgumentParser()
  parser.add_argument('--first_order_weight', type=float, default=1.0)
  parser.add_argument('--second_order_weight', type=float, default=1e-8)
  parser.add_argument('--save_dir_name', type=str, default="debug")
  args = parser.parse_args()

  for arg, value in vars(args).items():
    logging.info(f"{arg}: {value}")



  # generate training data
  x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))].to(device)
  x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))].to(device)
  x_pairs = torch.stack([x_0, x_1], dim=1)
  print(x_pairs.shape)


  # training code start
  iterations = 10000
  batchsize = 2048
  input_dim = 2

  rectified_flow_1 = RectifiedFlow(first_order_model=MLP(input_dim, hidden_num=100).to(device), second_order_model=MLP_2nd_order(input_dim, hidden_num=100).to(device), num_steps=100)
  rectified_flow_model_parameters = list(rectified_flow_1.first_order_model.parameters()) + list(rectified_flow_1.second_order_model.parameters())
  optimizer = torch.optim.Adam(rectified_flow_model_parameters, lr=5e-3)

  if wandb_enable:
      # wandb init here
      wandb.init(
          # set the wandb project where this run will be logged
          project="rectified_flow",
          name=wandb_log_name

          # # track hyperparameters and run metadata
          # config={
          # "learning_rate": 0.02,
          # "architecture": "CNN",
          # "dataset": "CIFAR-100",
          # "epochs": 10,
          # }
      )

  rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, x_pairs, batchsize, iterations, args.first_order_weight, args.second_order_weight)
  plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
  plt.title('Training Loss Curve')

  # save rectified_flow_1
  save_dir = os.path.join(ckpt_dir, args.save_dir_name)
  os.makedirs(save_dir, exist_ok=True)
  first_order_model_save_path = os.path.join(save_dir, 'first_order_model.pt')
  second_order_model_save_path = os.path.join(save_dir, 'second_order_model.pt')
  torch.save(rectified_flow_1.first_order_model.state_dict(), first_order_model_save_path)
  torch.save(rectified_flow_1.second_order_model.state_dict(), second_order_model_save_path)



