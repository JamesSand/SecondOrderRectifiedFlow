
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
from train_gpu import MLP, MLP_2nd_order, RectifiedFlow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# fisrt_order_weight = 1.0
# # second_order_weight_list = ["0.0"]
# # second_order_weight_list = [f"1e-{i}" for i in range(1, 14)]
# second_order_weight_list = ["1e-1", "1e-9", "1e-10", "1e-11"]
# # save_dir_name_list = [f"{fisrt_order_weight}_{second_order_weight}" for second_order_weight in second_order_weight_list]

# wandb_log_name = "second_order_v6"
ckpt_dir = "checkpoints"
input_dim = 2

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
# print('Shape of the samples:', samples_0.shape, samples_1.shape)


def load_model_from_dir(model_dir):

    rectified_flow_1 = RectifiedFlow(first_order_model=MLP(input_dim, hidden_num=100), second_order_model=MLP_2nd_order(input_dim, hidden_num=100), num_steps=100)

    save_dir = os.path.join(model_dir)
    first_order_model_save_path = os.path.join(save_dir, 'first_order_model.pt')
    second_order_model_save_path = os.path.join(save_dir, 'second_order_model.pt')
    rectified_flow_1.first_order_model.load_state_dict(torch.load(first_order_model_save_path))
    rectified_flow_1.second_order_model.load_state_dict(torch.load(second_order_model_save_path))

    # print("-" * 50)
    # print("model load success", model_dir)
    # print("-" * 50)

    return rectified_flow_1

def gen_overall_mask(traj):
    num_points = traj[0].shape[0]
    overall_mask = torch.ones(num_points, dtype=torch.bool)
    for point in traj:
        mask = (-5 <= point) & (point <= 5)
        mask = mask[:, 0] & mask[:, 1]
        # after this, we have inside = False, outside = True
        mask = ~mask
        overall_mask = overall_mask & mask
    return overall_mask

@torch.no_grad()
def eval_model(rectified_flow):
    N = 100
    z0=initial_model.sample([2000])
    traj = rectified_flow.sample_ode(z0=z0, N=N)
    overall_mask = gen_overall_mask(traj)
    # count outside num
    outside_num = overall_mask.sum()
    # print(f"Outside num: {outside_num}")
    total_num = z0.shape[0]
    good_ratio = outside_num / total_num
    # print in percetage way
    # print(f"Good ratio: {good_ratio * 100}%")
    return good_ratio


if __name__ == "__main__":

    # torch fix random seed
    torch.manual_seed(42)
    final_result = "exp_result.txt"
    if os.path.exists(final_result):
        os.remove(final_result)
    
    # sub_dir_path = "1.0_0.0_V"
    sub_dir_path = "1.0_1e-1_V"

    # for second_order_weight in second_order_weight_list:
    model_dir = os.path.join(ckpt_dir, sub_dir_path)
    rectified_flow = load_model_from_dir(model_dir)
    good_ratio = eval_model(rectified_flow)
    print("-" * 50)
    print(sub_dir_path)
    # print("second_order_weight:", second_order_weight)
    print(f"Good ratio: {good_ratio * 100}%")
    # print("-" * 50)

    # with open(final_result, "a+") as fw:
    #     fw.write(f"{second_order_weight}: {good_ratio}\n")

        # print("\n")
