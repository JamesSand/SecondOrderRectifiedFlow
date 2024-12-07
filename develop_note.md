## Develop Note

> This is the develop note made by JamesSand to facilitate his development of this project. You can ignore this file

| Version    | Description |
| :--------: | :-------: |
| v1  | have nan loss, the output of the model is nan    |
| v2 | avoid random to 1 timestep, try to eliminate nan     |
| v3 | balance the second order loss. First order loss mean: 541; second order loss mean: 1.2e+14. So we divide second order loss with 1e-11     |
| v4 | second order loss scale 1e-8. Because the spikes of second order loss are more sharp than first order. We cannot only estimate by their means.   |
| v5 | scale up first order loss with 1e6   |




