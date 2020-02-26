import torch
import torch.optim as optim


def discount(r_list, gamma):
    """
    Compute discounted sum of future values
    """
    discounted_return = 0
    for i in range(len(x)):
        discounted_return += gamma**i * r_list[i]
    return discounted_return

def inverse_distance(h, h_i, epsilon=1e-3):
    return 1 / (torch.dist(h, h_i) + epsilon)

def get_optimizer(optimizer_name,params,lr):
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(params,lr)
