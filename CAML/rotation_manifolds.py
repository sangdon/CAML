import numpy as np
import torch as tc
import torch.tensor as T
import torch.nn.functional as F


def rotate_images(xs, deg, delta=1.0):
    xs_rot_all = []
    for d in np.arange(-deg, deg+delta, delta):  
        d_rad = d / 360.0 * 2 * np.pi
        theta = T([[np.cos(d_rad), -np.sin(d_rad), 0.0], [np.sin(d_rad), np.cos(d_rad), 0.0]],
                  device=xs.device).repeat(xs.size(0), 1, 1)
        grid = F.affine_grid(theta, xs.size())
        xs_rot = F.grid_sample(xs, grid)
        xs_rot_all.append(xs_rot.unsqueeze(1))
    xs_rot_all = tc.cat(xs_rot_all, 1)
    
    return xs_rot_all

def learn_mu_icov(Xs, compute_cov=True, esp_const=1e-3):
    """
    X: n x dims
    """
    ## mean
    mus = Xs.mean(1)
    ## zero mean: bs x n_rot x p
    Xbs = Xs - mus.unsqueeze(1)
    ## cov mat
    if compute_cov:
        # unbiased estimator
        eps = (esp_const*tc.eye(mus.size(1), mus.size(1), device=Xs.device)).unsqueeze(0)
        covs = (Xbs.transpose(1, 2).matmul(Xbs)) / (Xbs.size(1) - 1) + eps
        
        Ms = tc.inverse(covs)
        Ms = Ms/2.0 + Ms.transpose(1, 2)/2.0
    else:
        Ms = None
    
    return mus, Ms