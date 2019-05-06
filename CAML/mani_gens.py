import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn.functional as F


class MVNManifoldApproximator:
    def learn_mu_icov(self, Xs, compute_cov=True, esp_const=1e-3):
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

    def __call__(self, xs_mani, feature_map, slow=False):
        if slow:
            zs_mani = []
            with tc.no_grad():
                for x in xs_mani:
                    zs_mani.append(feature_map(x).view(x.size(0), -1).unsqueeze(0))
                zs_mani = tc.cat(zs_mani, 0)    
        else:
            zs_mani = feature_map(xs_mani.view(-1, *xs_mani.size()[2:]))
            zs_mani = zs_mani.view(*xs_mani.size()[0:2], -1)
        # learn params
        mus, Ms = self.learn_mu_icov(zs_mani)
        return mus, Ms
        

class ImageRotationManifold:
    def __init__(self, deg_max, deg_delta=1.0):
        self.deg_max = deg_max
        self.deg_delta = deg_delta
        
    def __call__(self, xs):
        xs_rot_all = []
        deg = self.deg_max
        delta = self.deg_delta
        for d in np.arange(-deg, deg+delta, delta):  
            d_rad = d / 360.0 * 2 * np.pi
            theta = T([[np.cos(d_rad), -np.sin(d_rad), 0.0], [np.sin(d_rad), np.cos(d_rad), 0.0]],
                      device=xs.device).repeat(xs.size(0), 1, 1)
            grid = F.affine_grid(theta, xs.size())
            xs_rot = F.grid_sample(xs, grid)
            xs_rot_all.append(xs_rot.unsqueeze(1))
        xs_rot_all = tc.cat(xs_rot_all, 1)
    
        # bs x n_manifolds x dim
        return xs_rot_all

class OneStepShiftManifold:
    def __init__(self, sig_start_idx, sig_end_idx):
        self.sig_start_idx = sig_start_idx
        self.sig_end_idx = sig_end_idx
    
    def __call__(self, xs):
        xs_all = []
        for i in range(self.sig_start_idx+1, self.sig_end_idx+1):
            xs_i = xs.clone()
            xs_i[:, i] = xs_i[:, i-1]
            xs_all.append(xs_i.unsqueeze(1))
        xs_all = tc.cat(xs_all, 1)
        # bs x n_manifolds x dim
        return xs_all
        
        