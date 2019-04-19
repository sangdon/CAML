import numpy as np
import sys, os

import torch as tc
from torch import nn

class CalibrationError(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins
        bin_boundaries = tc.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    
    def get_acc_conf_mat(self, yhs, phs, ys):
        accs = yhs.eq(ys)
        confs = phs
        
        acc_conf_mat = tc.zeros(self.n_bins, 3)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confs > bin_lower.item()) & (confs <= bin_upper.item())
            acc_conf_mat[i, 0] = in_bin.float().sum()
            if acc_conf_mat[i, 0] > 0:
                acc_conf_mat[i, 1] = accs[in_bin].float().sum()
                acc_conf_mat[i, 2] = confs[in_bin].float().sum()
        
        return acc_conf_mat
        

    # ECE: emprical calibration error
    def ECEmat2ECE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        ECE_mat[:, 0] = ECE_mat[:, 0].div(ECE_mat[:, 0].sum())
        ECE = ECE_mat[:, 0].mul((ECE_mat[:, 1] - ECE_mat[:, 2]).abs()).sum()
        return ECE

    # MOCE: maximum-overconfident calibration error
    def ECEmat2MOCE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # mean accuracy
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # mean confidence
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        MOCE = (ECE_mat[:, 2] - ECE_mat[:, 1]).clamp(0.0, np.inf).max()
        return MOCE

    # EUCE: expected-underconfident calibration error
    def ECEmat2EUCE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # mean accuracy
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # mean confidence
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        # frequency of each bin
        ECE_mat[:, 0] = ECE_mat[:, 0].div(ECE_mat[:, 0].sum())
        #FIXME: does not count all samples, loose information, need to use with MOCE
        EUCE = ECE_mat[:, 0].mul((ECE_mat[:, 1] - ECE_mat[:, 2]).clamp(0.0, np.inf)).sum()
        return EUCE

    def ECEmat2MOEUCE(self, ECE_mat):
        MOCE = ECEmat2MOCE(ECE_mat)
        EUCE = ECEmat2EUCE(ECE_mat)
        return MOCE + EUCE


    def decomposeECEmat(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        n_samples = ECE_mat[:, 0]
        mean_accuracy = ECE_mat[:, 1]
        mean_accuracy[ind] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        mean_confidence = ECE_mat[:, 2]
        mean_confidence[ind] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        return n_samples, mean_confidence, mean_accuracy

    
    def compute_ECE(self, label_pred, conf_pred, lds, return_ECE_mat=False):
        ECE_mat = None
        for ld in lds:
            for i, (xs, ys) in enumerate(ld):
                xs = xs.cuda()
                ys = ys.cuda()
                yhs = label_pred(xs).argmax(1)
                phs = conf_pred(xs, yhs)

                ECE_mat_b = self.get_acc_conf_mat(yhs, phs, ys)
                ECE_mat = ECE_mat + ECE_mat_b if ECE_mat is not None else ECE_mat_b

        ECE = self.ECEmat2ECE(ECE_mat)
        if return_ECE_mat:
            return ECE, ECE_mat
        else:
            return ECE
    
#     def plot_reliablity_diagram(self, fig_fn, label_pred, conf_pred, lds):
#         ECE, ECE_mat = self.compute_ECE(label_pred, conf_pred, lds, True)
#         n_samples, mean_confidence, mean_accuracy = self.decomposeECEmat(ECE_mat)
#         plot_reliability_diag(self.n_bins, mean_accuracy, n_samples, fig_fn=fig_fn, fontsize=20, 
#                               ECE=ECE)
