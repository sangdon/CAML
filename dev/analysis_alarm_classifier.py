import os, sys
import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch as tc
import torch.tensor as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.append("../")
from CAML.callib import CalibrationError


class AdaboostWithRejection(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, xs):
        return xs
    
    def conf_pred(self, xs, yhs=None):
        fhs = self(xs)
        if yhs is None:
            yhs = fhs.argmax(1)
        phs = F.softmax(fhs, 1).gather(1, yhs.view(-1, 1)).squeeze()
        return phs

if __name__ == "__main__":
    
    matfile = loadmat('ys_fhs.mat')
    ys_data = T(matfile['ys_fhs'][0][0][0]).long().squeeze()
    fhs_data = T(matfile['ys_fhs'][0][0][1]).float()
    
    
    n_val = int(ys_data.size(0) * 0.5)
    n_te = ys_data.size(0) - n_val

#     n_val = 96
#     n_te = ys_data.size(0) - n_val
    print("# val data = %d, test data = %d"%(n_val, n_te))
    
    ##FIXME: split test/val based on the patient id
    
          
    ld_val = DataLoader(TensorDataset(fhs_data[:n_val], ys_data[:n_val]), 
                        batch_size=10, shuffle=True)
    ld_te = DataLoader(TensorDataset(fhs_data[n_val:], ys_data[n_val:]), 
                       batch_size=10, shuffle=True)
    
    awr = AdaboostWithRejection()
    
    ## test error
    error = 0.0
    n_total = 0.0
    yhs_te = []
    fhs_te = []
    ys_te = []
    for xs, ys in ld_te:
        fhs = awr(xs)
        yhs = fhs.argmax(1)
        yhs_te.append(yhs)
        ys_te.append(ys)
        fhs_te.append(fhs[:, 1])
        error += (ys != yhs).sum().float()
        n_total += float(xs.size(0))
    print("# test error = %d / %d = %2f"%(error, n_total, error/n_total))
        
    ## calibration error
    ECE, ECE_mat = CalibrationError().compute_ECE(awr, awr.conf_pred, [ld_te], True)
    print("# ECE = %2.2f%%"%(ECE * 100.0))
    print(ECE_mat)
    
#     ## temperature scaling
#     from temperature_scaling import _ECELoss, ModelWithTemperature
#     awr_cal = ModelWithTemperature(awr)
#     awr_cal.set_temperature(ld_val, lr=0.1, max_iter=1000)
#     ECE, ECE_mat = CalibrationError().compute_ECE(awr_cal, awr_cal.conf_pred, [ld_te], True)
#     print("# ECE = %2.2f%%"%(ECE * 100.0))
#     print(ECE_mat)
    
    ## histogram
    n_bins = 20
    ys_te = tc.cat(ys_te)
    yhs_te = tc.cat(yhs_te)
    fhs_te = tc.cat(fhs_te)
    bins = tc.linspace(1.0, 10.0, n_bins+1)
    bins_lb = bins[0:-1]
    bins_ub = bins[1:]
    
    print("ys_te: ", ys_te)
    print("yhs_te: ", yhs_te)
    print("fhs_te: ", fhs_te)
    print("# P(correct | true 'alarm') = %d / %d"%((ys_te & yhs_te).sum(), ys_te.sum()))
    print("# P(correct | false 'alarm') = %d / %d"%(
        ((~ys_te.byte()) & (~yhs_te.byte())).sum(), (~ys_te.byte()).sum()))
    
    print("# n_true_positive: ", ((yhs_te == 1) & (ys_te == 1)).sum())
    acc_bins = []
    for lb, ub in zip(bins_lb, bins_ub):
        idx = (fhs_te >= lb) & (fhs_te < ub)
#         acc_i = ((yhs_te[idx] == 1) & (ys_te[idx] == 1)).float().mean()
#         acc_i = (yhs_te[idx] == ys_te[idx]).float().mean()
        acc_i = (yhs_te[idx] == ys_te[idx]).float().sum()
        acc_bins.append(acc_i.view(1))
    acc_bins = tc.cat(acc_bins)
    print(acc_bins)
    print(acc_bins.sum())
    
    plt.figure(1)
    plt.clf()
    xs_ori = tc.linspace(1.0, 10.0, n_bins+1)
    xs = xs_ori[0:-1] + (xs_ori[1:] - xs_ori[0:-1]) / 2.0
    w = (xs[1] - xs[0]) * 0.75

    plt.bar(xs.numpy(), acc_bins.numpy(), width=w, color='r', edgecolor='k')
    plt.xlabel("Score")
    plt.ylabel("Accuracy")
    plt.grid("on")
    plt.xlim([0, 9])
    plt.title("Score histogram of accurate predictions")
    plt.savefig("acc_hist_alarm.png")
    ## apply a non-parametric approach , e.g., histogram binning
    
    
    