import os, sys
import numpy as np
import types
from scipy.io import loadmat

import torch as tc
import torch.tensor as T
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
sys.path.append('../examples')
from SGD import SGD

def load_alarm_data(batch_size):
    matfile = loadmat('alarm_data.mat')
    
    xs_train = T(matfile['xs_train'].astype(np.float32)) / 100.0
    xs_train = xs_train[:, 0, :].unsqueeze(1)
    ind = tc.isnan(xs_train).sum(1).sum(1) == 0
    xs_train = xs_train[ind]
    assert(tc.isnan(xs_train).sum() == 0)
    ys_train = T(matfile['ys_train']).long().squeeze()
    ys_train = ys_train[ind]
    
    xs_val = T(matfile['xs_val'].astype(np.float32)) / 100.0
    xs_val = xs_val[:, 0, :].unsqueeze(1)
    ind = tc.isnan(xs_val).sum(1).sum(1) == 0
    xs_val = xs_val[ind]
    assert(tc.isnan(xs_val).sum() == 0)
    ys_val = T(matfile['ys_val']).long().squeeze()
    ys_val = ys_val[ind]
    
    xs_test = T(matfile['xs_test'].astype(np.float32)) / 100.0
    xs_test = xs_test[:, 0, :].unsqueeze(1)
    ind = tc.isnan(xs_test).sum(1).sum(1) == 0
    xs_test = xs_test[ind]
    assert(tc.isnan(xs_test).sum() == 0)
    ys_test = T(matfile['ys_test']).long().squeeze()
    ys_test = ys_test[ind]
    
    print("# train data = %d, # val data = %d, test data = %d"%(
        ys_train.size(0), ys_val.size(0), ys_test.size(0)))
        
    ld = types.SimpleNamespace()   
    ld.train = DataLoader(TensorDataset(xs_train, ys_train), batch_size=batch_size, shuffle=True)
    ld.val = DataLoader(TensorDataset(xs_val, ys_val), batch_size=batch_size, shuffle=True)
    ld.test = DataLoader(TensorDataset(xs_test, ys_test), batch_size=batch_size, shuffle=True)
    return ld

def load_alarm_data_spec(batch_size):
    matfile = loadmat('alarm_data_spec.mat')
    
    xs_train = T(matfile['xs_train'].astype(np.float32))
    ys_train = T(matfile['ys_train']).long().squeeze()
    
    
#     xs_val = T(matfile['xs_val'].astype(np.float32))
#     ys_val = T(matfile['ys_val']).long().squeeze()
    
    xs_test = T(matfile['xs_test'].astype(np.float32))
    ys_test = T(matfile['ys_test']).long().squeeze()
    
    print("# train data = %d, test data = %d"%(
        ys_train.size(0), ys_test.size(0)))
        
    ld = types.SimpleNamespace()   
    ld.train = DataLoader(TensorDataset(xs_train, ys_train), batch_size=batch_size, shuffle=True)
    ld.test = DataLoader(TensorDataset(xs_test, ys_test), batch_size=batch_size, shuffle=True)
    return ld


class AlarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=[2, 3], stride=1)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[2, 3], stride=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=[1, 3], stride=1)
#         self.pool3 = nn.MaxPool1d(3, stride=2)
#         self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(256, 2)
        
    def feature(self, xs):
        xs = self.pool1(F.relu(self.conv1(xs)))
        xs = self.pool2(F.relu(self.conv2(xs)))
        xs = F.relu(self.conv3(xs))
#         xs = self.conv4(xs)
        return xs
        
    def forward(self, xs):
        xs = self.feature(xs)
        xs = self.fc1(xs.view(xs.size(0), -1))
        return xs
    
    def load(self, model_full_name):
        self.load_state_dict(tc.load(model_full_name))
        
    def save(self, model_full_name):
        tc.save(self.state_dict(), model_full_name)
    
if __name__ == "__main__":
    ## parameters
    exp_name = "Alarm"
    params = types.SimpleNamespace()
    params.model_name = "model_"+exp_name
    params.model_root = "snapshots"
    params.batch_size = 50
    params.optimizer = "Adam"
    params.n_epochs = 200
    params.learning_rate = 0.001
    params.lr_decay_n_epochs = 50
    params.lr_decay_rate = 0.5
    params.load_model = True
    params.n_epoch_print = 1

    ld = load_alarm_data_spec(params.batch_size)
    model = AlarmNet()
    learner = SGD(params, model)
    learner.train(ld.train)
    #learner.test([ld.train, ld.val, ld.test], ["train", "val", "test"])
    learner.test([ld.train, ld.test], ["train", "test"])
    
    
    
    ##  measure correct predictions among true positives
    n_ts = 0.0
    n_tps = 0.0
    n_fs = 0.0
    n_fps = 0.0
    for xs, ys in ld.test:
        xs = xs.cuda()
        ys = ys.cuda()
        
        yhs = model(xs)[:, 1] > -10.0
        idks = (model(xs)[:, 1] <= -10.0) & (model(xs)[:, 1] >= -50.0)
#         yhs = model(xs).argmax(1)
#         idks = tc.zeros_like(ys).cuda().byte()
        ts = (ys == 1) & (~idks)
        tps = (yhs == 1) & ts  & (~idks)
        
        fs = (ys == 0)  & (~idks)
        fps = (yhs == 1) & fs & (~idks)
        
        n_ts += ts.sum().float()
        n_tps += tps.sum().float()
        n_fs += fs.sum().float()
        n_fps += fps.sum().float()
        
    print("# detection / true positives = %d / %d = %f"%(n_tps, n_ts, n_tps/n_ts))
    print("# detection / false positives = %d / %d = %f"%(n_fps, n_fs, n_fps/n_fs))
        
    
    
