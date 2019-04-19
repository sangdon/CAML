import os, sys
import numpy as np
import pickle
import glob 
import argparse
import time

import torch as tc
import torch.tensor as T
from  torch import nn

from CAML.callib import CalibrationError

class MahCalibrator(nn.Module):
    def __init__(self, params, model):
        super().__init__()
        self.model = model
        self.params = params
        self.width_max = None
        self.mus_loaded = None ##FIXME
        
    def save(self, fn):
        pass
    
    def load(self):
        # load manifold parameters
        model_root = self.params.caml_model_root
        self.manifold_model_fns = glob.glob(os.path.join(model_root, "params_manifolds_*.pk"))
        if len(self.manifold_model_fns) == 0:
            return False
        
        # load hyperparameters
        w_best_fn = os.path.join(model_root, "width_max_best.pk")
        if os.path.exists(w_best_fn): 
            w_best = pickle.load(open(w_best_fn, "rb"))
            self.set_width_max(w_best)
        else:
            return False
        return True
    
    def set_width_max(self, width_max):
        self.width_max = width_max*tc.ones(1).cuda()
    
    def get_width_max(self):
        return self.width_max.item()

    
    def compute_error(self, ld, model=None):
        error = 0.0
        n_total = 0.0
        with tc.no_grad():
            for xs, ys in ld:
                xs = xs.cuda()
                ys = ys.cuda()
                if model is None:
                    yhs = self(xs).argmax(1)
                else:
                    yhs = model(xs).argmax(1)
                error += (ys != yhs).sum().float()
                n_total += float(xs.size(0))
        return error/n_total

    ##FIXME: rewrite:
    def mah_dist_minibatch(self, zs, mus, Ms):
        if zs.size(0) == mus.size(0):
            zs_bar = zs.unsqueeze(0) - mus.unsqueeze(1)
            dd = zs_bar.matmul(Ms).matmul(zs_bar.transpose(1, 2))
            ds = tc.cat([d.diag().unsqueeze(1) for d in dd], 1).sqrt()
        else:
            ds = []
            for z in zs:
                z_bar = (z.unsqueeze(0) - mus).unsqueeze(1)
                ds_ = z_bar.matmul(Ms).matmul(z_bar.transpose(1, 2)).sqrt().squeeze()
                ds.append(ds_.unsqueeze(0))
            ds = tc.cat(ds, 0)
        return ds
    
    def compute_rotation_manifolds(self, xs, pred, deg_max, delta=1.0, slow=False):
        from CAML.rotation_manifolds import rotate_images, learn_mu_icov
        # rotate images
        xs_rot = rotate_images(xs, deg_max, delta=delta)
        if slow:
            zs_rot = []
            with tc.no_grad():
                for x in xs_rot:
                    zs_rot.append(pred.feature(x).view(x.size(0), -1).unsqueeze(0))
                zs_rot = tc.cat(zs_rot, 0)    
        else:
            zs_rot = pred.feature(xs_rot.view(-1, *xs_rot.size()[2:])).view(*xs_rot.size()[0:2], -1)
        # learn params
        mus, Ms = learn_mu_icov(zs_rot)
        return mus, Ms

    def find_best_width_linesearch(self, ld_val, w_lb, w_ub, w_delta):
        error_best = T(np.inf).cuda()
        w_best = None
        for i, w in enumerate(tc.arange(w_lb, w_ub+w_delta, w_delta)):
            self.set_width_max(w)
            t_start = time.time()
            error = self.compute_error(ld_val)
            t_end = time.time()
            if error_best*1.01 < error:
                break
            if error < error_best:
                error_best = error
                w_best = w
            print("[search width_max, %.2f sec.] "
                  "w = %.4f, w_lb = %.4f, w_ub = %.4f, best_w = %.4f, "
                  "error = %.4f, error_best = %.4f"%
                  (t_end-t_start, w, w_lb, w_ub, w_best, error, error_best))
        return w_best
                              
        
    def find_best_width_coarse2fine_linesearch(self, ld_val, w_lb, w_ub, n_w, eps=1e-1):
        w_best_prev = T(0.0).cuda()
        w_best = T(np.inf).cuda()
        while (w_best - w_best_prev).abs() > eps:
            w_rng = tc.linspace(w_lb, w_ub, n_w+1)
            error_best = T(np.inf).cuda()
            for i, w in enumerate(w_rng):
                self.set_width_max(w)
                error = self.compute_error(ld_val)
                if error < error_best:
                    error_best_prev = error_best
                    error_best = error
                    
                    w_best_prev = w_best
                    w_best = w
                    w_lb = w_rng[i-1] if i>0 else w_lb
                    w_ub = w_rng[i+1] if i<len(w_rng)-1 else w_ub
                    print("[search width_max] "
                          "w = %.4f, w_lb = %.4f, w_ub = %.4f, best_w = %.4f, error = %.4f"%
                          (w, w_rng[0], w_rng[-1], w_best, error_best))
        
        return w_best
            
            
    
    ##FIXME: change the slow option to something else
    def train(self, ld_tr, ld_val, slow=False):
        ## parameters
        n_manifolds = self.params.n_manifolds
        model_root = self.params.caml_model_root
        deg_max = self.params.rotation_max
        delta = self.params.rotation_delta
        
        ## init
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        
        ## generate manifolds
        n_manifold_generated = 0
        for i, (xs, ys) in enumerate(ld_tr):
            if n_manifold_generated >= n_manifolds:
                break
            xs = xs.cuda()
            ys = ys.cuda()
            batch_size = ys.size(0)

            save_fn = os.path.join(model_root, "params_manifolds_%d-%d.pk")%(
                (i)*batch_size, (i+1)*batch_size-1)
            if not os.path.exists(save_fn):
                print("[%d/%d] learn manifolds..."%((i+1)*batch_size, n_manifolds))
                
                ## compute means and inverse covariance matrices of manifolds
                manifold_model = self.compute_rotation_manifolds(
                    xs, self.model, deg_max, delta, slow)
                ## save means and inverse covariance matrices of manifolds
                with open(save_fn, 'wb') as f:
                    pickle.dump([m.cpu() for m in manifold_model] + [ys.cpu()], f)
            n_manifold_generated += batch_size
        self.manifold_model_fns = glob.glob(os.path.join(model_root, "params_manifolds_*.pk"))
        
        ## choose a hyperparameter
        w_best_fn = os.path.join(model_root, "width_max_best.pk")
        if not os.path.exists(w_best_fn):
#             w_best = self.find_best_width_coarse2fine_linesearch(ld_val, 0.0, 1e5, 5, eps=1e-1)
            w_best = self.find_best_width_linesearch(
                ld_val, self.params.width_max_lb, self.params.width_max_ub, 
                self.params.width_search_delta)
            pickle.dump(w_best, open(w_best_fn, "wb"))
        else:
            w_best = pickle.load(open(w_best_fn, "rb"))
        self.set_width_max(w_best)
        
        return self
    
    def test(self, ld):
        # compute ECE
        with tc.no_grad():
            class_error = self.compute_error(ld, self.model)
            ECE = CalibrationError(self.params.n_bins).compute_ECE(
                self.model.forward, self.forward_conf, [ld])
        return ECE, class_error
            
    
    def forward(self, xs):
        with tc.no_grad():
            zs = self.model.feature(xs)
            # compute distances
            ds_tr = []
            ys_tr = []
            if self.mus_loaded is not None:
                for mus, Ms, ys in zip(self.mus_loaded, self.Ms_loaded, self.ys_loaded):
                    ds_b = self.mah_dist_minibatch(zs, mus, Ms).detach_()
                    ds_tr.append(ds_b)
                    ys_tr.append(ys)
            else:
                for i, fn in enumerate(self.manifold_model_fns):
                    with open(fn, 'rb') as f:
                        mus, Ms, ys = pickle.load(f)
                    mus = mus.cuda().detach_() ##FIXME: cuda()
                    Ms = Ms.cuda().detach_() ##FIXME: cuda()
                    ys = ys.cuda().detach_() ##FIXME: cuda()
                    
                    ds_b = self.mah_dist_minibatch(zs, mus, Ms).detach_()
                    ds_tr.append(ds_b)
                    ys_tr.append(ys)
                
            ds_tr = tc.cat(ds_tr, 1).detach()
            ys_tr = tc.cat(ys_tr).detach()
            
            # estimate a label distribution
            phs = []
            for y in range(self.params.n_labels):
                # label distribution
                I_close = (ds_tr <= self.get_width_max()).float()
                I_close_y = I_close.mul((ys_tr == y).float().unsqueeze(0))
                ph = I_close_y.sum(1).div(I_close.sum(1))
                phs.append(ph.unsqueeze(1))
                
 
            phs = tc.cat(phs, 1)
            ind = tc.isnan(phs).sum(1) == self.params.n_labels
            phs[ind, :] = 1.0/float(self.params.n_labels)
            phs[tc.isnan(phs)] = 0.0
            
        return phs

    def forward_conf(self, xs, yhs=None):
        if yhs is None:
            yhs = self.model(xs).argmax(1)
        phs = self(xs)
        phs = phs.gather(1, yhs.view(-1, 1)).squeeze()
        return phs
        
        

