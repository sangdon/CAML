import os, sys

from CAML.CAML import MahCalibrator
from CAML.option import EvalArgParser
from CAML.dataloader import data_loader
from CAML.utils import *

if __name__ == "__main__":
    
    ## read options
    params = EvalArgParser().read_args()
    
    ## load a classifier
    model = load_model(params)
    print(model)
    
    ## init a source dataset loader
    ld_learning = data_loader(
        params.dataset_root, params.batch_size, params.image_size, params.gray_scale)
    print("#training:", len(ld_learning.train)*params.batch_size)
    print("#val:", len(ld_learning.val)*params.batch_size)
    print("#test:", len(ld_learning.test)*params.batch_size)
    
    ##  test CAML
    mah_cal = MahCalibrator(params, model)
    mah_cal.load()
    ece, cls_error = mah_cal.test(ld_learning.test)
    print("[test results] cls_error = %2.4f%%, ECE = %2.4f%%"%(cls_error*100.0, ece*100.0))
    
    

