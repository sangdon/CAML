import os, sys
from PIL import Image
import time
from torchvision import datasets, transforms

from CAML.CAML import MahCalibrator ##FIXME: too redundent naming
from CAML.option import RunArgParser
from CAML.utils import *


if __name__ == "__main__":
    
    ## read options
    params = RunArgParser().read_args()

    ## load a classifier
    model = load_model(params)
    print("# network:")
    print(model)
    
    ## train CAML
    mah_cal = MahCalibrator(params, model)
    mah_cal.load()
    
    ## read image and predict confidence
    x = Image.open(params.image_path)
    x = transforms.ToTensor()(x).unsqueeze(0)
    t_start = time.time()
    ph = mah_cal.forward_conf(x)
    t_end = time.time()
    print("# [%2.4f sec.] Confidence: %2.4f"%(t_end-t_start, ph))
    
    
    
    

