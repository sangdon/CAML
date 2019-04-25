import argparse
import os

class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
            fromfile_prefix_chars='@')
        
        # CAML
        self.parser.add_argument('--exp_name', type=str, required=True, 
                                 help='name of an experiment')
        self.parser.add_argument('--batch_size', type=int, default=100, 
                                 help='size of batch images for learning')
        self.parser.add_argument('--relearn_model', type=int, default=0, 
                                 help='option whether re-learn model or not')
        self.parser.add_argument('--n_manifolds', type=int, default=2000, 
                                 help='number of manifolds to generate')
        self.parser.add_argument('--n_bins', type=int, default=15, 
                                 help='number of bins for evaluating calibration error')
        
        # classifier
        self.parser.add_argument('--model_def_path', type=str, 
                                 default="user_input/model.py", 
                                 help='path of a file that includes a neural net model definition')
        self.parser.add_argument('--model_name', type=str, default="LeNet5", 
                                 help='name of a neural net model')
        self.parser.add_argument('--model_path', type=str, default="user_input/model_LeNet5.pt", 
                                 help='path of a pretrained neural net model')

        # data
        self.parser.add_argument('--dataset_root', type=str, default="user_input/MNIST", 
                                 help='root path of a dataset directory')
        self.parser.add_argument('--image_size', type=int, default=28, 
                                 help='size of image. '
                                 'Images are rescaled by image_size x image_size.')
        self.parser.add_argument('--gray_scale', type=int, default=1, 
                                 help='whether convert an image into a gray scale image')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params
    
class TrainArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        
        # manifolds parameters
        self.parser.add_argument('--rotation_max', type=float, default=30.0, 
                                 help='maximum rotation angle in degree when generating manifolds.')
        self.parser.add_argument('--rotation_delta', type=float, default=1.0, 
                                 help='increase-size of rotation angle when generating manifolds.')
        
        self.parser.add_argument('--width_max_lb', type=float, default=0.0, 
                                 help='lower bound of width_max when searching a width_max parameter.')
        self.parser.add_argument('--width_max_ub', type=float, default=1000.0, 
                                 help='upper bound of width_max when searching a width_max parameter.')
        self.parser.add_argument('--width_search_delta', type=float, default=1.0, 
                                 help='increase-size of with_max when searching a width_max parameter.')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params
        
class TestArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--image_path', required=True, help='path to a test image file')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params