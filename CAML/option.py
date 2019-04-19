import argparse
import os

class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
            fromfile_prefix_chars='@')
        
        # CAML
        self.parser.add_argument('--exp_name', type=str, required=True, help='...')
        self.parser.add_argument('--batch_size', type=int, default=100, help='...')
        self.parser.add_argument('--learn_model', type=str, default='n', help='...')
        self.parser.add_argument('--n_labels', type=int, default=10, help='...')
        ##FIXME: ratio?
        self.parser.add_argument('--n_manifolds', type=int, default=2000, help='...')
        self.parser.add_argument('--n_bins', type=int, default=15, help='...')
        
        # classifier
        self.parser.add_argument('--model_def_path', type=str, default="user_input/model.py", 
                                 help='...')
        self.parser.add_argument('--model_name', type=str, default="LeNet5", help='...')
        self.parser.add_argument('--model_path', type=str, default="user_input/model_LeNet5.pt", 
                            help='...')

        ##FIXME: what is default or required?
        # data
        self.parser.add_argument('--dataset_root', type=str, default="user_input/MNIST", help='...')
        self.parser.add_argument('--image_size', type=int, default=28, help='...')
        self.parser.add_argument('--gray_scale', type=int, default=1, help='...')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params
    
class TrainArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        
        # manifolds parameters
        self.parser.add_argument('--rotation_max', type=float, default=30.0, help='...')
        self.parser.add_argument('--rotation_delta', type=float, default=1.0, help='...')
        
        self.parser.add_argument('--width_max_lb', type=float, default=0.0, help='...')
        self.parser.add_argument('--width_max_ub', type=float, default=1000.0, help='...')
        self.parser.add_argument('--width_search_delta', type=float, default=1.0, help='...')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params
        
class TestArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--image_path', required=True, help='path to an image file')
        
    def read_args(self):
        ## read options
        params, _ = self.parser.parse_known_args()
        params.caml_model_root = os.path.join("cache", params.exp_name, "caml_model_root")
        return params