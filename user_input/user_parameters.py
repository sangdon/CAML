import argparse

parser = argparse.ArgumentParser(description='Confidence Auditor based on Manifold Learning (CAML)', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# meta information
parser.add_argument('--exp_name', type=str, default='test_CAML', metavar='N', 
                    help='experiment name')
parser.add_argument('--use_cuda', type=str, default='y', metavar='N', 
                    help='option for cuda')

# learning parameters
parser.add_argument('--batch_size', type=int, default='100', metavar='N', 
                    help='batch size for a learner')
# parser.add_argument('--n_epochs', type=int, default='200', metavar='N', 
#                     help='the number of learning epochs')
# parser.add_argument('--optimizer', type=str, default='Adam', metavar='N', 
#                     help='optimizer')
# parser.add_argument('--learning_rate', type=float, default='0.001', metavar='N', 
#                     help='learning rate')
# parser.add_argument('--momentum', type=float, default='0.0', metavar='N', 
#                     help='momentum parameter')
# parser.add_argument('--n_epochs_decay', type=int, default='50', metavar='N', 
#                     help='the number of epochs to decay learning rate')
# parser.add_argument('--decay_rate', type=float, default='0.5', metavar='N', 
#                     help='decay rate')
# parser.add_argument('--std_loss_weight', type=float, default='1.0', metavar='N', 
#                     help='the weight of a standard loss')
# parser.add_argument('--reg_type', type=str, default='L2', metavar='N', 
#                     help='regularization type')
# parser.add_argument('--reg_param', type=float, default='1e-8', metavar='N', 
#                     help='the regularization parameter')
# parser.add_argument('--load_model', type=str, default='n', metavar='N', 
#                     help='loading trained model')
parser.add_argument('--learn_model', type=str, default='n', metavar='N', 
                    help='option to learn a model')

parser.add_argument('--n_labels', type=int, default='10', metavar='N', 
                    help='option for the number of labels')
parser.add_argument('--n_manifolds', type=int, default='2000', metavar='N', 
                    help='option for the number of manifolds to maintain')

# # verbose mode parameter
# parser.add_argument('--n_epoch_print', type=int, default='1', metavar='N', 
#                     help='the number of epochs to print learning status')
# parser.add_argument('--vis_decision_boundary', type=str, default='n', metavar='N', 

# read arguments
params = parser.parse_args()
