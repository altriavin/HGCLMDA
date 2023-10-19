import argparse
import torch

def parse_args():
  parser = argparse.ArgumentParser('Model Description')
  parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
  parser.add_argument('--batch', default=256, type=int, help='batch size')
  parser.add_argument('--sslbatch', default=2048, type=int, help='SSL batch size')
  parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
  parser.add_argument('--epoch', default=30, type=int, help='number of epochs')
  parser.add_argument('--decayRate', default=0.96, type=float, help='decay rate for learning rate')
  parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
  parser.add_argument('--latdim', default=256, type=int, help='embedding size')
  parser.add_argument('--rank', default=4, type=int, help='embedding size')
  parser.add_argument('--memosize', default=2, type=int, help='memory size')
  parser.add_argument('--n_factors', default=4, type=int, help='Number of factors to disentangle the original embed-size representation.')
  parser.add_argument('--n_iterations', default=2, type=int, help='Number of iterations to perform the routing mechanism.')
  parser.add_argument('--sampNum', default=100, type=int, help='batch size for sampling')
  parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
  parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
  parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
  parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
  parser.add_argument('--load_model', default=None, help='model name to load')
  parser.add_argument('--shoot', default=20, type=int, help='K of top k')
  parser.add_argument('--data', default='mDA1', type=str, help='name of dataset')
  parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
  parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
  parser.add_argument('--droprate', default=0.5, type=float, help='rate for dropout')
  parser.add_argument('--slot', default=5, type=float, help='length of time slots')
  parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
  parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')
  parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
  parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
  parser.add_argument('--hyperReg', default=1e-4, type=float, help='regularizer for hyper connections')
  parser.add_argument('--temp', default=1, type=float, help='temperature in ssl loss')
  parser.add_argument('--ssl_reg', default=1e-4, type=float, help='reg weight for ssl loss')
  parser.add_argument('--percent', default=0.0, type=float, help='percent of noise for noise robust test')
  parser.add_argument('--tstNum', default=99, type=int, help='Numer of negative samples while testing, -1 for all negatives')
  parser.add_argument('--seed', default=10, type=int, help='Random seed')
  parser.add_argument('--gpu_id', default=0, type=int, help='gpu id to use')
  parser.add_argument('--hyper', default=1, type=int, help='using the hypergraph?')
  parser.add_argument('--save', default=0, type=int, help='save the .npy(1)')
  parser.add_argument('--save_emb', default=0, type=int, help='save the emb(1)')
  parser.add_argument('--count', default=10, type=int, help='run count times')

  #	return parser.parse_args()
  return parser

args, _ = parse_args().parse_known_args()

print(f'args: {args}')

args.decay_step = args.trnNum/args.batch
if torch.cuda.is_available():
    args.device = "cuda:" + str(args.gpu_id)
else:
    args.device = "cpu"