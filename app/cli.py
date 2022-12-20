import argparse
from train import RL_ALGO
from pcgrl.utils.utils import ActivationFunc, RlAlgo
import numpy as np

def _get_args():
    """Parse arguments from the command line and return them."""
    parser = argparse.ArgumentParser(description=__doc__)
    # add the argument for the environment to run
    parser.add_argument('--env', '-e',
        type=str,
        help='The environment name to play.'
    )
    
    parser = argparse.ArgumentParser()        
    parser.add_argument("--type", help="Training(T) or inference(I)", default="T", type=str)        
    parser.add_argument("--algo", help="RL Algorithm", default=RlAlgo.PPO.value, type=str, choices=list(RL_ALGO.keys()))
    parser.add_argument("--results_dir", help = "Results folder", default="./results/")
    parser.add_argument("--total_timesteps", help = "The total time steps", default = 10000, type=int)
    parser.add_argument("--n_steps", default = 2048, type=int)
    parser.add_argument("--learning_rate", help = "The learning rate", default = 3e-4, type=float)
    parser.add_argument("--gamma", default = 0.99, type=float)
    parser.add_argument("--n_epochs", default = 64, type=int)
    parser.add_argument("--batch_size", default = 10, type=int)
    parser.add_argument("--policy", default="MlpPolicy", type=str)        
    parser.add_argument("--activation", default=ActivationFunc.SIGMOID.value, type=str)        
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uuid", default="", type=str)
    args = parser.parse_args()
            
    if args.seed < 0:        
            args.seed = np.random.randint(2**32 - 1, dtype="int64").item()        
            
    if args.type != "":
            assert args.type != "T" or args.type != "I", "the type must be informed training(T) or inference(I)"                        
    
    return parser.parse_args()

if __name__ == "__main__":        
    # get arguments from the command line
    args = _get_args()