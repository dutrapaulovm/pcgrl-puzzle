import argparse
from pcgrl import *
from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.wrappers import *
from utils import ActivationFunc
from inference import run_inference

if __name__ == "__main__":
        #envs = [Game.ZELDA.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZE.value, Game.DUNGEON.value]
        envs = [Game.ZELDA.value]

        parser = argparse.ArgumentParser()
        parser.add_argument("--results_dir", default="./results/")
        parser.add_argument("--total_timesteps", default = 1000, type=int)
        parser.add_argument("--n_steps", default = 128, type=int)
        parser.add_argument("--learning_rate", default = 2.5e-4, type=float)
        parser.add_argument("--policy_size", default=[64, 64], type=int, nargs='*')
        parser.add_argument("--activation", default=ActivationFunc.SIGMOID.value, type=str)
        parser.add_argument("--entropy_min", default = 1.80, type=float)
        parser.add_argument("--envs", default=envs, type=str, nargs='*')
        parser.add_argument("--agent", default=Experiment.AGENT_HQBDC.value, type=str)
        parser.add_argument("--representations", default=[Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value], type=str, nargs='*')
        parser.add_argument("--observations", default=[WrappersType.MAP.value], type=str, nargs='*')
        parser.add_argument("--seed", type=int, default=1000)
        args = parser.parse_args()

        run_inference(results_dir = args.results_dir,
                total_timesteps = args.total_timesteps,
                learning_rate   = args.learning_rate,        
                n_steps         = args.n_steps,   
                policy_size     = args.policy_size,             
                act_func        = args.activation,
                entropy_min     = args.entropy_min,
                agent           = args.agent,                        
                envs            = args.envs,
                seed            = args.seed)