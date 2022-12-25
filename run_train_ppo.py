import argparse
from pcgrl import *
from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.utils.utils import ActivationFunc, RlAlgo
from simple_trainv1 import RL_ALGO, train_ppo
from pcgrl.wrappers import *
from pcgrl.utils.experiment import ExperimentManager

if __name__ == "__main__":
        agents   = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HHPD.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPD.value] 
        env = [Game.ZELDA.value, Game.MAZE.value, Game.MAZECOIN.value, Game.MINIMAP.value,  Game.DUNGEON.value]        
        representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value]
        parser = argparse.ArgumentParser()
        parser.add_argument("--type", help="Training(T) or inference(I)", default="T", type=str)        
        parser.add_argument("--algo", help="RL Algorithm", default=RlAlgo.PPO.value, type=str, choices=list(RL_ALGO.keys()))
        parser.add_argument("--results_dir", help = "Results folder", default="./results/")
        parser.add_argument("--total_timesteps", help = "The total time steps", default = 50000, type=int)
        parser.add_argument("--n_steps", default = 128, type=int)
        parser.add_argument("--learning_rate", help = "The learning rate", default = 2.5e-4, type=float)
        parser.add_argument("--gamma", default = 0.99, type=float)
        parser.add_argument("--n_epochs", default = 64, type=int)
        parser.add_argument("--batch_size", default = 10, type=int)
        parser.add_argument("--policy", default="MlpPolicy", type=str)        
        parser.add_argument("--activation", default=ActivationFunc.SIGMOID.value, type=str)
        parser.add_argument("--entropy_min", default = 1.80, type=float)
        parser.add_argument("--agent", default=agents, type=str, nargs='*', choices=list(agents))
        parser.add_argument("--representations", default=representations, type=str, nargs='*', choices=list(representations))
        parser.add_argument("--envs", default=env, help="environments games", type=str, nargs='*', choices=list(env))
        parser.add_argument("--observations", default=[WrappersType.MAP.value], type=str, nargs='*')
        parser.add_argument("--seed", type=int, default=1000)
        parser.add_argument("--uuid", default="", type=str)
        args = parser.parse_args()
                
        if args.seed < 0:        
                args.seed = np.random.randint(2**32 - 1, dtype="int64").item()        
                
        if args.type != "":
                assert args.type != "T" or args.type != "I", "the type must be informed training(T) or inference(I)"                
        
        experiment_manager = ExperimentManager(rl_algo = args.algo,
                results_dir = args.results_dir,
                total_timesteps = args.total_timesteps,
                learning_rate   = args.learning_rate,        
                n_steps         = args.n_steps,
                gamma           = args.gamma,        
                policy          = args.policy,                
                batch_size      = args.batch_size,
                n_epochs        = args.n_epochs,
                act_func        = args.activation,
                entropy_min     = args.entropy_min,
                envs            = args.envs,
                agent           = args.agent,
                representations = args.representations,      
                observations    = args.observations,
                seed            = args.seed,
                uuid            = args.uuid)
        experiment_manager.learn()