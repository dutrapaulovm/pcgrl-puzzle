import argparse
import uuid
from pcgrl import *
from pcgrl.BasePCGRLEnv import Experiment
from utils import ActivationFunc
from testplotter import run_plotter
from pcgrl.wrappers import *

if __name__ == "__main__":
        
        uuids = ["-21-(2, 3)","-61-(2, 3)","-180-(2, 3)","-21-(2, 4)","-61-(2, 4)","-180-(2, 4)"]
        entropies = [2.25, 2.25, 2.25, 2.75, 2.75, 2.75]
        #uuids = ["-61-(2, 4)"]
        #entropies = [2.75]        
        #uuids = ["-21-(2, 3)","-61-(2, 3)","-180-(2, 3)"]
        #entropies = [2.75, 2.75, 2.75]
        #uuids = ["-21-(2, 4)","-61-(2, 4)","-180-(2, 4)"]
        #uuids = ["-21-(2, 4)","-61-(2, 4)","-180-(2, 4)"]
        #uuids = ["-180-(2, 3)"]
        #entropies = [2.25, 2.25, 2.25]        
        #entropies = [2.75, 2.75, 2.75]        
        #entropies = [2.25]
        #uuids = ["-21-(4, 2)","-61-(4, 2)","-180-(4, 2)"]
        #entropies = [2.75, 2.75, 2.75]        
        #uuids = ["-180-(4, 2)"]
        #entropies = [2.75]        
        #uuids = ["-21-(3, 2)","-61-(3, 2)","-180-(3, 2)"]
        #entropies = [2.25, 2.25, 2.25]        
        #boards = [ [2,3]  ]
        boards = [ [2,3],[2,3] ,[2,3], [2,4],[2,4] ,[2,4]   ]
        #boards = [ [2,3],[2,3] ,[2,3] ]
        #boards = [ [2,4] ]
        #tag = "reward_neighbors_reward_distance-V2"
        #tag = "reward_neighbors-V2"
        #tag = "-V2"
        #tag = "reward_neighbors_reward_distance"
        #tag = "reward_neighbors"
        tag = ""
        for el in range(len(uuids)):
                
                #env = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value]
                #env = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value]
                env = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value, Game.ZELDALOWMAPS.value]
                #env = [Game.ZELDALOWMAPS.value]                

                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HHPD.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPD.value]
                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value]
                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value]
                agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPEX.value]
                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPEX.value]                

                parser = argparse.ArgumentParser()
                #parser.add_argument("--results_dir", default="./results/")
                #parser.add_argument("--results_dir", default="D:/ResultsDissertacao/old/")
                parser.add_argument("--results_dir", default="F:/Experimentos/pcgrlpuzzle-results/")
                #parser.add_argument("--results_dir", default="F:/Experimentos/pcgrlpuzzle-results-threshold/")
                parser.add_argument("--total_timesteps", default = 100000, type=int)        
                #parser.add_argument("--total_timesteps", default = 50000, type=int)        
                #parser.add_argument("--total_timesteps", default = 20000, type=int)        
                #parser.add_argument("--total_timesteps", default = 30000, type=int)        
                #parser.add_argument("--learning_rate", default = 2.5e-4, type=float)
                #parser.add_argument("--learning_rate", default = 1e-4, type=float)
                parser.add_argument("--learning_rate", default = 3e-4, type=float)
                parser.add_argument("--n_steps", default = 2048, type=int)
                #parser.add_argument("--n_steps", default = 4096, type=int)
                #parser.add_argument("--n_steps", default = 512, type=int)
                #parser.add_argument("--n_steps", default = 128, type=int)
                parser.add_argument("--n_epochs", default = 10, type=int)        
                parser.add_argument("--batch_size", default = 64, type=int)        
                parser.add_argument("--activation", default=ActivationFunc.SIGMOID.value, type=str)
                #parser.add_argument("--activation", default=ActivationFunc.Tanh.value, type=str)
                parser.add_argument("--entropy_min", default = entropies[el], type=float)
                parser.add_argument("--board", default = boards[el], type=float)
                
                #parser.add_argument("--entropy_min", default = 2.75, type=float)
                #parser.add_argument("--entropy_min", default = 2.25, type=float)
                #parser.add_argument("--entropy_min", default = 1.80, type=float)        
                #parser.add_argument("--entropy_min", default = 2.5, type=float)        
                #parser.add_argument("--entropy_min", default = 1.91, type=float)
                parser.add_argument("--n_inference", default = 1000, type=float)
                parser.add_argument("--envs", default=env, type=str, nargs='*')
                parser.add_argument("--agent", default=agents, type=str, nargs='*')
                #parser.add_argument("--representations", default=[Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value], type=str, nargs='*')
                #parser.add_argument("--representations", default=[Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value], type=str, nargs='*')       
                parser.add_argument("--representations", default=[Behaviors.NARROW_PUZZLE.value], type=str, nargs='*')       
                parser.add_argument("--observations", default=[WrappersType.SEGMENT.value], type=str, nargs='*')
                #parser.add_argument("--observations", default=[WrappersType.MAP.value], type=str, nargs='*')
                #parser.add_argument("--seed", type=int, default=1000)        
                #parser.add_argument("--seed", type=int, default=2291215719)        
                #parser.add_argument("--seed", type=int, default=1771007676)        
                #parser.add_argument("--seed", type=int, default=1997508302)        
                #parser.add_argument("--seed", type=int, default=4058299762)                
                parser.add_argument("--seed", type=int, default=42)                
                parser.add_argument("--language", type=str, default="pt-br")                
                #parser.add_argument("--uuid", default="-00V1400-SEGMENT-1x64-SHARE", type=str)                                
                #parser.add_argument("--uuid", default="21-3x2-V002", type=str)     
                #parser.add_argument("--uuid", default="180-3x2-V003", type=str)     
                #parser.add_argument("--uuid", default="-4x2-V001", type=str)                                
                #parser.add_argument("--uuid", default="21-4x2-V002", type=str)     
                #parser.add_argument("--uuid", default="180-4x2-V003", type=str)                  
                #parser.add_argument("--uuid", default="-21-(3, 2)", type=str)  
                #parser.add_argument("--uuid", default="-61-(3, 2)", type=str)  
                #parser.add_argument("--uuid", default="-180-(3, 2)", type=str)                
                parser.add_argument("--uuid", default=uuids[el], type=str)   
                parser.add_argument("--tag", default=tag, type=str)
                                
                args = parser.parse_args()

                run_plotter(results_dir = args.results_dir,
                        total_timesteps = args.total_timesteps,
                        learning_rate   = args.learning_rate,        
                        n_steps         = args.n_steps,                
                        batch_size      = args.batch_size,
                        n_epochs        = args.n_epochs,
                        act_func        = args.activation,
                        entropy_min     = args.entropy_min,
                        n_inference     = args.n_inference,
                        envs            = args.envs,
                        agents          = args.agent,
                        representations = args.representations,      
                        observations    = args.observations,
                        seed            = args.seed,
                        uuid            = args.uuid,
                        language        = args.language,
                        board           = args.board,
                        tag             = args.tag)