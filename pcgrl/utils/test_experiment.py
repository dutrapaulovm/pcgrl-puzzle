from pcgrl.utils.experiment import *
import torch as th

if __name__ == '__main__':

    rl_algo = RlAlgo.PPO.value
    act_func = ActivationFunc.SIGMOID.value        
    total_timesteps = 1000
    learning_rate   = 3e-4
    n_steps         = 2048 #Horizon, see paper PPO               
    gamma           = 0.99
    batch_size      = 64
    n_epochs        = 10
    max_changes     = 61
    time_steps_inference     = 1000
    entropy_min              = None
    reward_best_done_bonus   = 50
    reward_medium_done_bonus = 10
    reward_low_done_bonus    = 0
    reward_entropy_penalty   = 0
    reward_change_penalty    = -0.1
    board = (1, 8)
    path_results = "F:\Experimentos\Results-new-Factor10"
    seeds = [42]  
    uuid = "-{}-{}".format(max_changes,board)
    factor_reward = 1
    render        = False
    show_hud      = False
    record_video  = False
    action_change = True
    action_rotate = False
    env_rewards   = False
    save_level    = False
    piece_size    = 8

    policy_kwargs = dict(net_arch = [64, 64], activation_fn=th.nn.Sigmoid)

    for seed in seeds:
        representations = [Behaviors.NARROW_PUZZLE.value]
        observations = [WrappersType.SEGMENT.value]                
        #envs = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value]          
        envs = [Game.SMB.value]          
        agents = [Experiment.AGENT_HEQHP.value]
        #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value]
        
        for a in agents:
            for e in envs:
                experiment_manager = ExperimentManager(#policy=policy,
                                                    learning_rate = learning_rate, 
                                                    results_dir = path_results, n_steps = n_steps, 
                                                    agent = [a], envs = [e],
                                                    representations = representations,
                                                    observations = observations,    
                                                    board = board,
                                                    reward_best_done_bonus   = reward_best_done_bonus,
                                                    reward_medium_done_bonus = reward_medium_done_bonus,
                                                    reward_low_done_bonus    = reward_low_done_bonus,   
                                                    reward_entropy_penalty   =  reward_entropy_penalty,     
                                                    reward_change_penalty    = reward_change_penalty,                                                                                                                                                                                                        
                                                    entropy_min              = entropy_min,                                     
                                                    total_timesteps          = total_timesteps, 
                                                    piece_size = piece_size,
                                                    factor_reward = factor_reward,
                                                    action_change=action_change,
                                                    action_rotate=action_rotate,
                                                    max_changes=max_changes,
                                                    env_rewards = env_rewards,                                                    
                                                    gamma = gamma, act_func=act_func, rl_algo = rl_algo, 
                                                    n_epochs=n_epochs, batch_size=batch_size,                                                                                                        
                                                    policy_kwargs = policy_kwargs, uuid = uuid, seed = seed)
                experiment_manager.learn(use_function_set_random_seed=True, render=render, save_image_level=save_level, show_hud=show_hud)
        
        for a in agents:
            for e in envs:
                experiment_manager = ExperimentManager(learning_rate = learning_rate, 
                                                    results_dir = path_results, n_steps = n_steps, 
                                                    agent = [a], envs = [e],
                                                    representations = representations,
                                                    observations = observations,      
                                                    board = board,                                                    
                                                    reward_best_done_bonus = reward_best_done_bonus,
                                                    reward_medium_done_bonus = reward_medium_done_bonus,
                                                    reward_low_done_bonus = reward_low_done_bonus,   
                                                    reward_entropy_penalty =  reward_entropy_penalty,    
                                                    reward_change_penalty = reward_change_penalty,                                                                                                                                    
                                                    entropy_min = entropy_min,  
                                                    piece_size = piece_size,
                                                    action_change=action_change,   
                                                     action_rotate=action_rotate,
                                                    max_changes=max_changes,                                                                
                                                    env_rewards = env_rewards,
                                                    total_timesteps = total_timesteps, 
                                                    factor_reward = factor_reward,                                                    
                                                    gamma = gamma, act_func=act_func, rl_algo = rl_algo, 
                                                    n_epochs=n_epochs, batch_size=batch_size,
                                                    policy_kwargs = policy_kwargs, uuid = uuid, seed = seed)
                experiment_manager.inference(time_steps=time_steps_inference, use_function_set_random_seed=True, seeds=[seed], render=render, record_video=record_video, show_hud=show_hud)        