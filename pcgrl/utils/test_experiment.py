from pcgrl.utils.experiment import *
import torch as th

if __name__ == '__main__':

    rl_algo = RlAlgo.PPO.value
    act_func = ActivationFunc.SIGMOID.value
    n_experiment    = 1
    total_timesteps = 100000
    learning_rate   = 3e-4
    n_steps         = 2048 #Horizon, see paper PPO               
    gamma           = 0.99
    batch_size      = 64
    n_epochs        = 10
    max_changes     = [61, 180]
    #max_changes     = [21, 61, 180]
    time_steps_inference     = 1000
    entropy_min              = None
    reward_best_done_bonus   = 50
    reward_medium_done_bonus = 10
    reward_low_done_bonus    = 0
    reward_entropy_penalty   = 0
    reward_change_penalty    = -0.1
    #boards = [(2, 3), (2, 4)]
    boards = [(2, 3)]
    #path_results = "F:\Experimentos\Results-Reward_threshold"
    path_results = "F:\Experimentos\Resultados-5Experimentos"
    #seeds = [42, 43, 44, 45, 46]  
    #seeds = [42]
    seeds = [396297772, 1267813114, 1293202981, 2031146959, 2967016284]
    current_seed  = 0
    factor_reward = 1
    render        = False
    show_hud      = False
    show_logger   = False
    record_video  = False
    action_change = False
    action_rotate = False
    env_rewards   = False
    save_level    = False
    piece_size    = 8
    is_training   = True
    is_inference  = True
    plot_results_experiments = False

    #for n in range(5):
    #    seed = gen_random_number()
    #    seeds.append(seed)
    
    policy = "MlpPolicy"
    if policy == "MlpPolicy":
        policy_kwargs = dict(net_arch = [64, 64], activation_fn=th.nn.Sigmoid)
    elif policy == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class = CustomCNNV2,
            features_extractor_kwargs = dict(features_dim=512),
        )    

    for b in boards:
        board = b
        if board[0] == 2 and board[1] == 3:
            rewards_threshold = {Experiment.AGENT_SS.value      : 2.3  * factor_reward, 
                                Experiment.AGENT_HHP.value      : 2.3  * factor_reward, 
                                Experiment.AGENT_HEQHP.value    : 7.9  * factor_reward, 
                                Experiment.AGENT_HEQHPEX.value  : 7.9  * factor_reward, 
                                Experiment.AGENT_HEQHPD.value   : 79.0 * factor_reward}
        elif board[0] == 2 and board[1] == 4:
            rewards_threshold = {Experiment.AGENT_SS.value      : 2.8  * factor_reward, 
                                Experiment.AGENT_HHP.value      : 2.8  * factor_reward, 
                                Experiment.AGENT_HEQHP.value    : 8.3  * factor_reward, 
                                Experiment.AGENT_HEQHPEX.value  : 8.3  * factor_reward, 
                                Experiment.AGENT_HEQHPD.value   : 83.0 * factor_reward}

        #rewards_threshold = []
        for m_changes in max_changes:

            uuid = "-{}-{}".format(m_changes,board)

            for seed in seeds:
                representations = [Behaviors.NARROW_PUZZLE.value]
                observations = [WrappersType.SEGMENT.value]                            

                #envs = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value]                    
                envs = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDALOWMAPS.value]                                    
                #envs = [Game.ZELDA.value]                                    
                #envs = [Game.ZELDALOWMAPS.value]
                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value]#, Experiment.AGENT_HEQHPD.value]
                agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value]
                #agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPEX.value]
                #agents = [Experiment.AGENT_HEQHP.value]                
                if is_training:
                    for a in agents:
                        for e in envs:
                            experiment_manager = ExperimentManager(policy                = policy,
                                                                learning_rate            = learning_rate, 
                                                                results_dir              = path_results, 
                                                                n_steps                  = n_steps, 
                                                                agent                    = [a], 
                                                                envs                     = [e],
                                                                representations          = representations,
                                                                observations             = observations,    
                                                                board                    = board,
                                                                reward_best_done_bonus   = reward_best_done_bonus,
                                                                reward_medium_done_bonus = reward_medium_done_bonus,
                                                                reward_low_done_bonus    = reward_low_done_bonus,   
                                                                reward_entropy_penalty   = reward_entropy_penalty,     
                                                                reward_change_penalty    = reward_change_penalty,                                                                                                                                                                                                        
                                                                entropy_min              = entropy_min,                                     
                                                                total_timesteps          = total_timesteps, 
                                                                piece_size               = piece_size,
                                                                factor_reward            = factor_reward,
                                                                action_change            = action_change,
                                                                action_rotate            = action_rotate,
                                                                max_changes              = m_changes,
                                                                env_rewards              = env_rewards,                                                    
                                                                show_logger              = show_logger,
                                                                gamma                    = gamma, 
                                                                act_func                 = act_func, 
                                                                rl_algo                  = rl_algo, 
                                                                n_epochs                 = n_epochs, 
                                                                batch_size               = batch_size,                                                                                                        
                                                                policy_kwargs            = policy_kwargs, 
                                                                uuid                     = uuid, 
                                                                seed                     = seed,
                                                                rewards_threshold=rewards_threshold)
                            experiment_manager.learn(use_function_set_random_seed=True, render=render, save_image_level=save_level, show_hud=show_hud)
                if is_inference:
                    for a in agents:
                        for e in envs:
                            experiment_manager = ExperimentManager(learning_rate = learning_rate, 
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
                                                                piece_size               = piece_size,
                                                                action_change            = action_change,   
                                                                action_rotate            = action_rotate,
                                                                max_changes              = m_changes,                                                                
                                                                env_rewards              = env_rewards,
                                                                total_timesteps = total_timesteps, 
                                                                factor_reward = factor_reward, 
                                                                show_logger = show_logger,                                                   
                                                                gamma = gamma, act_func=act_func, rl_algo = rl_algo, 
                                                                n_epochs=n_epochs, batch_size=batch_size,
                                                                policy_kwargs = policy_kwargs, uuid = uuid, seed = seed)
                            experiment_manager.inference(time_steps=time_steps_inference, use_function_set_random_seed=True, seeds=[seed], render=render, record_video=record_video, show_hud=show_hud, n_experiments=n_experiment)
                if plot_results_experiments:
                    for a in agents:
                        for e in envs:
                            experiment_manager = ExperimentManager(learning_rate = learning_rate, 
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
                                                                piece_size               = piece_size,
                                                                action_change            = action_change,   
                                                                action_rotate            = action_rotate,
                                                                max_changes              = m_changes,                                                                
                                                                env_rewards              = env_rewards,
                                                                total_timesteps = total_timesteps, 
                                                                factor_reward = factor_reward, 
                                                                show_logger = show_logger,                                                   
                                                                gamma = gamma, act_func=act_func, rl_algo = rl_algo, 
                                                                n_epochs=n_epochs, batch_size=batch_size,
                                                                policy_kwargs = policy_kwargs, uuid = uuid, seed = seed)                            
                            experiment_manager.plot_result_experiments()