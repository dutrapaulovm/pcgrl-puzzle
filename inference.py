import pylab
from pcgrl.wrappers import *
from pcgrl import *
from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.MultiAgentEnv import *
from pcgrl.minimap.MiniMapEnv import MiniMapEnv 
from pcgrl.maze.MazeEnv import MazeEnv
from pcgrl.mazecoin.MazeCoinEnv import MazeCoinEnv
from pcgrl.zelda.ZeldaEnv import ZeldaEnv
from pcgrl.dungeon.DungeonEnv import DungeonEnv
from pcgrl.Utils import *

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from utils import *

def plot_bar(df, path, filename, title, steps):           
    
    total = np.array(df["Total"])
    error = total / steps

    labels = []
    
    for index, row in df.iterrows():
        labels.append(row['Representation']+'/'+row['Observation'])        
        
    plt.rcdefaults()
    fig, ax = plt.subplots()    
    
    y_pos = np.arange(len(total))
    hbars = ax.barh(y_pos, total, xerr=error,align='center')
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Steps')
    ax.set_title(title)  
    ax.bar_label(hbars) #, fmt='%.2f')
    ax.set_xlim(right=steps)
    fig.tight_layout()
    plt.savefig(path+"/"+filename+".png")        
    plt.close()

def plot_all_rewards(average, scores, episodes, path, filename, title):    
    
    linestyle_str = [
        ('solid', 'solid'),      # Same as (0, ()) or '-'
        ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
        ('dashed', 'dashed'),    # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]    
    
    linestyles = []
    for i, (name, linestyle) in enumerate(linestyle_tuple[::-1]):
        linestyles.append(linestyle)    
    
    fig, axs = plt.subplots()
    plt.figure(figsize=(18, 9))
    c = 0    
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]        
        #plt.plot(episodes[i][2], scores[i][2], linestyle=linestyles[i], label='Rewards: ' + label)      
        plt.plot(episodes[i][2], scores[i][2], label='Rewards: ' + label)      
        c += 2
            
    plt.title("Inference: " + title)
    plt.ylabel('Rewards', fontsize=18)
    plt.xlabel('Steps', fontsize=18) 
    plt.grid(True)
    plt.legend()  
    plt.savefig(path+"/"+filename+".png")    
    plt.close()
    
def plot_all_average(average, scores, episodes, path, filename, title):    
    
    linestyle_str = [
        ('solid', 'solid'),      # Same as (0, ()) or '-'
        ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
        ('dashed', 'dashed'),    # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]    
    
    #linestyles = []
    #for i, (name, linestyle) in enumerate(linestyle_tuple[::-1]):
    #    linestyles.append(linestyle)        
    
    fig, axs = plt.subplots()
    plt.figure(figsize=(18, 9))
    c = 0
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]
        #plt.plot(episodes[i][2], average[i][2], linestyle=linestyles[i], label='Average: ' + label)        
        plt.plot(episodes[i][2], average[i][2], label='Average: ' + label)        
        c += 2
    
    plt.title("Inference: " + title)
    plt.ylabel('Rewards Average', fontsize=18)
    plt.xlabel('Steps', fontsize=18) 
    plt.grid(True)
    plt.legend()  
    plt.savefig(path+"/"+filename+".png")    
    plt.close()    

def plot_all(average, scores, episodes, path, filename):
    
    fig, axs = plt.subplots()
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]
        plt.plot(episodes[i][2], average[i][2], label='Average: ' + label)
        plt.plot(episodes[i][2], scores[i][2],  label='Rewards: ' + label)    
    
    plt.ylabel('Rewards', fontsize=18)
    plt.xlabel('Steps', fontsize=18)        
    plt.title("Inference")
    plt.grid()
    plt.legend()
    plt.savefig(path+"/"+filename+".png")
    plt.close()

def plot(average, scores, episodes, path, filename, rep, game):
    pylab.figure(figsize=(18, 9))
    pylab.plot(episodes, average,'r', label='Average')
    pylab.plot(episodes, scores, 'b', label='Rewards')
    pylab.ylabel('Rewards', fontsize=18)
    pylab.xlabel('Steps', fontsize=18)        
    pylab.title(game + ", Representation: " + rep)
    pylab.grid()
    pylab.legend()
    pylab.savefig(path+"/"+filename+".png")

def run_inference(results_dir = "./results/",
                total_timesteps = 50000,              
                learning_rate: float = 2.5e-4, 
                n_steps:int   = 128,                                        
                policy_size  = [64, 64],                
                act_func = ActivationFunc.SIGMOID.value,                            
                agent = Experiment.AGENT_HHP.value,
                entropy_min:int = 1.80,
                envs = [Game.DUNGEON.value],
                representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
                observations = [WrappersType.MAP.value],              
                seed:int = 1000):    
    
    n_experiments   = 1        
    render          = True
    interation_path = 10000    
    
    RL_ALG          = "PPO"
    versions        = [agent]
    
    action_change = False

    save_image_level = True
    show_logger = False    
    show_hud = True

    #Parametros ambiente    
    max_changes            = 61
    use_done_max_changes   = True
    games = envs
    
    timesteps = [total_timesteps]    
    mlp_units = [policy_size[0]]      

    n_experiments = 1
   
    model_dir = "models"
        
    parent_dir = mk_dir(os.path.dirname(__file__), results_dir)        
    path_model_dir = parent_dir
    path_model_dir = mk_dir(path_model_dir, model_dir)
    
    if not os.path.isdir(path_model_dir):
        os.mkdir(path_model_dir)  
    
    for version in versions:
        
        action_change = False
        
        for t_time_s in timesteps:
            
            interation_path = t_time_s
            
            for mlp_u in mlp_units:        
                
                n_units = mlp_u
                n_layers  = 2    
        
                for name_game in games:

                    for par in range(n_experiments):
                        
                        main_dir = "Experiment 0" + str(par+1) + "-"+version+"-"+name_game+"-"+RL_ALG                        
                        
                        scores_inf, episodes_inf, average_inf = [], [], []       
                        all_time_elapsed                      = []
                        map_counter                           = []
                        
                        for _rep in representations:
                            
                            representation = _rep
                            rep_path = representation                            
                            for _obs in observations:
                                    
                                observation = _obs
                                rep_path   =  representation + "-" + observation                                
                                
                                
                                done = False                     

                                parent_dir = os.path.dirname(__file__) + "/results/"
                                dirname_experiments = "Inference-"+str(interation_path)+"-Steps"+str(n_steps)+"-L"+str(n_units)+"-E"+str(entropy_min)+"-LR"+str(learning_rate)+"SD"+str(seed)+act_func
                                path_experiments = os.path.join(parent_dir, dirname_experiments)
                                    
                                if not os.path.isdir(path_experiments):
                                    os.mkdir(path_experiments)
                                    
                                path_experiments = os.path.join(path_experiments, main_dir)
                                
                                path_part = path_experiments
                                
                                if not os.path.isdir(path_experiments):
                                    os.mkdir(path_experiments)      
                                    
                                path_experiments    = os.path.join(path_experiments, rep_path)
                                
                                if not os.path.isdir(path_experiments):
                                    os.mkdir(path_experiments)                                             
                        
                                path_best = os.path.join(path_experiments, "best")                            
                                if not os.path.isdir(path_best):
                                    os.mkdir(path_best)                                
                                    
                                path_worst = os.path.join(path_experiments, "worst")                            
                                if not os.path.isdir(path_worst):
                                    os.mkdir(path_worst)         
                                    
                                path_map = mk_dir(path_experiments, "map")
                                
                                mk_dir(path_map, "best")
                                
                                mk_dir(path_map, "worst")                                             
                                    
                                path_monitors_experiments = os.path.join(path_experiments, "monitors")                            
                                if not os.path.isdir(path_monitors_experiments):
                                    os.mkdir(path_monitors_experiments)                
                                
                                path_ppo = path_model_dir + "/"+representation + "-" + observation+"-"+str(interation_path)+"-Steps"+str(n_steps)+"-L"+str(n_units)+"-"+main_dir+"-E"+str(entropy_min)+"-LR"+str(learning_rate)+"SD"+str(seed)+act_func                                
                                
                                model = PPO.load(path_ppo)                            

                                if name_game == Game.MAZECOIN.value:             
                                    singleEnv = MazeCoinEnv(seed=seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, action_change=action_change)
                                    singleEnv.show_logger = show_logger
                                    singleEnv.use_done_max_changes = use_done_max_changes
                                    singleEnv.max_changes         = max_changes                                                                     
                                elif name_game == Game.MAZE.value:             
                                    singleEnv = MazeEnv(seed=seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, action_change=action_change)
                                    singleEnv.show_logger = show_logger
                                    singleEnv.use_done_max_changes = use_done_max_changes
                                    singleEnv.max_changes         = max_changes                                 
                                elif name_game == Game.DUNGEON.value:             
                                    singleEnv = DungeonEnv(seed=seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, action_change=action_change)
                                    singleEnv.show_logger = show_logger
                                    singleEnv.use_done_max_changes = use_done_max_changes
                                    singleEnv.max_changes         = max_changes   
                                elif name_game == Game.ZELDA.value:             
                                    singleEnv = ZeldaEnv(seed=seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, action_change=action_change)
                                    singleEnv.show_logger = show_logger
                                    singleEnv.use_done_max_changes = use_done_max_changes
                                    singleEnv.max_changes         = max_changes                                                                                     
                                elif name_game == Game.MINIMAP.value:             
                                    singleEnv = MiniMapEnv(seed=seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=True, action_change=action_change)
                                    singleEnv.show_logger = show_logger
                                    singleEnv.use_done_max_changes = use_done_max_changes
                                    singleEnv.max_changes         = max_changes 
                                
                                singleEnv.exp = version 
                                singleEnv.game.show_hud       = show_hud                                  
                                singleEnv.entropy_min = entropy_min                                
                                
                                env =  make_env(singleEnv, observation=observation)                                
                                exp = 1                                        
                                
                                env = RenderMonitor(env, exp, path_monitors_experiments, rrender=render)                 
                                
                                scores, episodes, average = [], [], []
                                
                                env = DummyVecEnv([lambda: env])
                                obs = env.reset()
                                done = False
                                start = timer()
                                print("Start: ", start)
                                
                                time_elapsed = []
                                total_rewards = []                                
                                sum_rewards = 0
                                steps = 0                                
                                
                                for e in range(total_timesteps):
                                    obs = env.reset()              
                                    done = False                
                                    env.render()           
                                    score = 0                                                
                                    steps = 0
                                    while not done:
                                        action, _states = model.predict(obs)                        
                                        obs, reward, done, info = env.step(action)
                                        score += reward[0]
                                        sum_rewards += reward[0]  
                                        steps += 1
                                        #total_average = (sum_rewards / steps)
                                        
                                        print()
                                        print("Representation: {}, Observation: {}".format(representation, observation))                                         
                                        print("Episode: {}, Score: {}, Total rewards: ".format(e, score, sum_rewards))                    
                                        #print("Average: {} ".format(total_average))
                                        
                                        if done:
                                            scores.append(score)
                                            episodes.append(e)
                                            average.append(sum(scores) / len(scores))                
                                            total_rewards.append(sum_rewards)
                                                                                        
                                        #Se não conseguiu atingir a construção do nível durante a inferência    
                                        #então houve falha no treinamento
                                        """"
                                        if steps > max_changes+1: 
                                            total_fail += 1
                                            done = True
                                            steps = 0
                                            columnsnames = {"count"}
                                            results_writer = ResultsWriter(
                                                filename="InferenceFail.csv",
                                                path=path_experiments,                 
                                                fieldsnames=columnsnames
                                            )                                                                            
                                            results_writer.write_row({"count" : total_fail})                                           
                                        """    
                                
                                map_counter.append({"Representation" : representation, "Observation" : observation, "Total" : singleEnv.counter_done})
                                
                                inference = {"Episodes": episodes, "Scores" : scores, "Average": average, "Total Rewards" : total_rewards}                
                                df = pad.DataFrame(inference)
                                df.to_csv(path_part+"/Inference"+_rep+"-"+_obs+".csv",  index=False)                
                                
                                #label="Ai({:.2f}), alpha({:.2f}), gamma({:.2f})".format(auxiliary_input, rew_alpha, rew_gamma)
                                plot(average, scores, episodes, path_experiments, "inference", _rep + ", Observation: " + observation, name_game)                                
                                end = timer()        
                                print("End: ", end)
                                time_ela = timedelta(seconds=end-start)
                                print("Time elapsed: ", time_ela)
                                
                                d = {"Representation": representation, "Observation" : _obs, "start": start, "end" : end, "time elapsed": time_ela}
                                
                                time_elapsed.append(d)                
                                all_time_elapsed.append(d)
                                                
                                df = pad.DataFrame(time_elapsed)
                                df.to_csv(path_experiments+"/Time elapsed.csv",  index=False)                            
                                            
                                scores_inf.append((_rep, _obs, scores))
                                episodes_inf.append((_rep, _obs, episodes))
                                average_inf.append((_rep, _obs, average))                

                                #game.quit()        
                        
                        title = name_game
                        
                        df = pad.DataFrame(all_time_elapsed)
                        df.to_csv(path_part+"/Time elapsed-"+title+".csv",  index=False)                              
                        
                        df = pad.DataFrame(map_counter)
                        df.to_csv(path_part+"/MapCounter-"+title+".csv",  index=False)
                        info_mlp = "MLP: Units {}, Layers {}".format(n_units, n_layers)                        
                        t = "Quantidade de Mapas Gerados\n"+title+" - "+info_mlp 
                        
                        plot_bar(df, path_part, "Mapcounter-"+title, t, total_timesteps)
                        
                        plot_all_rewards(average_inf, scores_inf, episodes_inf, path_part, "inference-rewards-all-"+title, title+" - "+info_mlp)
                        plot_all_average(average_inf, scores_inf, episodes_inf, path_part, "inference-average-all-"+title, title+" - "+info_mlp)

#if __name__ == '__main__':
#    run_inference()