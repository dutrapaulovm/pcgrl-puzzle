# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import time     
import random
from tqdm import tqdm
from enum import Enum

def plot_experiments(df, df_params, path, title):
        
    experiment = np.array(df["experiment"])
    best  = np.array(df["best"])
    worst = np.array(df["worst"])          

    layer = np.array(df_params["layer"])
    units = np.array(df_params["units"])
    rew_alpha = np.array(df_params["rew_alpha"])              
    rew_gamma = np.array(df_params["rew_gamma"])                  
    auxiliary_input = np.array(df_params["auxiliary_input"])                  
    representation = np.array(df_params["representation"])                  
    observation = np.array(df_params["observation"])           
    total_timesteps = np.array(df_params["total_timesteps"])
    n_steps = np.array(df_params["n_steps"])
    learning_rate = np.array(df_params["learning_rate"])

    fig, ax = plt.subplots()            
            
    ap = 'alpha = %.02f' % (rew_alpha[0])
    gm = 'gamma = %.02f' % (rew_gamma[0])
    ai = 'Ai = %.02f' % (auxiliary_input[0])            
    rep = 'Rep = {}'.format(representation[0])           
    obs = 'Obs = {}'.format(observation[0])             
    n_steps = 'n_steps = {}'.format(n_steps[0])             
    total_timesteps = 'Total Timesteps = {}'.format(total_timesteps[0])             
    learning_rate = 'Learning rate = {}'.format(learning_rate[0])             
    title= rep+', '+obs+', '+ai+', '+ap+', '+gm + '\n'+total_timesteps + ', ' + n_steps + ', ' + learning_rate
                    
    ax.bar(experiment, worst, label="Worst: {}".format(worst.sum()))
    ax.bar(experiment, best, label="Best: {}".format(best.sum()), bottom=worst)
    
    ax.legend()
    ax.grid()
    plt.ylabel('Quantidade de mapas gerados')
    plt.xlabel('Episódios') 
    plt.title(title)        
    
    plt.savefig(path)    
    #plt.show()
    plt.close()

if __name__ == '__main__':
    
    path = os.path.dirname(__file__) + "/results/experiments/"
    print(path)
    os.listdir(path)
    include = {'Parte 01', 'Parte 02', 'Parte 03', 'Parte 04'}
        
    params = ""
    experiment_monitor = ""
    # read all images in PATH, resize and write to DESTINATION_PATH
    title = ""
    print("Gerando gráficos dos experimentos")
    for subdir in os.listdir(path):
        if subdir in include:            
            current_path = os.path.join(path, subdir) 
            
            for path_rep in os.listdir(current_path):
                title = path_rep
                path_rep = os.path.join(current_path, path_rep)                                
                params = ""
                experiment_monitor = ""                
                
                for file in os.listdir(path_rep):
                    
                    if file[:] in {'ExperimentMonitor.csv'}:
                        experiment_monitor = file
                    if file[:] in {'params.csv'}:
                        params = file
                    
                    if params == 'params.csv' and experiment_monitor == 'ExperimentMonitor.csv':
                
                        path_savefile = os.path.join(path_rep, "ExperimentMonitor.png")
                        
                        file_experiment = os.path.join(path_rep, experiment_monitor)                                
                        df_experiment = pd.read_csv(file_experiment, usecols= ['experiment', 'best', 'worst', 'time'], index_col=False)
                        
                        file_experiment = os.path.join(path_rep, params)                                
                        df_params = pd.read_csv(file_experiment, index_col=False)                        
                        
                        plot_experiments(df_experiment, df_params, path_savefile, title)      
                        experiment_monitor = ""                                 
                        params = ""
                                            
    print("Geração dos gráficos finalizada")                                                                      