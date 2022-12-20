import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab

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

def plot_moving_average_all_rewards(rewards_games, path, filename):
    
    plt.figure(figsize=(16, 5))
    moving_average_num = 50
    markers = ['o', 's', 'd', 'x', 'h']
    m = 1    
    for key, value in rewards_games.items():
        rewards = value
        # moving average        
        def moving_average(x, n=moving_average_num):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[n:] - cumsum[:-n]) / float(n)

        # plotting
        #plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
        #plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
        #plt.subplot(21+str(m))
        plt.plot(moving_average(rewards), label=key)
        m += 1
    
    
    #plt.title(f'PPO moving average of last {moving_average_num} episodes')
    #plt.xlabel('Episode')
    #plt.ylabel('Reward')
    plt.title(f'Média móvel dos últimos {moving_average_num} episódios')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensas')    
    plt.legend()
    plt.grid(True)    
    plt.savefig(path+"/"+filename+".png")    
    plt.close()

def plot_moving_average_agents_all_rewards(rewards_games, path, filename):
    
    plt.figure(figsize=(16, 5))
    moving_average_num = 10
    for key, value in rewards_games.items():
        rewards = value
        # moving average        
        def moving_average(x, n=moving_average_num):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[n:] - cumsum[:-n]) / float(n)

        # plotting
        #plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
        #plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
        plt.plot(moving_average(rewards), label=key)
    
    
    plt.title(f'PPO moving average of last {moving_average_num} episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)    
    plt.savefig(path+"/"+filename+".png")    
    plt.close()
    
def plot_average_rewards(rewards, path, filename, game_name):
    
    plt.figure(figsize=(16, 5))
    # moving average
    moving_average_num = 10
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'PPO on {game_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)    
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
    
def plot_simulate(alphas, gammas, epsilons, rewards):
    """
    Imprime o gráfico com os resultados da média de recompenas
    """
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)

    for eps in range(len(epsilons)-2):
        ep = '$\epsilon = %.02f$' % (epsilons[eps])
        ap = '$\\alpha = %.02f$' % (alphas[eps])
        gm = '$\gamma = %.02f$' % (gammas[eps])
        plt.plot(rewards[eps], label=ep+', '+ap+', '+gm)

    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')    
    plt.legend()

    plt.subplot(2, 1, 2)    
    for eps in range(2, len(epsilons)):
        ep = '$\epsilon = %.02f$' % (epsilons[eps])
        ap = '$\\alpha = %.02f$' % (alphas[eps])
        gm = '$\gamma = %.02f$' % (gammas[eps])
        plt.plot(rewards[eps], label=ep+', '+ap+', '+gm)

    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')    
    plt.legend()

    plt.savefig('./results/SimulateResults.png')
    plt.show()
    plt.close()    

def plot_penalties(path, episodes, results, title):    
    xSteps = np.array(episodes)    
    yData = np.array(results)        

    plt.grid(True)

    plt.title(title)       
    plt.plot(xSteps, yData, label="Rewards(Q-Learning)")
    
    #ep = '$\epsilon = %.02f$' % (epsilon)
    #ap = '$\\alpha = %.02f$' % (alpha)
    #gm = '$\gamma = %.02f$' % (gamma)
    #label=ep+', '+ap+', '+gm              
    label = ""    
    
    plt.legend(['T - ' + label], loc=9)
    plt.xlabel('Episodes')
    plt.ylabel('Penalties')    

    #plt.show()
    plt.savefig(path)
    plt.close()      