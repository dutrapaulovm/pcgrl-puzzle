from Generator import *
from Utils import *
from scipy.special import rel_entr
import matplotlib.pyplot as plt
def kl_divergence_(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == "__main__":
    
    #agents = ["SS", "HHP", "HEQHP"]
    agents = ["HEQHP"]
    games  = ["Dungeon"]
    data = []

    for agent in agents:
        for game in games:
            path = os.path.abspath(os.path.join("resultados", os.pardir))
            path = os.path.join(path, "resultados\\{}\\{}\\map\\".format(agent, game))
            for file in os.listdir(path):
                print(file)
                if file[-3:] in {'csv'}:
                    filepath = os.path.join(path, file)
                    generatorA = Generator(path=filepath, piece_size=(8, 8), loadmap=True, border=True)
                    js_sum = 1    
                    for m in range(generatorA.count()-1, 0, -1):
                        q = generatorA.get_piece(m)        
                        for n in range(generatorA.count()):
                            if (m != n):
                                p = generatorA.get_piece(n)                    
                                js = js_divergence(p, q)
                                #js_divergence(p, q)
                                js_sum += js                                
                                #js_sum = 1 / round(js_sum, 2)
                    #print(js_sum / generatorA.count())                    
                    #print(js_sum)                    
                    #time.sleep(1)
                    print(js_sum)
                    data.append(js_sum)
    print()
    #data = [1, 1, 1, 1, 1]    
    #data.append(100000)
    #norm = np.linalg.norm(np.array(data)) #normalize(np.array(data))
    #data = data / norm
    normalized =  normalize(np.array(data.copy()))
    print(normalized)
    fig, axs = plt.subplots()
    bins = 100 #len(normalized)
    axs.hist(normalized, bins, density=True, histtype='barstacked', rwidth=0.8)
    fig.tight_layout()
    plt.show()