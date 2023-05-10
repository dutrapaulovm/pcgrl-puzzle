from Generator import *
from Utils import *
#from scipy.special import rel_entr
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
def calcTileProb(lv, fh=4, fw=4):
    mp = []
    h, w = lv.shape
    for i in range(h-fh+1):
        for j in range(w-fw+1):
            k = (lv[i:i+fh, j:j+fw]).flatten() #tuple((lv[i:i+fh, j:j+fw]).flatten())
            mp.append(k)
            #mp[k] = (mp[k]+1) if (k in mp.keys()) else 1
    return mp

def calKLFromMap(mpa, mpb, w=0.5, eps=0.001):
    result = 0
    keys = set(mpa.keys()) | set(mpb.keys())
    suma = sum([mpa[e] for e in mpa.keys()])
    sumb = sum([mpb[e] for e in mpb.keys()])
    #print(suma)
    #print(sumb)
    #print(keys)
    for e in keys:
        a = ((eps + mpa[e]) / (suma + len(keys) * eps)) if (e in mpa.keys()) else (eps / (suma + len(keys) * eps))
        b = ((eps + mpb[e]) / (sumb + len(keys) * eps)) if (e in mpb.keys()) else (eps / (sumb + len(keys) * eps))
        result += w * a * math.log2(a / b) + (1 - w) * b * math.log2(b / a)
    return result

def kl_divergence_(p, q, epsilon: float = 1e-8): 
    
    p = np.array(p).flatten()
    q = np.array(q).flatten()

    p = p+epsilon
    q = q+epsilon

    #P = torch.Tensor(p)
    #Q = torch.Tensor(q)

    #(P * (P / Q).log()).sum()
    # tensor(0.0863), 10.2 µs ± 508

    #return F.kl_div(Q.log(), P, None, None, 'sum')        
             
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def dkl(p, q, w = 0.5, epsilon: float = 1e-8):
    
    p = np.array(p)
    p = p.flatten()      
    
    q = np.array(q)
    q = q.flatten()
        
    #p = p+epsilon
    #q = q+epsilon

    return -(w * kl_divergence_(p, q) + (1 - w) * kl_divergence_(q, p))

if __name__ == "__main__":
    
    # this is the same example in wiki
    PP = [0.36, 0.48, 0.16]
    QQ = [0.36, 0.48, 0.16]
    #QQ = [0.333, 0.333, 0.333]
    #P = torch.Tensor(PP)
    #Q = torch.Tensor(QQ)

    #(P * (P / Q).log()).sum()
    # tensor(0.0863), 10.2 µs ± 508

    #print(F.kl_div(Q.log(), P, None, None, 'sum')    )
    
    print(dkl(np.array(PP), np.array(QQ)))
    #time.sleep(5)
    
    #agents = ["SS", "HHP", "HEQHP"]
    agents = ["HEQHP"]
    games  = ["Dungeon"]
    data = []

    for agent in agents:
        for game in games:
            path = os.path.abspath(os.path.join("resultados", os.pardir))
            path = os.path.join(path, "resultados\\{}\\{}\\".format(agent, game))
            for file in os.listdir(path):
                print(file)
                if file[-3:] in {'csv'}:
                    filepath = os.path.join(path, file)
                    generatorA = Generator(path=filepath, piece_size=(8, 8), loadmap=True, border=True)
                    js_sum = 0.001  
                    js_hist = []  
                    sn = 0
                    for m in range(generatorA.count()-1, 0, -1):
                        p = calcTileProb(generatorA.get_piece(m))      
                        #print(p)
                        #time.sleep(5)
                        for n in range(generatorA.count()):
                            if (m != n):
                                qq = generatorA.get_piece(n)                                                         
                                q = calcTileProb(generatorA.get_piece(n))
                                js = dkl(p, q)                                 
                                js_sum += js                                  
                                js_hist.append(js)                                
                                
                                #js_sum = 1 / round(js_sum, 2)
                        sn += 1
                    #print(js_sum / generatorA.count())                    
                    #print(js_sum)                    
                    #time.sleep(1)
                    #print(js_sum)
                    print("\t Normalidez", normalize(np.array(js_hist.copy())).sum())
                    print("\t", js_sum)
                    data.append(js_sum)
    print()
    #data = [1, 1, 1, 1, 1]    
    #data.append(100000)
    #norm = np.linalg.norm(np.array(data)) #normalize(np.array(data))
    #data = data / norm
    """
    normalized =  normalize(np.array(data.copy()))
    print(normalized)
    fig, axs = plt.subplots()
    bins = 100 #len(normalized)
    axs.hist(normalized, bins, density=True, histtype='barstacked', rwidth=0.8)
    fig.tight_layout()
    plt.show()
    """