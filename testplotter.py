from tkinter import E
from unicodedata import decimal
from results_plot import *
from expressive_range_plot import *
import os
import math
from matplotlib.ticker import ScalarFormatter
from pcgrl import *
from pcgrl.BasePCGRLEnv import *
from pcgrl.Utils import cum_mean
import pandas as pd
from pcgrl.log import ResultsWriter
from pcgrl.wrappers import WrappersType
from utils import * 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.tri as tri
import numpy as np


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


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"

class PlotResults:
    
    def __init__(self) -> None:                
        self.data = {}

    def add(self, key, value):
        self.data[key] = value
        
    def clear(self):        
        self.data = {}
        
    def addv2(self, key, key2, value):
        
        if key not in self.data:                    
            self.data[key] = []                        
        self.data[key].append({ key2 : value})          

    def __dataframe_format(self, df,  field, decimals = 2):
        l = np.round_(np.array(df[field]), decimals = decimals)
        df = pd.DataFrame()  
        df = pd.DataFrame({field : l})        
        
        return df

    def plot_stackbarvert(self, title, path, filename, n = 1000, entropy_min = 1.80):
        
        labels = [key for key in self.data]       
        
        best  = []
        best_std  = []
        
        worst = []
        worst_std = []        
        #category_names = ['H >= 1.80', ' H < 1.80']
        category_names = ['H >= {}'.format(entropy_min), ' H < {}'.format(entropy_min)]
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            if 'entropy' in df.columns:      

                df = self.__dataframe_format(df, "entropy")
                
                dtbest  = df.where(df["entropy"] >= entropy_min)
                dtworst = df.where((df["entropy"] >= 0.0) & (df["entropy"] < entropy_min))
                
                pctbest = round(float((dtbest["entropy"].count() / n) * 100), 2)
                pctworst = round(float((dtworst["entropy"].count() / n) * 100), 2)
                
                best.append(pctbest)
                best_std.append(dtbest["entropy"].std())
                
                worst.append(pctworst)
                worst_std.append(dtworst["entropy"].std())
                
            else:
                best.append(0)
                worst.append(0)  
                best_std.append(0)                             
                worst_std.append(0)                
        
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()        
        graph_best = ax.bar(labels, best, width, label='H >= {}'.format(entropy_min))
        graph_worst = ax.bar(labels, worst, width,  bottom=best , label='H < {}'.format(entropy_min))
        i = 0
        for p in graph_best:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
            plt.text(x+width/2,
                    y+height*0.10,
                    str(best[i])+'%',
                    ha='center',
                    weight='bold')
            i += 1
            
        i = 0 
        for p in graph_worst:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
                    
            plt.text(x+width/2,
                    y+height*1.01,
                    str(worst[i])+'%',
                    ha='center',
                    weight='bold')
                
            i += 1            

        ax.set_ylabel('% of Levels generated')
        ax.set_title(title)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')
        
        plt.title("Inference: " + title)
        plt.grid(True)        
        plt.savefig(path+"/"+filename+".pdf")    
        plt.savefig(path+"/"+filename+".png")    
        plt.close()
    
    def plot_contour(self, title, path, filename):
        
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            if 'entropy' in df.columns:        
                npts = df["entropy"].count()
                ngridx = npts
                ngridy = npts
                x = np.array(df["entropy"])#np.random.uniform(0, 3, npts)
                y = np.array(df["entropy"]) #np.random.uniform(0, 3, npts)
                #z = x * np.exp(-x**3 - y**3)                
                #fig, (ax1, ax2) = plt.subplots(nrows=2)
                fig, ax1 = plt.subplots()

                # -----------------------
                # Interpolation on a grid
                # -----------------------
                # A contour plot of irregularly spaced data coordinates
                # via interpolation on a grid.

                # Create grid values first.
                xi = np.linspace(0, 3, ngridx)
                yi = np.linspace(0, 3, ngridy)

                
                ax1.plot(x, y, 'ko', ms=3)
                ax1.set(xlim=(0, 3), ylim=(0, 3))
                ax1.set_title('grid and contour (%d points, %d grid points)' %
                            (npts, ngridx * ngridy))
                
                plt.title(title)
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(path+"/"+filename+".pdf")    
                plt.savefig(path+"/"+filename+".png")    
    
    def plot_stackbar(self, title, path, filename, n = 1000):
        
        labels = [key for key in self.data]       
        best  = []
        worst = []
        results = { }
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            if 'entropy' in df.columns:                    
                
                df = self.__dataframe_format(df, "entropy")
                
                dtbest  = df.where(df["entropy"] >= 1.80)
                dtworst = df.where((df["entropy"] >= 0.0) & (df["entropy"] < 1.80))
                pctbest = round(float((dtbest["entropy"].count() / n) * 100), 2)
                pctworst = round(float((dtworst["entropy"].count() / n) * 100), 2)
                
                results[key] = [pctbest, pctworst]
            else:
                results[key] = [0, 0]
                
        category_names = ['H >= 1.80', ' H < 1.80']

        def survey(results, category_names):
            """
            Parameters
            ----------
            results : dict
                A mapping from question labels to a list of answers per category.
                It is assumed all lists contain the same number of entries and that
                it matches the length of *category_names*.
            category_names : list of str
                The category labels.
            """
            labels = list(results.keys())
            data = np.array(list(results.values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.colormaps['RdYlGn'](
                np.linspace(0.15, 0.85, data.shape[1]))

            fig, ax = plt.subplots(figsize=(9.2, 5))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                rects = ax.barh(labels, widths, left=starts, height=0.5, label=colname)
                r, g, b, _ = color
                text_color = 'black' if r * g * b < 0.5 else 'darkgrey'
                ax.bar_label(rects, label_type='center', color=text_color)
                
            ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')

            return fig, ax                

        fig, ax = survey(results, category_names)
        
        plt.title("Inference: " + title)
        plt.grid(True)        
        plt.savefig(path+"/"+filename+".pdf")    
        plt.savefig(path+"/"+filename+".png")    
        plt.close()

    def plot_bar_segmentused(self, title, path, filename):
        category_names = ['H >= 1.80', ' H < 1.80']
        labels = [key for key in self.data]       

        segments_games = {Game.ZELDA.value : 300,                           
                          Game.ZELDALOWMAPS.value : 150,
                          Game.MINIMAP.value : 300, 
                          Game.MAZECOIN.value : 240, 
                          Game.MAZECOINLOWMAPS.value : 80,
                          Game.MAZE.value : 50, 
                          Game.SMB.value : 120, 
                          Game.DUNGEON.value:40}
        segments = []      
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            aux_segments = [] 
            if "segments" in df:     
                segm = df["segments"]            
                for s in segm:                
                    sg = s.replace("[", "")
                    sg = sg.replace("]", "")
                    segm1 = sg.split()        
                    segm1 = np.array(segm1).astype(int)                                         
                    for s1 in segm1:
                        aux_segments.append(s1)                        
            else:
                aux_segments.append(0)
                
            dt_segments = pd.DataFrame({"segments" : aux_segments})                                   
            dt_segments = dt_segments.groupby(['segments'])['segments'].count()                            
            rows = dt_segments.shape[0]             
            
            total_segments = segments_games[key]
            
            pct = (rows / total_segments) * 100
            segments.append(round(pct,2))
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        #plt.figure(figsize=(16,5))
        graph_best = ax.bar(x - width/2, segments, width, label='H >= 1.80')        

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('% of Segments used')
        ax.set_ylabel('% de Segmentos utilizados')
        ax.set_title('Segments used by game')
        ax.set_xticks(x, labels)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)
                
        i = 0
        for p in graph_best:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
            plt.text(x+width/2,
                    y+height*1.01,
                    str(segments[i])+'%',
                    ha='center',
                    weight='bold')
            i += 1           

        fig.tight_layout()
        #plt.title("Inference: " + title)
        plt.grid(True)
        #plt.legend()  
        plt.savefig(path+"/"+filename+".pdf")    
        plt.savefig(path+"/"+filename+".png")    
        plt.close()
    
    def plot_entropy_dist(self):
        
        entropies = []      
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)                                                   
            if 'entropy' in df.columns:                    
                
                df = self.__dataframe_format(df, "entropy")
                
                ent = df["entropy"]                        
                dt_entropy = pd.DataFrame({"entropy" : ent})                                   
                values = np.array(dt_entropy['entropy'])                
                entropies = numpy.append(entropies, values, axis = None)                        
        
        df = pd.DataFrame({"entropy" : entropies})
        df = df.groupby(['entropy'])['entropy'].count().reset_index(name='counts')        
        yentropia = np.array(df['entropy'])                        
    
    def plot_levels_repeted(self, title, path, filename, n = 1000):
        labels = [key for key in self.data]       
        
        levels = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)                            
            if "segments" in df:     
                segm = df["segments"]            
                levels_count = segm.count()
                dt_segments = pd.DataFrame({"segments" : segm})                                   
                dt_segments = dt_segments.groupby(['segments'])['segments'].count().reset_index(name='counts')                                                               
                #time.sleep(10)
                total  = dt_segments.where(dt_segments["counts"] > 1).count()
                total  = np.array(total["counts"]).sum()
                total = float((total / levels_count) * 100)                                
                levels.append(round(total, 2))
            else:
                levels.append(0)
                
        maxlevels = np.max(levels)
        
        y = np.linspace(0, maxlevels, round(12 / 2))
        x = np.arange(len(labels))     
        y = np.arange(maxlevels)     
        fig, ax = plt.subplots()        
        graph_best = ax.bar(x, levels)        
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('% of Levels generated')
        #ax.set_ylabel('% de cenários repetidos')
        #ax.set_xlabel("Agentes")
        ax.set_xlabel("Agents")
        #ax.set_title('Número de repetições')
        #ax.set_title('Entropy by game')
        ax.set_xticks(x, labels)        
        if (maxlevels == 0):
            ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        #ax.legend(bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')
        
        i = 0
        for p in graph_best:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
            plt.text(x+width/2,
                    y+height*1.01,
                    str(levels[i])+'%',
                    ha='center',
                    weight='bold')
            i += 1        
                
        #plt.title(title)
        plt.tight_layout()
        plt.grid(True)
        #plt.legend()  
        plt.savefig("{}/{}.pdf".format(path, filename))
        plt.savefig("{}/{}.png".format(path, filename))    
        plt.close()

    def plot_segment_used(self, path, filename, game):
        
        labels = [key for key in self.data]       
        fontsize = 12
        segments_games = {Game.ZELDA.value : 300,                           
                          Game.ZELDALOWMAPS.value : 150,
                          Game.MINIMAP.value : 300, 
                          Game.MAZECOIN.value : 240, 
                          Game.MAZECOINLOWMAPS.value : 80,
                          Game.MAZE.value : 50, 
                          Game.SMB.value : 120, 
                          Game.DUNGEON.value:40}
        segments = []      
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            aux_segments = [] 
            if "segments" in df:     
                segm = df["segments"]            
                for s in segm:                
                    sg = s.replace("[", "")
                    sg = sg.replace("]", "")
                    segm1 = sg.split()        
                    segm1 = np.array(segm1).astype(int)                                         
                    for s1 in segm1:
                        aux_segments.append(s1)                        
            else:
                aux_segments.append(0)
                
            dt_segments = pd.DataFrame({"segments" : aux_segments})                                   
            dt_segments = dt_segments.groupby(['segments'])['segments'].count()                            
            rows = dt_segments.shape[0]             
            
            total_segments = segments_games[game]
            
            pct = (rows / total_segments) * 100
            segments.append(round(pct,2))
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        #plt.figure(figsize=(16,5))
        graph_segments = ax.bar(x - width/2, segments, width)        

        best_percentiles = [self.format(p) + "%" if p >=0 else '' for p in  segments]

        ax.bar_label(graph_segments, best_percentiles,
                            padding=0, color='black', fontweight='bold', fontsize=fontsize)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('% of Segments used', fontsize = 12)
        #ax.set_ylabel('% de Segmentos utilizados', fontsize = 12)
        #ax.set_xlabel('Agentes', fontsize = 12)
        ax.set_xlabel('Agents', fontsize = 12)
        #ax.set_title('Segments used by game', fontsize = 12)
        ax.set_xticks(x, labels, fontsize = 12)        
        
        
        """
        i = 0
        for p in graph_best:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
            plt.text(x+width/2,
                    y+height*1.01,
                    str(segments[i])+'%',
                    ha='center',
                    weight='bold')
            i += 1           
        """        

        fig.tight_layout()
        #plt.title("Inference: " + title)
        plt.grid(True)
        #plt.legend()  
        plt.savefig(path+"/"+filename+".pdf")    
        plt.savefig(path+"/"+filename+".png")    
        plt.close()
                
    def format(self, value):
        value = "%.02f" % (value)        
        value = str(value).replace(".", ",")
        return value

    def plot(self, title, path, filename, n = 1000, entropy_min = 1.80):
        category_names = ['H >= 1.80', 'H < 1.80']
        labels = [key for key in self.data]       
        best  = []
        worst = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)           
            if 'entropy' in df.columns:                    
                
                df = self.__dataframe_format(df, "entropy")
                
                dtbest  = df.where(df["entropy"] >= entropy_min)
                dtworst = df.where((df["entropy"] >= 0.0) & (df["entropy"] < entropy_min))
                pct = float((dtbest["entropy"].count() / n) * 100)                
                best.append(round(pct,2))
                pct = float((dtworst["entropy"].count() / n) * 100)
                worst.append(round(pct,2))
            else:
                best.append(0)
                worst.append(0)
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fontsize = 12
        fig, ax = plt.subplots()
        #plt.rcParams['figure.figsize'] = (10,10)
        #plt.figure(figsize=(16,5))  

        graph_best = ax.bar(x - width/2, best, width, label='H' + r'$\geq$' + '{}'.format(self.format(entropy_min)))
        graph_worst = ax.bar(x + width/2, worst, width, label='H < {}'.format(self.format(entropy_min)))

        #rects = ax.barh(y_pos, porcent_entropy, xerr=error, align='center', height=0.5)
        #rects = ax.barh(y_pos, porcent_entropy, align='center', height=0.6)

        #percentiles = [str(p) + "%" for p in porcent_entropy]
        #ax.bar_label(rects, percentiles, padding=5, color='black', fontweight='bold', fontsize=fontsize)

        best_percentiles = [self.format(p) + "%" if p >=0 else '' for p in  best]
        worst_percentiles = [self.format(p) + "%" if p >=0 else '' for p in  worst]
        #small_percentiles = [str(p) + "%" if p <= 35 else '' for p in  best]
        #ax.bar_label(graph_best, small_percentiles,
         #                   padding=5, color='black', fontweight='bold', fontsize=fontsize)
        ax.bar_label(graph_best, best_percentiles,
                            padding=0, color='black', fontweight='bold', fontsize=fontsize)
        ax.bar_label(graph_worst, worst_percentiles,
                            padding=0, color='black', fontweight='bold', fontsize=fontsize)


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('% of Levels generated')
        #ax.set_ylabel('% de cenários gerados', fontsize=fontsize)
        #ax.set_xlabel('Agentes', fontsize=fontsize)
        ax.set_xlabel('Agents', fontsize=fontsize)
        #ax.set_title('Entropia por ambiente')
        #ax.set_title('Entropy by game')
        ax.set_xticks(x, labels, fontsize=fontsize)
        #ax.legend(ncol=len(category_names), loc='best', fontsize='small')
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize=fontsize)
        #ax.legend(ncol=len(category_names), fontsize='small')

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)
        """
        i = 0
        for p in graph_best:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
            plt.text(x+width/2,
                    y+height*1.01,
                    str(best[i])+'%',
                    ha='center',
                    weight='bold', fontsize=fontsize)
            i += 1
            
        i = 0 
        for p in graph_worst:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()            
                    
            plt.text(x+width/2,
                    y+height*1.01,
                    str(worst[i])+'%',
                    ha='center',
                    weight='bold', fontsize=fontsize)
                
            i += 1                           
        """
        plt.tight_layout()
        #plt.title("Inference: " + title)
        #plt.title("Resultados inferência: " + title)
        #plt.title(title)
        plt.grid(True)
        #plt.legend()  
        plt.savefig(path+"/"+filename+".pdf")    
        plt.savefig(path+"/"+filename+".png")    
        plt.close()
    
    def plot_entropy(self, path, filename, title, entropy_min = 1.80, c = 0):
        
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
        
        colors = ['orange', 'blue', 'green', 'red'] #['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']        
        
        games = [key for key in self.data]       
        
        """
        for key, value in self.data.items():                 
            levels = []
            df = pd.DataFrame(value)                       
            if 'entropy' in df.columns:                    
                levels.append(np.array(df["entropy"]))  
            else:
                levels.append(np.array([]))      
            
            if len(levels) > 0:
                entropy_group = df.groupby(['entropy']).size().reset_index(name='quantidade') 
                qtd_entropy = entropy_group['quantidade']
                entropy    = entropy_group['entropy']   
                    
                nbins = len(entropy)
                histogram, bins = np.histogram(levels, bins=nbins)
                pdf = (histogram / (sum(histogram)))
                bin_centers = (bins[1:] + bins[:-1]) * 0.5
                fig, ax = plt.subplots()                       
                plt.title("PDF - Entropy")
                plt.ylabel("Valores de probabilidade")
                plt.xlabel("Entropia dos níveis")
                plt.ylim(0, max(pdf)+0.1)
                plt.plot(bin_centers, pdf, label="PDF")
                plt.grid()
                plt.savefig(path+"/PDF-"+filename+"-"+key+".pdf")        
                plt.close()          
        
        """
        levels = []        
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)                       
            if 'entropy' in df.columns:        
                df = self.__dataframe_format(df, "entropy")            
                levels.append(np.array(df["entropy"]))  
            else:
                levels.append(np.array([]))           
        
        for i in range(len(games)):
            fig, ax = plt.subplots(figsize=(10, 5))                       
            levels[i] = np.sort(levels[i])

            x = np.arange(0, len(levels[i]))                    

            if (len(levels[i]) > 0):
                avg = 'avg = %.02f' % (np.round_(levels[i].mean(), decimals = 2))
                std = 'std = %.02f' % (np.round_(levels[i].std(), decimals = 2))
                med = 'med = %.02f' % (np.round_(np.median(levels[i]), decimals = 2))
            else:
                avg = 'avg = %.02f' % (0)
                std = 'std = %.02f' % (0)   
                med = 'med = %.02f' % (0)

            avg = avg.replace(".", ",")
            std = std.replace(".", ",")
            med = med.replace(".", ",")
            info ='{},{} e{}'.format(avg, med, std)
            label=r"$H_q = {:.2f}$".format(entropy_min, info).replace(".", ",")
            #print("Games: {}".format(games))
            #print("\t x: {}, levels {} ".format( len(x), len(levels[i]) ))
            #print(len(x))
            #np.round_(np.array(df["entropy"]), decimals = 2)
            #ax.axvline(np.mean(levels[i]), color="k", linestyle="dashed", linewidth=3, label="Avg : {:.2f}".format(avg))                                    
            ax.axhline(entropy_min, color="k", linestyle="dashed", linewidth=3, label=label)            
            #ax.plot(0, entropy_min, 'k--', linewidth=1.5, label='Theoretical')
            #plt.plot(x, levels[i], color='#ff000a', label='{}\n{}'.format(avg, std) )                 
            plt.plot(x, levels[i], linewidth=1.5, color=colors[i] )                 
            #ax.legend(bbox_to_anchor=(0, 1),loc='lower left')
            plt.legend(fontsize=14)            
            plt.grid(True)            
            #plt.xlabel('Levels Generated')
            plt.xlabel('Níveis gerados', fontsize=16)
            plt.ylabel('Entropia', fontsize=16)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            #plt.ylabel('Entropy levels')
            #stitle='Entropy: ' + games[i] + " - " + title
            stitle='Entropia: ' + games[i] + " - " + title
            plt.tight_layout()
            #plt.title(stitle)
            plt.savefig(path+"/"+filename+"-"+games[i]+".pdf")        
            plt.savefig(path+"/"+filename+"-"+games[i]+".png")             
            plt.close()
                                    
    def plot_scatter(self, path, filename, title, yfield, xfield, xlabel, ylabel):               
        
        games = [key for key in self.data]       
        entropy = []        
        changes =[]
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)   
            
            entropy.append(np.array(df[yfield]))           
            changes.append(np.array(df[xfield]))           
        
        for i in range(len(games)):
            fig, ax = plt.subplots()                       
            
            x = np.arange(0, len(entropy[i]))                    
            
            plt.scatter(changes[i], entropy[i])   
            #plt.xticks(range(0, len(changes[i]))) #range(min(changes[i]), max(changes[i])+1))             
            #plt.legend()            
            plt.grid(True)            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            stitle='{} x  {}: '.format(xlabel, ylabel) + games[i] + " - " + title
            plt.title(stitle)
            plt.savefig(path+"/"+filename+"-"+games[i]+".pdf")    
            plt.savefig(path+"/"+filename+"-"+games[i]+".png")              
            #print(path+"/"+filename+"-"+games[i]+".pdf")
                
            plt.close()            
            
    def plot_entropyv2(self, path, filename, title):
        
        colors = ['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']        
        
        linestyles = []
        for i, (name, linestyle) in enumerate(linestyle_tuple[::-1]):
            linestyles.append(linestyle)           
        
        games = [key for key in self.data]       
        levels = []        
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)   
            if 'entropy' in df.columns:      
                df = self.__dataframe_format(df, "entropy")              
                levels.append(np.array(df["entropy"]))  
            else:
                levels.append(np.array([]))
                
        
        x = np.arange(0, len(levels[0]))       
        fig, ax = plt.subplots()
                
        for i in range(len(games)):                        

            if (len(levels[i]) > 0):
                avg = 'avg = %.05f' % (levels[i].mean())
                std = 'std = %.05f' % (levels[i].std())
            else:
                avg = 'avg = %.05f' % (0)
                std = 'std = %.05f' % (0)                            
            
            plt.plot(x, levels[i], color=colors[i], linestyle=linestyles[i], label='{}\n{}'.format(avg, std) )
            
            
        plt.legend()            
        plt.grid(True)            
        plt.xlabel('Levels Generated')
        plt.ylabel('Entropy levels')
        stitle='Entropy: ' + games[i] + " - " + title
        plt.title(stitle)
        plt.savefig(path+"/"+filename+"-"+games[i]+".pdf")        
        plt.savefig(path+"/"+filename+"-"+games[i]+".png")                
        plt.close()            
                        
    def plot_boxplot(self, path, filename, filename2, title, xlabel = "Games", entropy_min = 1.80):
        
        games = [key for key in self.data]          
        levels = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)   
            if 'entropy' in df.columns:                    
                
                if len(np.array(df["entropy"]) > 0):
                    df = self.__dataframe_format(df, "entropy")
                    levels.append(np.round_(np.array(df["entropy"]), decimals = 2))  
                else:                    
                    levels.append(np.array([0]))                  
            else:                
                levels.append(np.array([0]))                
                    
        data = levels

        fig, ax = plt.subplots()                 
        
        #stitle='Entropy: ' + title
        #stitle='Entropia: ' + title
        stitle=title
                
        bplot = ax.boxplot(data)         
                            #,vert=True,  # vertical box alignment
                            #patch_artist=True,  # fill with color
                            #      )


        #colors = ['pink', 'lightblue', 'lightgreen']
        #idx = 0
        #for box in bplot['boxes']:
        #     box.set(facecolor = colors[idx] )
        #     idx += 1
        
        if len(data) > 0:
            if (len(data[0]) > 0) or (len(data[1]) > 0):            
                ax.violinplot(data)
        
                                    
        ax.set_xticklabels(games, fontsize=16)
        #ax.set_title(stitle)
        ax.set_ylabel("Entropy Levels", fontsize=18)
        #ax.set_ylabel("Entropia dos cenários", fontsize=16)
        #ax.set_ylabel("Entropia")
        ax.set_xlabel("Agents", fontsize=18)
        #ax.set_xlabel("Agentes", fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #ax.set_xlabel(xlabel)
        ax.grid()
        plt.tight_layout()
        #plt.show()
        plt.savefig(path+"/"+filename+".pdf")                  
        plt.savefig(path+"/"+filename+".png")                          
        plt.close()
        """
        fig, ax = plt.subplots()

        for i in range(len(levels)):                                  
            levels[i] = np.sort(levels[i])
            x = np.arange(0, len(levels[i]))   
 
            ecdf_event1 = sm.distributions.ECDF(levels[i])            
            x1 = ecdf_event1.x            
            y1 = ecdf_event1.y

            ax.step(x1,  y1, label=games[i])

        plt.legend(fontsize=14)            
        plt.grid(True)            

        ax.set_xlabel("Entropia dos cenários", fontsize=16)
        ax.set_ylabel("% de distribução", fontsize=16)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.tight_layout()        
        plt.savefig(path+"/"+filename2+".pdf")        
        plt.savefig(path+"/"+filename2+".png")             
        plt.close()   
        """

        fig, ax = plt.subplots(figsize=(10, 5))                      

        label=r"$H_q = {:.2f}$".format(entropy_min).replace(".", ",")
        ax.axhline(entropy_min, color="k", linestyle="dashed", linewidth=3, label=label)
        
        for i in range(len(games)):            
            levels[i] = np.sort(levels[i])
            x = np.arange(0, len(levels[i]))                                            

            plt.plot(x, levels[i], linewidth=1.5, label=games[i] )              

        plt.legend(fontsize=14)            
        plt.grid(True)                        
        plt.xlabel('Níveis gerados', fontsize=16)
        plt.ylabel('Entropia', fontsize=16)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    
        plt.tight_layout()            
        plt.savefig(path+"/"+filename2+"-V2.pdf")        
        plt.savefig(path+"/"+filename2+"-V2.png")             
        plt.close()

    def plot_boxplot_complete(self, path, filename, xlabel, title, envs = None):
        
        games = envs.copy()             
            
        exp = [key for key in self.data]          

        
        data_boxplot = []
        
        for key in exp:
            data = self.data[key]
            levels = {}
            print("\tTamanho dados: ", len(data))
            print("\tGames: ", games)
            for g in range(len(data)):                                 
                _game = games[g]
                df = data[g]                                                                        
                df = df[_game]                
                levels[_game] = 0    
                print("Key", key)             
                if 'entropy' in df.columns:
                    if len(np.array(df["entropy"]) >= 0):                        
                        df = self.__dataframe_format(df, "entropy")
                        entropies = np.array(df["entropy"])        
                        data_boxplot.append(entropies)

        fig, ax = plt.subplots()                 
        
        #stitle='Entropy: ' + title
        #stitle='Entropia: ' + title
        stitle=title
        bplot = ax.boxplot(data_boxplot)  

        #bplot = ax.boxplot(data_boxplot)  
        #if len(data_boxplot) > 0:
        #    if (len(data_boxplot[0]) > 0) or (len(data_boxplot[1]) > 0):            
        #        ax.violinplot(data_boxplot)
        
                                    
        ax.set_xticklabels(exp, fontsize=16)
        #ax.set_title(stitle)
        ax.set_ylabel("Entropy Levels", fontsize=18)
        #ax.set_ylabel("Entropia dos cenários", fontsize=16)
        #ax.set_ylabel("Entropia")
        ax.set_xlabel("Agents", fontsize=18)
        #ax.set_xlabel("Agentes", fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #ax.set_xlabel(xlabel)
        ax.grid()
        plt.tight_layout()
        #plt.show()
        plt.savefig(path+"/"+filename+".pdf")                  
        plt.savefig(path+"/"+filename+".png")                          
        plt.close()                        
        
    
    def plot_boxplot_counter_changes(self, path, filename, title, xlabel = "Agentes"):
        
        games = [key for key in self.data]          
        levels = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)   
            if 'counter_changes' in df.columns:                    
                
                if len(np.array(df["counter_changes"]) > 0):
                    levels.append(np.array(df["counter_changes"]))  
                else:                    
                    levels.append(np.array([0]))                  
            else:                
                levels.append(np.array([0]))                
                    
        data = levels
        fig, ax = plt.subplots()                 
        
        #stitle='Changes: ' + title
        #stitle='Alterações: ' + title
        stitle = title
                
        ax.boxplot(data)        
        if len(data) > 0:
            if (len(data[0]) > 0) or (len(data[1]) > 0):            
                ax.violinplot(data)
                                    
        ax.set_xticklabels(games)
        #ax.set_title(stitle)
        #ax.set_ylabel("Number of Changes")
        ax.set_ylabel("Quantidade de Alterações")
        ax.set_xlabel(xlabel)
        ax.grid()
        plt.tight_layout()
        plt.savefig(path+"/"+filename+".pdf")                  
        plt.savefig(path+"/"+filename+".png")         
        
        plt.close()        
        
        
    def plot_similarity_levels(self, path, filename):
        print("Generated Similarity")        
        
        games = [Game.ZELDA.value, Game.ZELDALOWMAPS.value, Game.MINIMAP.value, Game.MAZECOIN.value,  Game.MAZE.value, Game.SMB.value, Game.DUNGEON.value]                    
        #games = [Game.DUNGEON.value]                    
        
        exp = [key for key in self.data]
        colors_name=['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']                                        
        dt = {}
        for key in exp:   
            data = self.data[key]            
            print("\tTamanho dados: ", len(data))
            index_colors = 0
            for g in range(len(data)):                                         
                _game = games[g]
                print("Game: {}".format(_game))
                df = data[g]                                                        
                df = df[_game]                
                print("Key", key)
                levelIndex = 0
                segments = {}          
                segments = {}                
                x, y = [], []
                if "segments" in df:
                    segm = df["segments"]                                
                    for s in segm:
                        sg = s.replace("[", "")
                        sg = sg.replace("]", "")
                        segm1 = sg.split()
                        segm1 = np.array(segm1).astype(int)                                                                 
                        segments[levelIndex] = segm1                        
                        levelIndex += 1                
                
                for j in range(0, len(segments)):
                    x.append(j)
                    y.append(entropy(segments[j]))                    
                
                plt.figure(figsize=(16,8))
                plt.plot(x, y, color=colors_name[index_colors])
                index_colors += 1     
                
                plt.legend()                    
                plt.xlabel('Levels')
                plt.ylabel('Similarity beween levels')
                title = 'Similarity of levels of ' + _game + ", Agent : " + key                 
                plt.title(title)                                         
                plt.savefig(path+"/"+filename+"-"+_game+"-"+key+".pdf")
                plt.savefig(path+"/"+filename+"-"+_game+"-"+key+".png")                
                plt.close()                           
                
        """
        for key in exp:   
            data = self.data[key]            
            print("\tTamanho dados: ", len(data))            
            for g in range(len(data)):                                 
                _game = games[g]
                print("Game: {}".format(_game))
                df = data[g]                                                        
                df = df[_game]                
                print("Key", key)
                levelIndex = 0
                segments = {}                
                if "segments" in df:
                    segm = df["segments"]                                
                    for s in segm:
                        sg = s.replace("[", "")
                        sg = sg.replace("]", "")
                        segm1 = sg.split()
                        segm1 = np.array(segm1).astype(int)                                                                 
                        segments[levelIndex] = segm1                        
                        levelIndex += 1
                else:                    
                    segm1 = np.array([])              
                    segments = {}
                    segments[levelIndex] = segm1
                    levelIndex += 1
                
                levels = [3]
                colors_name=['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']
                
                for i in range(0, 5): #len(segments)):
                    x, y = [], []
                    xs, ys = [], []
                    plt.figure(figsize=(16,8))
                    for j in range(0, len(segments)):
                        x.append(j)
                        js = js_divergence(segments[i], segments[j])
                        jsn = 0
                        if (js > 0):
                            jsn = math.pi / math.sqrt(js + 0.5)
                        
                        y.append(sigmoid(jsn))
                        if (i == j):
                            plt.annotate("Level %i, JS %.02f" % (i+1, jsn)   , xy=(i, jsn), xytext=(i+1, jsn), fontsize=12 )
                        elif (jsn >= 4.0):                            
                            xs.append(j)
                            ys.append(sigmoid(jsn))
                            #plt.annotate("Level %i, JS %.02f" % (j+1, jsn), xy=(j, jsn), xytext=(j+1, jsn), fontsize=12)
                                                                                    
                        print("{}, {}, S1={}, S2={}, {} = {}".format(segments[i], segments[j], sigmoid(jsn), sigmoid(js), jsn, js ))
                    
                    #plt.scatter(xs, ys, label='JS >= 4.0', color=colors_name[0])
                    #plt.plot(x, y, color=colors_name[0])                                        
                    markerline, stemlines, baseline = plt.stem(
                    x, y, linefmt='grey', markerfmt='D', bottom=0.9)
                    markerline.set_markerfacecolor('none')
                    #plt.legend(levels)
                    plt.legend()                    
                    plt.xlabel('Levels')
                    plt.ylabel('Similarity beween level ' + str(i+1) + " and other levels")
                    title = 'Similarity of levels of ' + _game + ", Agent : " + key                 
                    plt.title(title)                                         
                    plt.savefig(path+"/"+filename+"-"+_game+"-"+key+"-Level"+str(i+1)+".pdf")
                    plt.close()
                    #time.sleep(0.5)                          
                    
        """                    
                    
    def csv_comparative(self, path, filename, n = 1000, envs = None, entropy_min = 1.80):
        
        info = ["EntropyQTotal", "EntropyRelativeSuccess", "EntropyQ", "EntropyMinMax", "EntropiaAvgMean", "EntropyMean", "EntropyMedian", "EntropyStd", "EntropyVar", "ChangeMeanMedian", "ChangeMean", "ChangeAll", "ChangeMedian", "ChangeStd", "ChangeMedian", "ChangeVar", "Levels", "SegmentsMean", "SegmentsStd", "SegmentsMedian", "SegmentsVar", "SegmentsUsed", "LevelsRepeated"]
        #info = ["LevelsRepeated"]
        
        segments_games = {Game.ZELDA.value : 300,                           
                          Game.ZELDALOWMAPS.value : 150,
                          Game.MINIMAP.value : 300, 
                          Game.MAZECOIN.value : 240, 
                          Game.MAZECOINLOWMAPS.value : 80, 
                          Game.MAZE.value : 50, 
                          Game.SMB.value : 120, 
                          Game.DUNGEON.value:40}
        aux_envs = envs.copy()
        aux_envs.append("Agent")
        for info in info:
            if envs is None:
                games = [Game.ZELDA.value, Game.ZELDALOWMAPS.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZECOINLOWMAPS.value, Game.MAZE.value, Game.SMB.value, Game.DUNGEON.value, "Agent"]
            else:
                games = aux_envs                
                
            results_writer = None
            exp = [key for key in self.data]

            columnsnames = games
            if path is not None:
                results_writer = ResultsWriter(
                    filename="{}-{}.csv".format(info, filename),
                    path=path,                 
                    fieldsnames=columnsnames
                )
            else:
                results_writer = None

            for key in exp:   
                data = self.data[key]                                      
                levels = {}
                print("\tTamanho dados: ", len(data))
                for g in range(len(data)):                 
                    _game = games[g]
                    print(_game)
                    df = data[g]                                                        
                    df = df[_game]
                    levels[_game] = 0    
                    print("Key", key)
                    if (info == "EntropyQTotal"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                
                                df = self.__dataframe_format(df, "entropy")
                                
                                entropies = np.array(df["entropy"])        
                                best = len(entropies[entropies >= entropy_min])                        
                                worst = len(entropies[entropies < entropy_min])                                                        
                                count = "%i$-$%i" % (worst, best)                                                 
                                count = count.replace(".", ",")
                                levels[_game] = count
                            else:                    
                                levels[_game] = "-"  
                    if (info == "EntropyRelativeSuccess"):
                        
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)
                            
                        dt_segments = pd.DataFrame({"segments" : segments})                                   
                        dt_segments = dt_segments.groupby(['segments'])['segments'].count()                            
                        rows = dt_segments.shape[0]             
                        
                        total_segments = segments_games[_game]                        
                        
                        if 'entropy' in df.columns:                            
                            if len(np.array(df["entropy"]) >= 0):
                                
                                df = self.__dataframe_format(df, "entropy")
                                
                                entropies = np.array(df["entropy"])        
                                total = len(entropies)
                                best = len(entropies[entropies >= entropy_min])                        
                                worst = len(entropies[entropies < entropy_min])                                                        
                                
                                pct = (best / total) * 100                
                                best = round(pct,2)
                                pct = (worst / total) * 100
                                worst = round(pct,2)          
                                
                                segments_used= (rows / total_segments) * 100
                                
                                count = "%i$-$%.02f$-$%.02f$-$%.02f" % (total, segments_used, worst, best)                                                 
                                #count = "{}%$-${}%".format(worst, best)                                                 
                                #count = "{}$-${}".format(worst, best)                                                 
                                count = count.replace(".", ",")
                                levels[_game] = count
                            else:                    
                                levels[_game] = "-"                                                                                                                           
                    if (info == "EntropyQ"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                
                                df = self.__dataframe_format(df, "entropy")
                                
                                entropies = np.array(df["entropy"])        
                                best = len(entropies[entropies >= entropy_min])                        
                                worst = len(entropies[entropies < entropy_min])                                                        
                                
                                pct = (best / n) * 100                
                                best = round(pct,2)
                                pct = (worst / n) * 100
                                worst = round(pct,2)
                                
                                count = "%.02f$-$%.02f" % (worst, best)                                                 
                                #count = "{}%$-${}%".format(worst, best)                                                 
                                #count = "{}$-${}".format(worst, best)                                                 
                                count = count.replace(".", ",")
                                levels[_game] = count
                            else:                    
                                levels[_game] = "-"                                                                       
                    elif (info == "EntropyMinMax"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):   
                                
                                df = self.__dataframe_format(df, "entropy")

                                entropies = np.round_(np.array(df["entropy"]), decimals = 2)
                                entropies = np.round_(entropies, decimals = 2)                                                

                                #entropies = np.array(df["entropy"])
                                emax = np.max(entropies)
                                emin = np.min(entropies)
                                cmin = len(entropies[entropies == emin])
                                cmax = len(entropies[entropies == emax])                                
                                
                                l = len(np.array(df))

                                cmin = (cmin / l) * 100
                                cmax = (cmax / l) * 100
                                
                                if (emax == emin):
                                    #minmax = "%.02f(%.02f%s)" % (emax, cmax, "%")                                                 
                                    minmax = "%.02f(%.02f)" % (emax, cmax)                                                 
                                else:
                                    #minmax = "%.02f(%.02f%s)$-$%.02f(%.02f%s)" % (emin, cmin, "%", emax, cmax, "%")                                                 
                                    minmax = "%.02f(%.02f)$-$%.02f(%.02f)" % (emin, cmin, emax, cmax)                 
                                    minmax = minmax.replace(".", ",")
                                levels[_game] = minmax
                            else:                    
                                levels[_game] = "-"                                                  
                    elif (info == "EntropyMean"):
                        if 'entropy' in df.columns:
                            
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")    
                                levels[_game] = np.array(df["entropy"]).mean()  
                            else:                    
                                levels[_game] = 0
                    elif (info == "EntropyMedian"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                levels[_game] = np.median(np.array(df["entropy"]))
                            else:                    
                                levels[_game] = 0
                    elif (info == "EntropyStd"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                levels[_game] = np.array(df["entropy"]).std()  
                            else:                    
                                levels[_game] = 0
                    elif (info == "EntropyVar"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                levels[_game] = np.array(df["entropy"]).var()  
                            else:                    
                                levels[_game] = 0                                                                
                    elif (info == "ChangeAll"):                                
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                counter_changes = np.array(df["counter_changes"])
                                mean = counter_changes.mean()  
                                median = np.median(counter_changes)
                                std  = counter_changes.std()                                                                  
                                var = counter_changes.var()
                                #levels[_game] = "%.02f$-$%.02f$-$%.02f$-$%.02f" % (mean, median, std, var)
                                levels[_game] = "%.02f$-$%.02f$-$%.02f" % (mean, median, std)
                                levels[_game] = levels[_game].replace(".", ",")
                            else:                    
                                levels[_game] = 0
                    elif (info == "ChangeMeanMedian"):                                
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                counter_changes = np.array(df["counter_changes"])
                                tam = len(counter_changes)
                                mean = counter_changes.mean()  
                                median = np.median(counter_changes)
                                std  = counter_changes.std()      
                                levels[_game] = "%i$-$%.02f$-$%.02f" % (tam, mean, std)
                                levels[_game] = levels[_game].replace(".", ",")
                            else:                    
                                levels[_game] = 0
                                
                    elif (info == "ChangeMean"):
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                levels[_game] = np.array(df["counter_changes"]).mean()  
                            else:                    
                                levels[_game] = 0
                    elif (info == "ChangeMedian"):
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                levels[_game] = np.median(np.array(df["counter_changes"]))  
                            else:                    
                                levels[_game] = 0                                                                
                    elif (info == "ChangeStd"):
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                levels[_game] = np.array(df["counter_changes"]).std()  
                            else:                    
                                levels[_game] = 0                                                                
                    elif (info == "ChangeVar"):
                        if 'counter_changes' in df.columns:
                            if len(np.array(df["counter_changes"]) >= 0):
                                levels[_game] = np.array(df["counter_changes"]).var()  
                            else:                    
                                levels[_game] = 0                                
                    elif (info == "Levels"):
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                levels[_game] = len(np.array(df["entropy"]))
                            else:                    
                                levels[_game] = 0
                    elif (info == "SegmentsMean"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)
                                                        
                        levels[_game] = np.array(segments).mean()
                    elif (info == "SegmentsMedian"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)
                                                        
                        levels[_game] = np.median(np.array(segments))
                    elif (info == "SegmentsVar"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)
                                                        
                        levels[_game] = np.array(segments).var()                        
                    elif (info == "SegmentsStd"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)
                                                        
                        levels[_game] = np.array(segments).std()                                                
                    elif (info == "SegmentsUsed"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)

                        lev    = 0
                        mean   = 0
                        median = 0
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                entropy = np.array(df["entropy"])
                                lev = len(entropy)
                                mean = entropy.mean()                
                                median = np.median(entropy)                                                               
                            
                        dt_segments = pd.DataFrame({"segments" : segments})                                   
                        dt_segments = dt_segments.groupby(['segments'])['segments'].count()                            
                        rows = dt_segments.shape[0]             
                        
                        total_segments = segments_games[_game]
                        
                        pct = (rows / total_segments) * 100
                        levels[_game] = "-" 
                        if dt_segments.count() > 0:
                            #levels[_game] = "%.02f%s/%i/%.02f/%.02f" % (pct, "%", lev, mean, median)                                                                         
                            #levels[_game] = "%.02f$-$%i$-$%.02f$-$%.02f" % (pct, lev, mean, median)                                                                         
                            levels[_game] = "%.02f$-$%i" % (pct, lev)                                                                         
                            levels[_game] = levels[_game].replace(".", ",")
                    elif (info == "LevelsRepeated"):
                        print()
                        print("Levels Repeated")
                        segments = []                        
                        total = 0
                        if "segments" in df:     
                            segm = df["segments"]            
                            levels_count = segm.count()
                            dt_segments = pd.DataFrame({"segments" : segm})                                   
                            dt_segments = dt_segments.groupby(['segments'])['segments'].count().reset_index(name='counts')
                            total  = dt_segments.where(dt_segments["counts"] > 1).count()
                            total  = np.array(total["counts"]).sum()
                            total = float((total / levels_count) * 100)                                                            
                        else:
                            total = 0

                        levels[_game] = "%.02f" % (total)                                                                         
                        levels[_game] = levels[_game].replace(".", ",")

                    elif (info == "EntropiaAvgMean"):
                        segments = [] 
                        if "segments" in df:     
                            segm = df["segments"]            
                            for s in segm:                
                                sg = s.replace("[", "")
                                sg = sg.replace("]", "")
                                segm1 = sg.split()        
                                segm1 = np.array(segm1).astype(int)                                         
                                for s1 in segm1:
                                    segments.append(s1)                        
                        else:
                            segments.append(0)

                        lev = 0
                        mean = 0
                        median = 0
                        std = 0
                        if 'entropy' in df.columns:
                            if len(np.array(df["entropy"]) >= 0):
                                df = self.__dataframe_format(df, "entropy")
                                entropy = np.array(df["entropy"])
                                lev = len(entropy)
                                mean = entropy.mean()                
                                median = np.median(entropy)                                                            
                                std = np.std(entropy)                                                               
                            
                        dt_segments = pd.DataFrame({"segments" : segments})                                   
                        dt_segments = dt_segments.groupby(['segments'])['segments'].count()                            
                        rows = dt_segments.shape[0]             
                        
                        total_segments = segments_games[_game]
                        
                        pct = (rows / total_segments) * 100
                        levels[_game] = "-" 
                        if dt_segments.count() > 0:
                            #levels[_game] = "%.02f%s/%i/%.02f/%.02f" % (pct, "%", lev, mean, median)                                                                         
                            levels[_game] = "%.02f$-$%.02f$-$%.02f" % (mean, median, std)                                                                         
                            levels[_game] = levels[_game].replace(".", ",")                            
                
                levels["Agent"] = key
                results_writer.write_row(levels)                  
                
            results_writer.close()
                        
            print()
            print(f"Saving to {results_writer.filename}")            
            #print("Arquivo " + filename + " Salvo...")
            print()            
            
                       
        
    def plot_hist_segments_used(self, title,  path, filename):        

        games = [key for key in self.data]           
        segments = {}        
        
        for key, value in self.data.items():                 
            segments[key] = []
        
        dt = {}
        
        #Separando os segmentos por jogo
        for key, value in self.data.items():                 
            
            df = pd.DataFrame(value)
            
            if "segments" in df:     
                segm = df["segments"]            
                for s in segm:                
                    sg = s.replace("[", "")
                    sg = sg.replace("]", "")
                    segm1 = sg.split()        
                    segm1 = np.array(segm1).astype(int)                                         
                    for s1 in segm1:
                        v = {}
                        v["segments"] = s1
                        segments[key].append(v)                        
            else:
                v = {}
                v["segments"] = []
                segments[key].append(v)                        
                    
            dt[key] = pd.DataFrame(segments[key])                                   
        
        for g in games:
            print("plot_hist_segments_used: {}".format(g))
            fig, ax = plt.subplots()  
            
            col_segments = np.array([])
            if "segments" in dt[g]:     
                col_segments = np.array(dt[g]["segments"])
                dt_segments = dt[g].groupby(['segments'])['segments'].count()                
                print(dt_segments)
                col_segments = np.array(dt_segments)                
                dt_segments.to_csv(path+"/"+filename+"-"+g+".csv")                
            else:
                col_segments = np.array([])              
            
            print(col_segments)
            
            ax.set_title(" {}, {} ".format(g, title))
            ax.set_label('Segments levels')
            ax.set_xlabel('Segments levels')            
            ax.set_ylabel('Number of segments')            
            ax.hist(col_segments, color='g', alpha=0.70, label=g)    
            avg = 0
            std = 0
            mode = 0
            if (len(col_segments) > 0):
                avg = col_segments.mean()
                std = col_segments.std()
                #mode = max(statistics.multimode(col_segments))
            
            #ax.axvline(avg, color="k", linestyle="dashed", linewidth=3, label="Avg : {:.2f}, Std : {:.2f}".format(avg, std))
            ax.legend()
            plt.grid()                        
            plt.savefig(path+"/"+filename+"-"+g+".pdf")          
            plt.savefig(path+"/"+filename+"-"+g+".png")              
            plt.close()
            
        
    def plot_cummean(self, rewards, episodes, path, filename, title):
                
        games = [key for key in self.data]       
        
        fig, ax = plt.subplots()                       
        
        for i in range(len(episodes)):
            #label = rewards[i][0] + ", " + episodes[i][1]          
            rew = cum_mean(rewards[i][2])
            plt.plot(episodes[i][2], rew, color='#ff000a')                 
            
        plt.legend()            
        plt.grid(True)            
        plt.xlabel('Dones')
        plt.ylabel('Average rewards per done')
        stitle = title #'Entropy: ' + games[i] + " - " + title
        plt.title(stitle)
        plt.savefig(path+"/"+filename+".pdf")        
        plt.savefig(path+"/"+filename+".png")            
        plt.show()
        plt.close()

    def plot_density_changes(self, path, filename, title):
        
        games = [key for key in self.data]           
        counter_changes = {}
        
        for key, value in self.data.items():                 
            counter_changes[key] = []
        
        dt = {}
        
        #Separando os segmentos por jogo
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)
            
            if "counter_changes" in df.columns:                    
                changes = np.array(df["counter_changes"])
                for c in changes:
                    v = {}
                    v["counter_changes"] = c
                    counter_changes[key].append(v)
            else:
                v = {}
                v["counter_changes"] = 0
                counter_changes[key].append(v)
                    
            dt[key] = pd.DataFrame(counter_changes[key])        
        
        for g in games:
            fig, ax = plt.subplots()            
            
            if "counter_changes" in dt[g]:
                col_segments = np.array(dt[g]["counter_changes"])            
            else:
                col_segments = np.array([0])
            
            ax.set_title(" {}, {} ".format(g, title))
            ax.set_label('Changes levels')
            ax.set_xlabel('Changes')            
            ax.set_ylabel('Number of Levels')                        
            a = ax.hist(col_segments, label=g, density=True)                
            ax.legend()
            plt.plot(a[1][1:],a[0], label=g)            
            plt.grid()                        
            plt.savefig(path+"/"+filename+"-"+g+".pdf")          
            plt.savefig(path+"/"+filename+"-"+g+".png")                     
            plt.close()

    def plot_hist_changes(self, path, filename, title):
        
        games = [key for key in self.data]           
        counter_changes = {}
        
        for key, value in self.data.items():                 
            counter_changes[key] = []
        
        dt = {}
        
        #Separando os segmentos por jogo
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)
            
            if "counter_changes" in df.columns:                    
                changes = np.array(df["counter_changes"])
                for c in changes:
                    v = {}
                    v["counter_changes"] = c
                    counter_changes[key].append(v)
            else:
                v = {}
                v["counter_changes"] = 0
                counter_changes[key].append(v)
                    
            dt[key] = pd.DataFrame(counter_changes[key])        
        
        for g in games:
            fig, ax = plt.subplots()            
            
            if "counter_changes" in dt[g]:
                col_segments = np.array(dt[g]["counter_changes"])            
            else:
                col_segments = np.array([0])
            
            #ax.set_title(" {}, {} ".format(g, title))
            #ax.set_label('Changes levels')
            #ax.set_xlabel('Changes')                        
            ax.set_xlabel('Número de Alterações')            
            #ax.set_ylabel('Number of Levels')            
            ax.set_ylabel('Número de cenários')            
            ax.hist(col_segments, color='g', alpha=0.70, label=g)    
            avg = 0
            std = 0
            mode = 0
            if (len(col_segments) > 0):
                avg = col_segments.mean()
                std = col_segments.std()
                #mode = statistics.mode(col_segments)
            
            #ax.axvline(avg, color="k", linestyle="dashed", linewidth=3, label="Avg : {:.2f}, Std : {:.2f}".format(avg, std))
            ax.axvline(avg, color="k", linestyle="dashed", linewidth=3, label="Média : {:.2f}, Std : {:.2f}".format(avg, std))
            ax.legend()
            plt.grid()                        
            plt.tight_layout()
            plt.savefig(path+"/"+filename+"-"+g+".pdf")          
            plt.savefig(path+"/"+filename+"-"+g+".png")              
            plt.close()
        
    def plot_hist(self, path, filename, title):
        
        games = [key for key in self.data]       
        levels = []
        grouped_entropy = []
        counts_entropy = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)                       
            if 'entropy' in df.columns:                    
                df = self.__dataframe_format(df, "entropy")
                l = np.array(df["entropy"])
                levels.append(l)  
                
                df = pd.DataFrame()  
                df = pd.DataFrame({"entropy" : l})
                
                df = df.groupby(['entropy'])['entropy'].count().reset_index(name='counts')                
                counts = np.array(df["counts"])
                counts = np.round_((counts / len(l)) * 100, decimals = 2)
                counts_entropy.append(counts)
                
                ent = np.round_(np.array(df["entropy"]), decimals = 2)
                ent = np.round_(ent, decimals = 2)
                grouped_entropy.append(ent)                                
            else:
                levels.append(np.array([]))                    
                counts_entropy.append(np.array([]))                    
                grouped_entropy.append(np.array([]))                    
        
        
        for i in range(len(games)):
            fig, ax = plt.subplots()                   
                
            stitle='Entropy: - ' + title        

            ax.set_label('Entropy levels')
            ax.set_ylabel('Levels generated')
            ax.set_xlabel('Entropy')  
            ax.set_title(games[i])
            ax.hist(levels[i], color='g', alpha=0.70, label=games[i])    
            avg = 0
            if (len(levels[i]) > 0):
                avg = levels[i].mean()
            
            ax.axvline(np.mean(levels[i]), color="k", linestyle="dashed", linewidth=3, label="Avg : {:.2f}".format(avg))            
            ax.legend()
            ax.grid()   
        
            plt.savefig(path+"/"+filename+"-"+games[i]+".pdf")                  
            plt.savefig(path+"/"+filename+"-"+games[i]+".png")              
        
            plt.close()        
        fontsize = 12
        for i in range(len(games)):

            fig, ax = plt.subplots(figsize=(6, 8))            
            #fig, ax = plt.subplots(constrained_layout=True)            
            entropy = grouped_entropy[i]
            y_pos = np.arange(len(entropy))

            porcent_entropy = counts_entropy[i]


            m = np.amax(porcent_entropy)
            if m <= 90:
                m += 10
            else:
                m = max(m, 100)

            x = np.arange(0, m, 10)            
            
            error = np.random.rand(len(entropy))

            #rects = ax.barh(y_pos, porcent_entropy, xerr=error, align='center', height=0.5)
            rects = ax.barh(y_pos, porcent_entropy, align='center', height=0.6)

            #percentiles = [str(p) + "%" for p in porcent_entropy]
            #ax.bar_label(rects, percentiles, padding=5, color='black', fontweight='bold', fontsize=fontsize)

            large_percentiles = [str(p) + "%" if p > 35 else '' for p in  porcent_entropy]
            small_percentiles = [str(p) + "%" if p <= 35 else '' for p in  porcent_entropy]
            ax.bar_label(rects, small_percentiles,
                            padding=5, color='black', fontweight='bold', fontsize=fontsize)
            ax.bar_label(rects, large_percentiles,
                            padding=-60, color='white', fontweight='bold', fontsize=fontsize)            

            ax.set_xlim([0, m])
            ax.set_xticks(x)
            ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
            #ax.axvline(m+5, color='grey', alpha=0.25)  # median position

            ax.set_yticks(y_pos, labels=entropy)
            
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('% de Cenários', fontsize=fontsize)
            ax.set_ylabel('Entropia', fontsize=fontsize)                          

            #ax.set_title('How fast do you want to go today?')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.tight_layout()


            #ax.set_label('Entropy levels')
            #ax.set_ylabel('% of levels')
            #ax.set_xlabel('Entropy')  
            #             
            """
            ax.set_label('Entropia dos cenários')
            ax.set_ylabel('% de cenários', fontsize=14)
            ax.set_xlabel('Entropia', fontsize=14)                          
            #ax.set_title(games[i])

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            #plt.legend(fontsize=14)            
            
            graph = ax.bar(x, counts_entropy[i])                            
            ax.set_xticks(x, grouped_entropy[i])
            
            #ax.legend()
            ax.grid()   
            plt.tight_layout()
            """

            
            """
            j = 0
            for p in graph:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()     
                c = counts_entropy[i]       
                plt.text(x+width/2,
                        y+height*1.01,                        
                        str(c[j])+'%',
                        ha='center',
                        weight='bold',fontsize=14)
                j += 1            
            """

            plt.savefig(path+"/"+filename+"-"+games[i]+"-bar.pdf")                  
            plt.savefig(path+"/"+filename+"-"+games[i]+"-bar.png")             
        
            plt.close()                    
                 
    def plot_histv2(self, path, filename, title):
        
        games = [key for key in self.data]       
        
        levels = []
        for key, value in self.data.items():                 
            df = pd.DataFrame(value)                       
            if 'entropy' in df.columns:                    
                df = self.__dataframe_format(df, "entropy")
                levels.append(np.array(df["entropy"]))  
            else:
                levels.append(np.array([]))       
        
        fig, ax = plt.subplots()                    
        ax.set_title(title)
        
        for i in range(len(games)):                                

            ax.hist(levels[i], color='g', alpha=0.90, label=games[i])    
            avg = 0
            if (len(levels[i]) > 0):
                avg = levels[i].mean()
            ax.axvline(avg, color="k", linestyle="dashed", linewidth=3, label="Avg " + games[i] + ": {:.2f}".format(avg))            
            ax.set_label('Entropy levels')
            ax.set_ylabel('Levels Generated')
                        
            avg = 0
            if (len(levels[i+1]) > 0):
                avg = levels[i+1].mean()            
            ax.hist(levels[i+1], color='b', alpha=0.75, label=games[i+1])
            ax.axvline(avg, color="k", linestyle="dotted", linewidth=3, label="Avg " + games[i+1] + ": {:.2f}".format(avg))                        
            
            #('solid', 'solid'),      # Same as (0, ()) or '-'
            #('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
            #('dashed', 'dashed'),    # Same as '--'
            #('dashdot', 'dashdot')]  # Same as '-.'
            
            avg = 0
            if (len(levels[i+2]) > 0):
                avg = levels[i+2].mean()
            ax.hist(levels[i+2], color='r', alpha=0.55, label=games[i+2])
            ax.axvline(avg, color="k", linestyle="dashdot", linewidth=3, label="Avg " + games[i+2] + ": {:.2f}".format(avg))
            
            ax.set_label('Entropy levels')
            ax.set_ylabel('Levels Generated')            
            
            break
            
        ax.legend()
        ax.grid()   
    
        plt.savefig(path+"/"+filename+".pdf")                  
        plt.savefig(path+"/"+filename+".png")         
    
        plt.close()

    def plot_done_penalty(self):
                
        x = np.arange(1, 62)                 
        
        rewards = []                
        
        for i in range(61):
            #print(i)
            r =   - (1 + ((i+1) / 61.0))
            rewards.append(r)       

        plt.plot(x, rewards, color='#ff000a')    
        plt.legend()            
        plt.grid(True)            
        plt.xlabel('Changes')
        plt.ylabel('Rewards')        
        stitle = "Done Penalty"
        plt.title(stitle)
        plt.savefig(os.path.dirname(__file__) + "/DonePenalty.pdf")        
        plt.savefig(os.path.dirname(__file__) + "/DonePenalty.png")            
        plt.show()
        plt.close()
   
    def plot_bonus_factor(self):
        
        fig, ax = plt.subplots()        
        x = np.arange(2, 62)     
        
        rewards = []                
        
        for i in range(1, 61):
            #print(i)
            r =  1.0 / math.sqrt( (i+1) / 61.0)
            rewards.append(r)

        #print(rewards)
        plt.plot(x, rewards, color='#ff000a')                    
        ax.axvline(6, color="k", linestyle="dashed", linewidth=2, label="Min changes: {} ".format(6))        
        ax.plot([6], [3.188521078], 'ro', ms=8, mec='r')                
        
        ax.annotate('3.188521078', xy=(6, 3.188521078), xytext=(7, 3.20), fontsize=12 )        
        
        ax = plt.gca()
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        
        plt.legend()            
        plt.grid(True)            
        plt.xlabel('Changes')
        plt.ylabel('Factor')
        stitle = "Bonus Factor"
        plt.title(stitle)
        plt.savefig(os.path.dirname(__file__) + "/BonusFactor.pdf")        
        plt.savefig(os.path.dirname(__file__) + "/BonusFactor.png")          
        plt.show()
        plt.close()               
        
    def plot_entropy_distance(self):
        
        fig, ax = plt.subplots()                
        
        rewards = []                
        
        max_entropy = 3.0
        entropy_min = 2.75

        max_entropy = 2.58
        entropy_min = 2.25
                 
        ent_ = 0
        entropies = []
        epsilon = 0.1       
        r = 0
        w = math.pi
        while ent_ <= max_entropy:

            entropies.append(ent_)
            e = round(ent_, 2)
            #a = math.log2( (e ** ( (e * math.pi) + epsilon) ) + 1)
            #b = math.log2( (entropy_min ** ( (entropy_min * math.pi) + epsilon) ) + 1)
            #a = math.log2(((math.pi / (e  + 0.5)) * e) + 1 )
            #b = math.log2(((math.pi / (e  + 0.5)) * entropy_min) + 1)
            #r = (a - b) * math.pi
            r = (e**w - entropy_min**w) 
            #r = (((e**math.pi)+1) - ((entropy_min**math.pi)+1))
            #print("\t{} : {}".format(a, b))
            f = 1
            r = (((r + ((sign(r) ) * f))))
            print("{} = {}".format(e, r))
            ent_ += 0.25       
            rewards.append(r)
        """
        while ent_ <= max_entropy:            
            r =  (math.pi / (ent_ + 0.5) ) * 1.80
            ent_ += 0.05
            yy += 0.5
            entropies.append(ent_)
            rewards.append(-r)        
        """
                    
        plt.plot(entropies, rewards, color='#ff000a')                            
        ax.axvline(entropy_min, color="k", linestyle="dashed", linewidth=2, label=r"$\mathrm{HQ} = 2.25 \rightarrow R = 1,0$")        
        #ax.axvline(entropy_min, color="k", linestyle="dashed", linewidth=2, label=r"$\mathrm{HQ} = 2.75 \rightarrow R = 1,0$")
        

        ax.plot([entropy_min], [1.0], 'ro', ms=8, mec='r')                        
        #ax.annotate('-2.458637729', xy=(1.80, -2.458637729), xytext=(1.90, -3.0), fontsize=12 )        
        ax.annotate('$\mathrm{HQ} = 2.25$', xy=(entropy_min, 0), xytext=(1.75, 0.5), fontsize=12 )        
        #ax.annotate('$\mathrm{HQ} = 2.75$', xy=(entropy_min, 0), xytext=(2.10, 0.5), fontsize=12 )        
        
        plt.yticks([12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14])                
        #plt.yticks([12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14,-16,-18,-20,-22,-24,-26])                
        
        #ax.set_ylabel("Entropy Levels", fontsize=16)
        ax.set_ylabel("Rewards", fontsize=14)
        ax.set_xlabel("Entropy", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)            
        plt.grid(True)            
        #plt.xlabel('Entropy')
        #plt.xlabel('Entropia')
        #plt.ylabel('Rewards')
        plt.ylabel('Rewards')
        #stitle = "Entropy quality"
        #plt.title(stitle)
        plt.tight_layout()
        plt.savefig(os.path.dirname(__file__) + "/EntropyPenalty.pdf")        
        plt.savefig(os.path.dirname(__file__) + "/EntropyPenalty.png")        
        plt.show()
        plt.close()       



def gera_graficos_comparativos(results_dir = "./results/",
                            total_timesteps = 50000,              
                            learning_rate: float = 2.5e-4, 
                            n_steps:int   = 128,                                                        
                            batch_size:int = 64,
                            n_epochs:int = 10,                                          
                            act_func = ActivationFunc.SIGMOID.value,
                            entropy_min:int = 1.80,                            
                            envs = [Game.ZELDA.value, Game.ZELDALOWMAPS.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZE.value, Game.DUNGEON.value],
                            representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
                            observations = [WrappersType.MAP.value],              
                            agents   = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HHPD.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPD.value],
                            seed:int = 1000, 
                            n_inference     = 300,
                            uuid = ""):
    
    path_main_results = os.path.join(os.path.dirname(__file__), results_dir)
    
    path_results_experiments = "results-{}-{}-{}-{}-{}-{}-{}-{}{}".format(total_timesteps, n_steps, batch_size, n_epochs, entropy_min, learning_rate, seed, act_func, uuid)
    path_results_experiments = os.path.join(path_main_results, path_results_experiments)
    
    main_path = path_results_experiments #os.path.dirname(__file__) + "/results/Results-50000-128-64-256-4-1.8-0.00025-1000-Sigmoid/"      
      
    n_experiments   = 1        
    RL_ALG          = "PPO"        
    
    #representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value]
    
    games = envs #[Game.ZELDA.value, Game.MINIMAP.value, Game.MAZECOIN.value,  Game.MAZE.value, Game.DUNGEON.value]    
        
    #observations = [WrappersType.MAP.value]
    
    timesteps = [total_timesteps]            
    mlp_units = [64]        

    print("Gerando gráficos dos experimentos")
    plotResults2 = PlotResults()
    for t_time_s in timesteps:
        
        interation_path = t_time_s
        print()
        print("Gerando gráficos: time steps {}".format(t_time_s))
        print()
        
        for mlp_u in mlp_units:        
            
            for _rep in representations:
    
                episodes = []
                scores = []
                average = []
                
                for _obs in observations:                                
                        
                    for par in range(n_experiments):   

                        for name_game in games:                                            
                            
                            print("Game: ", name_game)
                            
                            data_versions = {}
                    
                            scores_inf, episodes_inf, average_inf = [], [], []
                        
                            plotResults = PlotResults()                            
                            
                            for version in agents:
                                
                                #print(version)
                                
                                main_results = path_results_experiments #os.path.dirname(__file__) + "/results/Results-50000-128-64-256-4-1.8-0.00025-1000-Sigmoid/"
                            
                                path_results = os.path.join(main_results, version)                                
                                
                                main_dir = "Experiment 0" + str(par+1) + "-"+version+"-"+name_game+"-"+RL_ALG
                                
                                path_experiments = os.path.join(path_results, main_dir)                    
                                
                                path_file_inference = os.path.join(path_experiments, "Inference"+_rep+"-"+_obs+".csv")                                                                                               
                                
                                data = {}
                                
                                if (os.path.exists(path_file_inference)):
                                    data = pd.read_csv(path_file_inference, index_col=False)
                                
                                    data_versions[version] = data
                            
                                    episodes = np.array(data['Episodes'].tolist())
                                    scores = np.array(data['Scores'].tolist())
                                    average = data['Average'].tolist()
                                    
                                    agent = get_agent(version)
                                    
                                    episodes_inf.append((agent, _obs, episodes))
                                    scores_inf.append((agent, _obs, scores))                            
                                    average_inf.append((agent,_obs,average))
                                    
                                    path_info = os.path.join(path_experiments, _rep+"-"+_obs)    
                                    path_info = os.path.join(path_info, "Info.csv")    
                                    #print(path_info)
                                    if os.path.exists(path_info):
                                        #print("Dados com arquivo")             
                                        data_info = pd.read_csv(path_info, index_col=False)
                                        #print(data_info["entropy"])
                                        plotResults.add(agent, data_info)                                                        
                                        plotResults2.addv2(agent, name_game, data_info)                                
                                    else:                                
                                        #print("Dados sem arquivo {}, {}, {}".format(name_game, agent, _rep))             
                                        dt = []                                        
                                        data_info = []
                                        data_info.append(dt)                                               
                                        plotResults.add(agent, data_info)
                                        plotResults2.addv2(agent, name_game, data_info)                                
                            
                            r = ""
                            if _rep == Behaviors.NARROW_PUZZLE.value:
                                r = "Narrow Puzzle"
                            elif _rep == Behaviors.WIDE_PUZZLE.value:
                                r = "Wide Puzzle"
                                  
                            #title = "Representation: {}, Episodes {} \n Game: {}".format(_rep, t_time_s, name_game)
                            #title = "Representação: {}, Episódios {} \n Ambiente: {}".format(r, t_time_s, name_game)
                            #title = "Representação {}, Ambiente {}\n".format(r, name_game)
                            title = "Ambiente {}\n".format(name_game)
                            
                            filename = "{}-Inference-bestworst-{}-steps-{}".format(name_game, _rep, t_time_s)
                            plotResults.plot(title, main_path, filename, entropy_min=entropy_min)                            

                            filename = "{}-SegmentsUsed-{}-steps-{}".format(name_game, _rep, t_time_s)
                            plotResults.plot_segment_used(main_path, filename, name_game)             
                            
                            filename = "{}-LevelsRepetad-{}-steps-{}".format(name_game, _rep, t_time_s)                                           
                            plotResults.plot_levels_repeted("", main_path, filename)
                            #filename = "{}-Inference-Stackbar-bestworst-{}-steps-{}".format(name_game, _rep, t_time_s)
                            #plotResults.plot_stackbar(title, main_path, filename)  
                            
                            #filename = "{}-Inference-Stackbarver-bestworst-{}-steps-{}".format(name_game, _rep, t_time_s)
                            #plotResults.plot_stackbarvert(title, main_path, filename)                              
                            
                            filename = "{}-Entropy-boxplot-{}-steps-{}".format(name_game, _rep, t_time_s)
                            filename2 = "Entropy-games-{}-{}-{}".format(name_game, _rep, t_time_s)
                            plotResults.plot_boxplot(main_path, filename, filename2, title, xlabel = "Agentes", entropy_min=entropy_min)
                            
                            filename = "{}-CounterChanges-boxplot-{}-steps-{}".format(name_game, _rep, t_time_s)
                            plotResults.plot_boxplot_counter_changes(main_path, filename, title)
                            
                            #filename = "{}-Entropy-hist-{}-steps-{}".format(name_game, _rep, t_time_s)
                            #plotResults.plot_histv2(main_path, filename, title)                       
                            
                    filename = "{}-Comparative-steps-{}".format(_rep, t_time_s)
                    plotResults2.csv_comparative(main_path, filename, envs = envs, entropy_min=entropy_min, n = n_inference)
                    
                    #filename = "{}-Entropy-boxplot-complete{}-steps-{}".format(name_game, _rep, t_time_s)                    
                    #plotResults2.plot_boxplot_complete(main_path, filename, title=title, xlabel = "Agentes", envs=envs)

                    #filename = "{}-Similarity-{}".format(_rep, t_time_s)
                    #plotResults2.plot_similarity_levels(main_path, filename)
                    
                    plotResults2.clear()
                    
def get_agent(version):
    
    agent = ""
                                                             
    if version == Experiment.AGENT_SS.value:
        agent = "S" #SS
    elif version == Experiment.AGENT_HHP.value:
        agent = "H" #HHP                                
    elif version == Experiment.AGENT_HHPD.value:
        agent = "HD"            
    elif version == Experiment.AGENT_HQHPD.value:
        agent = "HQHPD"                                                 
    elif version == Experiment.AGENT_HEQHP.value:
        agent = "HQ"        
    elif version == Experiment.AGENT_HEQHPD.value:
        agent = "HQD"                             
    elif version == Experiment.AGENT_HEQHPEX.value:
        agent = "HQEX"                                                                                   
    else:
        agent = "UNKNOW"            
        
    return agent        
                            
def gera_graficos_individuais(results_dir = "./results/",
                            total_timesteps = 50000,              
                            learning_rate: float = 2.5e-4, 
                            n_steps:int   = 128,                                                        
                            batch_size:int = 64,
                            n_epochs:int = 10,                                          
                            act_func = ActivationFunc.SIGMOID.value,
                            entropy_min:int = 1.80,
                            envs = [Game.ZELDA.value, Game.ZELDALOWMAPS.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZE.value, Game.DUNGEON.value],
                            representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
                            observations = [WrappersType.MAP.value],              
                            agents   = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HHPD.value, Experiment.AGENT_HEQHP.value, Experiment.AGENT_HEQHPD.value],
                            seed:int = 1000,
                            n_inference = 300,
                            uuid = ""):
    
    path_main_results = os.path.join(os.path.dirname(__file__), results_dir)
    
    path_results_experiments = "results-{}-{}-{}-{}-{}-{}-{}-{}{}".format(total_timesteps, n_steps, batch_size, n_epochs, entropy_min, learning_rate, seed, act_func, uuid)        
    path_results_experiments = os.path.join(path_main_results, path_results_experiments)
        
    n_experiments   = 1        
    RL_ALG          = "PPO"            
    
    #observations = [WrappersType.MAP.value]
    
    timesteps = [total_timesteps]    
    mlp_units = [64]           
    
    print("Gerando gráficos dos experimentos")
    print(agents)
    for version in agents:
        
        for t_time_s in timesteps:
            
            interation_path = t_time_s
            print()
            print()
            for mlp_u in mlp_units:                       
                
                for _rep in representations:
        
                    episodes = []
                    scores = []
                    average = []
                    
                    for _obs in observations:                                
                            
                        for par in range(n_experiments):   
                            
                            scores_inf, episodes_inf, average_inf = [], [], []
                            moving_rewards_inf = {}
                            plotResults = PlotResults()
                               
                            for name_game in envs:
                                
                                #path_results = os.path.dirname(__file__) + "/results/Results-50000-128-64-256-4-1.8-0.00025-1000-Sigmoid/"
                             
                                path_results = os.path.join(path_results_experiments, version)
                                
                                main_dir = "Experiment 0" + str(par+1) + "-"+version+"-"+name_game+"-"+RL_ALG                                                    
                                
                                path_experiments = os.path.join(path_results, main_dir)                    
                                
                                path_file_inference = os.path.join(path_experiments, "Inference"+_rep+"-"+_obs+".csv")
                                
                                path_experiments_inference = path_results
                                
                                data = pd.read_csv(path_file_inference, index_col=False)   
                                episodes = np.array(data['Episodes'].tolist())
                                scores = np.array(data['Scores'].tolist())
                                average = data['Average'].tolist()
                                                            
                                episodes_inf.append((name_game, _obs, episodes))
                                scores_inf.append((name_game, _obs, scores))                            
                                average_inf.append((name_game,_obs,average))
                                                                
                                path_info = os.path.join(path_experiments, _rep+"-"+_obs)    
                                path_info = os.path.join(path_info, "Info.csv")    
                                print(path_info)
                                if os.path.exists(path_info):
                                    print("Dados com arquivo")             
                                    data_info = pd.read_csv(path_info, index_col=False)
                                    plotResults.add(name_game, data_info)                                                        
                                else:                   
                                    print("Dados sem arquivo")             
                                    dt = []                                        
                                    data_info = []
                                    data_info.append(dt)                                               
                                    plotResults.add(name_game, data_info)
                                
                                plot_average_rewards(scores, path_experiments_inference, "PPO-"+_rep+"-"+_obs+"-"+name_game, name_game)
                                moving_rewards_inf[name_game] = scores
                                
                            agent = get_agent(version)
                                                            
                            r = ""
                            if _rep == Behaviors.NARROW_PUZZLE.value:
                                r = "Narrow Puzzle"
                            elif _rep == Behaviors.WIDE_PUZZLE.value:
                                r = "Wide Puzzle"                            
                            
                            #title = "Representation: {} - Episodes {} \n {}".format(_rep, t_time_s, agent)
                            title = "Representação: {} \n {}".format(r, agent)
                                                        
                            plot_all_average(average_inf, scores_inf, episodes_inf, path_experiments_inference, "inference-average-all-Games"+_rep, title)
                            plot_all_rewards(average_inf, scores_inf, episodes_inf, path_experiments_inference, "inference-rewards-all-Games"+_rep, title)                            
                            plot_moving_average_all_rewards(moving_rewards_inf, path_experiments_inference, "PPO-moving-average"+_rep)
                             
                            #plotResults.plot_entropy_dist()           
                            plotResults.plot_levels_repeted(title, path_experiments_inference, "Levels-reptead"+_rep, n = n_inference)
                            #plotResults.plot_contour(title, path_experiments_inference, "EntropyContour"+_rep)
                            plotResults.plot_hist_segments_used(title, path_experiments_inference, "HIST-segments"+_rep)
                            #plotResults.plot_stackbar(title, path_experiments_inference, "inference-stackbar-bestworst-all-Games"+_rep)
                            #plotResults.plot_stackbarvert(title, path_experiments_inference, "inference-stackbarver-bestworst-all-Games"+_rep)
                            plotResults.plot(title, path_experiments_inference, "inference-bestworst-all-Games"+_rep, entropy_min=entropy_min, n = n_inference)
                            plotResults.plot_bar_segmentused(title, path_experiments_inference, "inference-segments_used-games"+_rep)
                            plotResults.plot_entropy(path_experiments_inference, "Entropy-games"+_rep, title, entropy_min=entropy_min)                            
                            plotResults.plot_boxplot(path_experiments_inference, "Entropy-games-boxplot"+_rep, "Entropy-games-boxplot-v2"+_rep, title, entropy_min=entropy_min)
                            plotResults.plot_hist(path_experiments_inference, "Entropy-games-hist"+_rep, title)
                            plotResults.plot_hist_changes(path_experiments_inference, "Changes-games-hist"+_rep, title)                            
                            plotResults.plot_density_changes(path_experiments_inference, "Density-games"+_rep, title)
                           
    #plot_moving_average_all_agents_rewards(moving_rewards_inf, path_experiments_inference, "PPO-moving-average"+_rep)

def run_plotter(    
              results_dir = "./results/",
              total_timesteps = 50000,              
              learning_rate: float = 2.5e-4, 
              n_steps:int   = 128,                                          
              batch_size:int = 64,
              n_epochs:int = 10,                                          
              act_func = ActivationFunc.SIGMOID.value,
              entropy_min:int = 1.80,
              envs = [Game.ZELDA.value, Game.ZELDALOWMAPS.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZE.value, Game.DUNGEON.value],
              representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
              observations = [WrappersType.MAP.value],              
              agents   = [Experiment.AGENT_HHP.value],
              n_inference:int = 300,
              board = [[2,3], [2,4]],
              seed:int = 1000, uuid = "", language = "pt-br", tag = ""):
    """
    gera_graficos_individuais(results_dir = results_dir,
                            total_timesteps = total_timesteps,
                            learning_rate   = learning_rate,        
                            n_steps         = n_steps,                                                                                    
                            batch_size      = batch_size,
                            n_epochs        = n_epochs,
                            act_func        = act_func,
                            entropy_min     = entropy_min,
                            envs            = envs,
                            agents          = agents,
                            representations = representations,      
                            observations    = observations,
                            seed            = seed,
                            n_inference     = n_inference,
                            uuid            = uuid)
   
    gera_graficos_comparativos(results_dir  = results_dir,
                            total_timesteps = total_timesteps,
                            learning_rate   = learning_rate,        
                            n_steps         = n_steps,                                                                                    
                            batch_size      = batch_size,
                            n_epochs        = n_epochs,
                            act_func        = act_func,
                            entropy_min     = entropy_min,
                            envs            = envs,
                            agents          = agents,
                            n_inference     = n_inference,
                            representations = representations,      
                            observations    = observations,
                            seed            = seed,
                            uuid            = uuid)
    """
    gera_graficos_expressive_range(results_dir = results_dir,
                                total_timesteps = total_timesteps,
                                learning_rate   = learning_rate,        
                                n_steps         = n_steps,                                                                                    
                                batch_size      = batch_size,
                                n_epochs        = n_epochs,
                                act_func        = act_func,
                                entropy_min     = entropy_min,
                                envs            = envs,
                                agents           = agents,                                   
                                representations = representations,      
                                observations    = observations,
                                seed            = seed,
                                board           = board,
                                uuid            = uuid,
                                tag             = tag)
    

if __name__ == '__main__':
    #gera_graficos_individuais()
    #gera_graficos_comparativos()
    plotResults = PlotResults()
    #plotResults.plot_bonus_factor()
    #plotResults.plot_done_penalty()
    plotResults.plot_entropy_distance()