# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:26:23 2022

@author: ShengyueJiang
"""

import random
from gensim.models import Word2Vec
import networkx as nx
import dataLoader as dl

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

def deepWalk(G, link_list, length = 50):
    
        series = []
        for link in link_list:
            walk = [link[0]]
            for i in range(1, length):
                walk.append(str(random.choice(list(G.neighbors(int(walk[-1]))))))
            series.append(walk)
        model = Word2Vec(sentences=series,sg=1,workers=3)
        return model
    
def node2vec(G, link_list, length = 50, p = 2, q = 0.5):
        series = []
        for link in link_list:
            walk = [link[0]]
            
            for i in range(1, length):
                cur = int(walk[-1])
                nbr = list(G.neighbors(cur))
                if len(nbr) == 1:
                    walk.append(nbr[0])
                else:
                    path_dict = {}
                    if len(walk) > 1:
                        prev = walk[-2]
                    else:
                        prev = walk[-1]
                        
                    for node in nbr:
                        short_path = len(nx.shortest_path(G, prev, node))
                        if short_path == 1:
                            path_dict[node] = 1 / p
                        elif short_path == 2:
                            path_dict[node] = 1
                        elif short_path == 3:
                            path_dict[node] = 1 / q
                            
                    for node in path_dict:
                        path_dict[node] = path_dict[node] / sum(path_dict.values())
                        
                    walk.append(random_pick(path_dict.keys(), path_dict.values()))
                        
            series.append(walk)
        
        model = Word2Vec(sentences=series,sg=1,workers=3)
        return model
