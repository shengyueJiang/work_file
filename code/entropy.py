# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:15:15 2022

@author: ShengyueJiang
"""

import math
from collections import Counter
import dataLoader as dl
import pandas as pd

DATA_LIST = {
    'science':['BMJ','nature','science','jama','england'],
    'communication':['msg', 'email', 'SD01', 'SD02', 'SD03']
    }

DATA_E = {key:[] for key in ['BMJ','nature','science','jama','england',
                             'msg', 'email', 'SD01', 'SD02', 'SD03'] }

def Entropy(DataList):

    counts = len(DataList)      # 总数量
    counter = Counter(DataList) # 每个变量出现的次数
    prob = {i[0]:i[1]/counts for i in counter.items()}      # 计算每个变量的 p*log(p)
    H = - sum([i[1]*math.log2(i[1]) for i in prob.items()]) # 计算熵    
    return H

def _getTime(G, time_list, divide, node, type_):
    
    x_vector = [0 for i in time_list]
    #所有关于x的时间和关于y的时间
    time_x = [G[i[0]][i[1]]['t'] for i in list(G.edges(node))]

    x_cut = pd.cut(time_x, time_list, labels=False)


    
    if type_ == 'communication':

        
        # x_vector = [0 for i in range(divide)]
        
        for v in x_cut:

            try:
                x_vector[int(v)] = 1
            except Exception:
                pass
        

        return x_vector
    
    else:
        
        
        for s in range(divide):
            if s in x_cut:
                x_vector.append(1)
            else:
                x_vector.append(0)

        
        return x_vector

for data_type in DATA_LIST:
    for data_name in DATA_LIST[data_type]:
        
        data = dl.data_load(data_name, data_type)
        
        print('---building {0} net---'.format(data_name))
        G,train_graph, test_graph = dl.data_div(data, index_type = 'topo')
        
        
        if data_type == 'science':
            time_list = dl.generate_time_list(data, 1)
        elif data_type == 'communication':
            time_list = dl.generate_time_list(data, 300)
            
        divide = int(len(time_list))
        
        for node in G.nodes():
            DataList = _getTime(G, time_list, divide, node, type_ = data_type)
            DATA_E[data_name].append(Entropy(DataList))

for data in DATA_E:
    print(data, sum(DATA_E[data]) / len(DATA_E[data]))
    data_h = pd.DataFrame({'H':DATA_E[data]})
    data_h.to_csv('../result/analysis/{0}_H.csv'.format(data))