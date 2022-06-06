# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:35:30 2022

@author: ShengyueJiang
"""
import experiment
import dataLoader as dl
import networkx as nx
import pandas as pd
from experimentAll import experiment
from tqdm import tqdm
from informationIndex import CN_MI,CN_TE,MI
from behaviorIndex import TCN_mean_CN,TCN_min_CN
from evaluation import evaluation_auc,evaluation_precision

DATA_LIST = {
    'science':['BMJ','nature','science','jama','england'],
    'communication':['msg', 'email', 'SD01', 'SD02', 'SD03']
    }

INDEX_LIST = {
    'topo' : ['CN', 'AA', 'RA', 'JACC', 'LHN', ], 
    'weight' : ['WCN', 'WAA', 'WRA', 'rWCN', 'rWAA', 'rWRA'],
    'time' : ['TCN', 'TRA', 'TLHN', 'TJACC'],
    'information' : ['CN_MI','CN_TE'],
    'behavior' : ['TCN_mean_CN','TCN_min_CN'],
    }

# experiment.topo(DATA_LIST)
# experiment.information(DATA_LIST)
# experiment.weight(DATA_LIST)
# experiment.time(DATA_LIST)
# experiment.behavior(DATA_LIST)

columns = DATA_LIST['science'] + DATA_LIST['communication']
index = []

for i in INDEX_LIST:
    index.extend(INDEX_LIST[i])

AUC_result = pd.DataFrame(columns=columns, index=index)
PRE_result = pd.DataFrame(columns=columns, index=index)

for data_type in DATA_LIST:
    
    for data_name in DATA_LIST[data_type]:
        
        data = dl.data_load(data_name, data_type)
        
        print('---building {0} net---'.format(data_name))
        
        G,train_graph, test_graph = dl.data_div(data)
        print('---generate no list {0} net---'.format(data_name))
        no_list = dl.generate_no_list(train_graph, test_graph)
        
        if data_type == 'science':
            time_list = dl.generate_time_list(data, 1)
        elif data_type == 'communication':
            time_list = dl.generate_time_list(data, 60)
        
        auc, pre = experiment(data_name, train_graph, test_graph, no_list, time_list, data_type)
        print(auc)
        print('-' * 10)
        print(pre)
        AUC_result[data_name] = auc
        PRE_result[data_name] = pre

# AUC_result.to_csv('../result/auc.csv')
# PRE_result.to_csv('../result/precision.csv')

#取权重
# for t in DATA_LIST:
#     for data_name in DATA_LIST[t]:

#         data = dl.data_load(data_name)
        
#         G = nx.from_pandas_edgelist(data, 's', 'e', ['t', 'w'], create_using=nx.MultiGraph())
        
#         for u,v,key in G.edges(keys = True):
            
#             if key > 1:
                
#                 index = list(data[(data['s'] == u) & (data['e'] == v)].index)
#                 index.extend(list(data[(data['s'] == u) & (data['e'] == v)].index))
                
#                 w = len(index)
                
#                 if w > 0:
#                     for i in index:
#                         data.iloc[i]['w'] = w
        
#         data = data.drop_duplicates()
#         data.to_csv('../data/{}.csv'.format(data_name), index = False)

# for data_type in DATA_LIST:
    
#     for data_name in DATA_LIST[data_type]:
        
#         data = dl.data_load(data_name, data_type)
        
#         print('---building {0} net---'.format(data_name))
        
#         G,train_graph, test_graph = dl.data_div(data)
#         print('---generate no list {0} net---'.format(data_name))
#         no_list = dl.generate_no_list(train_graph, test_graph)
        
#         if data_type == 'science':
#             time_list = dl.generate_time_list(data, 1)
#         elif data_type == 'communication':
#             time_list = dl.generate_time_list(data, 86400)
        
        
#         real = []
#         fake = []
        
#         for edge in tqdm(test_graph.edges()):
#             real.append((edge,MI(train_graph, edge,time_list, data_type)))
            
#         for edge in tqdm(no_list):
#             fake.append((edge,MI(train_graph, edge,time_list, data_type)))
        
#         print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
#         print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))