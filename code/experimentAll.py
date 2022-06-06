# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:42:47 2022

@author: ShengyueJiang
"""
import pandas as pd
from tqdm import tqdm
import evaluation
from topoIndex import *
from informationIndex import *
from weightIndex import *
from timeIndex import *
from behaviorIndex import *

INDEX_LIST = {
    'topo' : ['CN', 'AA', 'RA', 'JACC', 'LHN', ], 
    'weight' : ['WCN', 'WAA', 'WRA', 'rWCN', 'rWAA', 'rWRA'],
    'time' : ['TCN', 'TRA', 'TLHN', 'TJACC'],
    'information' : ['CN_MI','CN_TE'],
    'behavior' : ['TCN_mean_CN','TCN_min_CN'],
    }



def experiment(data_name, train_graph, test_graph, no_list, time_list, data_type):
    
    index = []
    for i in INDEX_LIST:
        index.extend(INDEX_LIST[i])
    
    AUC = pd.DataFrame(columns = [data_name], index=index)
    PRE = pd.DataFrame(columns = [data_name], index=index)
    
    for index_type in INDEX_LIST:
        
        for index in INDEX_LIST[index_type]:
            
            real = []
            fake = []
            
            for edge in tqdm(test_graph.edges()):
                if index_type == 'information':
                    real.append((edge,eval(index)(train_graph, edge, time_list, data_type)))
                else:
                    real.append((edge,eval(index)(train_graph, edge)))
                
            for edge in tqdm(no_list):
                if index_type == 'information':
                    fake.append((edge,eval(index)(train_graph, edge, time_list, data_type)))
                else:
                    fake.append((edge,eval(index)(train_graph, edge)))
                
            AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
            PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
            
    return AUC, PRE