# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:20:04 2022

@author: ShengyueJiang
"""
import dataLoader as dl
import pandas as pd
from topoIndex import *
from informationIndex import *
from weightIndex import *
from timeIndex import *
from behaviorIndex import *
from tqdm import tqdm
import evaluation


INDEX_LIST = {
    'topo' : ['CN', 'AA', 'RA', 'JACC', 'LHN', ], 
    'weight' : ['WCN', 'WAA', 'WRA', 'rWCN', 'rWAA', 'rWRA'],
    'time' : ['TCN', 'TRA', 'TLHN', 'TJACC'],
    'information' : ['MI', 'TE_xiu','CN_MI', 'CN_TE_xiu'],
    'behavior' : ['TCN_mean','TCN_min'],
    }

def topo(DATA_LIST):

    TOPO_AUC = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['topo'])
    TOPO_PRE = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['topo'])
    #循环读取数据和数据类型
    for data_type in DATA_LIST:
        
        for data_name in DATA_LIST[data_type]:
            
            data = dl.data_load(data_name, data_type)
            
            print('---building {0} net---'.format(data_name))
            G,train_graph, test_graph = dl.data_div(data, index_type = 'topo')
            no_list = dl.generate_no_list(train_graph, test_graph)
            
            for index in INDEX_LIST['topo']:
                real = []
                fake = []
                
                print('---evaluating---')
                for edge in tqdm(test_graph.edges()):
                    real.append((edge,eval(index)(train_graph, edge)))
                    
                for edge in tqdm(no_list):
                    fake.append((edge,eval(index)(train_graph, edge)))
                    
                TOPO_AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
                TOPO_PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
                
                print('{1}_AUC:{0}'.format(TOPO_AUC[data_name][index], index))
                print('{1}_PRE:{0}'.format(TOPO_PRE[data_name][index], index))
                    
    TOPO_AUC.to_csv('../result/index/topo_auc.csv')
    TOPO_PRE.to_csv('../result/index/topo_pre.csv')
    
    print(TOPO_AUC)
    print(TOPO_PRE)

def weight(DATA_LIST):
    
    WEIGHT_AUC = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['weight'])
    WEIGHT_PRE = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['weight'])
    
    for data_type in DATA_LIST:
        
        for data_name in DATA_LIST[data_type]:
            
            data = dl.data_load(data_name, data_type)
            print('---building {0} net---'.format(data_name))
            G,train_graph, test_graph = dl.data_div(data, index_type = 'weight')
            no_list = dl.generate_no_list(train_graph, test_graph)
            
            for index in INDEX_LIST['weight']:
                real = []
                fake = []
                
                print('---evaluating---')
                for edge in tqdm(test_graph.edges()):
                    real.append((edge,eval(index)(train_graph, edge)))
                    
                for edge in tqdm(no_list):
                    fake.append((edge,eval(index)(train_graph, edge)))
                
                WEIGHT_AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
                WEIGHT_PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
                
                print('{1}_AUC:{0}'.format(WEIGHT_AUC[data_name][index], index))
                print('{1}_PRE:{0}'.format(WEIGHT_PRE[data_name][index], index))
                    
    WEIGHT_AUC.to_csv('../result/index/weight_auc.csv')
    WEIGHT_PRE.to_csv('../result/index/weight_pre.csv')
    
    print(WEIGHT_AUC)
    print(WEIGHT_PRE)

def time(DATA_LIST):
    TIME_AUC = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['time'])
    TIME_PRE = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['time'])
    
    for data_type in DATA_LIST:
        
        for data_name in DATA_LIST[data_type]:
            
            data = dl.data_load(data_name, data_type)
            print('---building {0} net---'.format(data_name))
            G,train_graph, test_graph = dl.data_div(data, index_type = 'time')
            no_list = dl.generate_no_list(train_graph, test_graph)
            
            if data_type == 'science':
                time_list = dl.generate_time_list(data, 1)
            elif data_type == 'communication':
                time_list = dl.generate_time_list(data, 300)
            
            for index in INDEX_LIST['time']:
                real = []
                fake = []
                
                print('---evaluating---')
                for edge in tqdm(test_graph.edges()):
                    real.append((edge,eval(index)(train_graph, edge)))
                    
                for edge in tqdm(no_list):
                    fake.append((edge,eval(index)(train_graph, edge)))
                
                TIME_AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
                TIME_PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
                
                print('{1}_AUC:{0}'.format(TIME_AUC[data_name][index], index))
                print('{1}_PRE:{0}'.format(TIME_PRE[data_name][index], index))
                    
    TIME_AUC.to_csv('../result/index/time_auc.csv')
    TIME_PRE.to_csv('../result/index/time_pre.csv')
    
    print(TIME_AUC)
    print(TIME_PRE)

def information(DATA_LIST):
    TOPO_AUC = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['information'])
    TOPO_PRE = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['information'])
    
    for data_type in DATA_LIST:
        
        for data_name in DATA_LIST[data_type]:
            
            data = dl.data_load(data_name, data_type)
            print('******building {0} net******'.format(data_name))
            G,train_graph, test_graph = dl.data_div(data, index_type = 'information')
            no_list = dl.generate_no_list(train_graph, test_graph)
            
            if data_type == 'science':
                time_list = dl.generate_time_list(data, 1)
            elif data_type == 'communication':
                time_list = dl.generate_time_list(data, 300)
            
            for index in INDEX_LIST['information']:
                real = []
                fake = []
                
                print('---evaluating---')
                for edge in tqdm(test_graph.edges()):
                    real.append((edge,eval(index)(train_graph, edge, time_list)))
                    
                for edge in tqdm(no_list):
                    fake.append((edge,eval(index)(train_graph, edge, time_list)))
                
                
                TOPO_AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
                TOPO_PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
                
                print('{1}_AUC:{0}'.format(TOPO_AUC[data_name][index], index))
                print('{1}_PRE:{0}'.format(TOPO_PRE[data_name][index], index))
                    
    TOPO_AUC.to_csv('../result/index/information_auc.csv')
    TOPO_PRE.to_csv('../result/index/information_pre.csv')
    
    print(TOPO_AUC)
    print(TOPO_PRE)

def behavior(DATA_LIST):

    behavior_AUC = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['behavior'])
    behavior_PRE = pd.DataFrame(columns=DATA_LIST['science'] + DATA_LIST['communication'], index=INDEX_LIST['behavior'])
    #循环读取数据和数据类型
    for data_type in DATA_LIST:
        
        for data_name in DATA_LIST[data_type]:
            
            data = dl.data_load(data_name, data_type)
            
            print('---building {0} net---'.format(data_name))
            G,train_graph, test_graph = dl.data_div(data, index_type = 'topo')
            no_list = dl.generate_no_list(train_graph, test_graph)
            
            for index in INDEX_LIST['behavior']:
                real = []
                fake = []
                
                print('---evaluating---')
                for edge in tqdm(test_graph.edges()):
                    real.append((edge,eval(index)(train_graph, edge)))
                    
                for edge in tqdm(no_list):
                    fake.append((edge,eval(index)(train_graph, edge)))
                    
                behavior_AUC[data_name][index] = evaluation.evaluation_auc(real, fake)
                behavior_PRE[data_name][index] = evaluation.evaluation_precision(real, fake, len(test_graph.edges()))
                
                print('{1}_AUC:{0}'.format(behavior_AUC[data_name][index], index))
                print('{1}_PRE:{0}'.format(behavior_PRE[data_name][index], index))
                    
    behavior_AUC.to_csv('../result/index/behavior_auc.csv')
    behavior_PRE.to_csv('../result/index/behavior_pre.csv')
    
    print(behavior_AUC)
    print(behavior_PRE)