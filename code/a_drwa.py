# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:58:14 2022

@author: ShengyueJiang
"""

import experiment
import dataLoader as dl
import networkx as nx
import pandas as pd
from experimentAll import experiment
from tqdm import tqdm
from informationIndex import CN_MI,CN_TE
from behaviorIndex import TCN_mean_CN,TCN_min_CN
from evaluation import evaluation_auc,evaluation_precision
import numpy as np
import matplotlib.pyplot as plt

COLOR = ['k','r','b','g']
MARK = ['>','x','+','*']

DATA_LIST = {
    # 'science':['BMJ'],
    'communication':['msg','email']
    }



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
            time_list = dl.generate_time_list(data, 300)
        
        auc_dict = {key:[] for key in ['MICN','TECN','ITCN(mean)','ITCN(mIn)']}
        pre_dict = {key:[] for key in ['MICN','TECN','ITCN(mean)','ITCN(mIn)']}
        for a in np.arange(0,1.1,0.1):
            print(a)
            real_link_dict = {key:[] for key in ['MICN','TECN','ITCN(mean)','ITCN(mIn)']}
            fake_link_dict = {key:[] for key in ['MICN','TECN','ITCN(mean)','ITCN(mIn)']}
            for edge in tqdm(test_graph.edges()):
                real_link_dict['ITCN(mean)'].append((edge,TCN_mean_CN(train_graph, edge, a)))
                real_link_dict['ITCN(mIn)'].append((edge,TCN_min_CN(train_graph, edge, a)))
                real_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type, a)))
                real_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type, a)))
                
            for edge in tqdm(no_list):
                fake_link_dict['ITCN(mean)'].append((edge,TCN_mean_CN(train_graph, edge, a)))
                fake_link_dict['ITCN(mIn)'].append((edge,TCN_min_CN(train_graph, edge, a)))
                fake_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type, a)))
                fake_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type, a)))
            
            # print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
            # print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
            for index in auc_dict:
                auc_dict[index].append(evaluation_auc(real_link_dict[index], fake_link_dict[index]))
                pre_dict[index].append(evaluation_precision(real_link_dict[index], fake_link_dict[index], len(test_graph.edges())))
        
        x = [str(round(i, 1)) for i in np.arange(0,1.1,0.1)]
        print(auc_dict)
        print(pre_dict)
        plt.figure(figsize=(7, 10))
        plt.subplot(2,1,1)
        for index,i in zip(auc_dict,[i for i in range(4)]):
            plt.plot(x,auc_dict[index],COLOR[i] + MARK[i] + '-')
        plt.xlabel('α')
        plt.ylabel('AUC')
        plt.legend()
        plt.subplot(2,1,2)
        for index,i in zip(auc_dict,[i for i in range(4)]):
            plt.plot(x,pre_dict[index],COLOR[i] + MARK[i] + '-', label = index)
        plt.xlabel('α')
        plt.ylabel('precision')
        plt.legend()
        plt.show()