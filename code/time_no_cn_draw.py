# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:10:11 2022

@author: ShengyueJiang
"""


import experiment
import dataLoader as dl
import networkx as nx
import pandas as pd
from experimentAll import experiment
from tqdm import tqdm
from informationIndex import MI,TE
from evaluation import evaluation_auc,evaluation_precision
import numpy as np
import matplotlib.pyplot as plt

DATA_LIST = {
    'science':['BMJ','nature','science','jama','england'],
    # 'communication':['msg', 'email', 'SD01', 'SD02', 'SD03']
    }

COLOR = ['k','r','b','g']
MARK = ['>','x','+','*']


for data_type in DATA_LIST:
    
    for data_name in DATA_LIST[data_type]:
        
        data = dl.data_load(data_name, data_type)
        
        print('---building {0} net---'.format(data_name))
        
        G,train_graph, test_graph = dl.data_div(data)
        print('---generate no list {0} net---'.format(data_name))
        no_list = dl.generate_no_list(train_graph, test_graph)
        
        
        auc_dict = {key:[] for key in ['MI','TE',]}
        pre_dict = {key:[] for key in ['MI','TE',]}
        
        
        
        if data_type == 'science':
            for t in range(1,20):
                print(t)
                time_list = dl.generate_time_list(data, t)
                print(time_list)
                real_link_dict = {key:[] for key in ['MI','TE']}
                fake_link_dict = {key:[] for key in ['MI','TE']}
                for edge in tqdm(test_graph.edges()):
                    # real_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # real_link_dict['TMIn_CN'].append((edge,TMIn_CN(train_graph, edge)))
                    real_link_dict['MI'].append((edge,MI(train_graph, edge,time_list, data_type)))
                    real_link_dict['TE'].append((edge,TE(train_graph, edge,time_list, data_type)))
                    
                for edge in tqdm(no_list):
                    # fake_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # fake_link_dict['TMIn_CN'].append((edge,TMIn_CN(train_graph, edge)))
                    fake_link_dict['MI'].append((edge,MI(train_graph, edge,time_list, data_type)))
                    fake_link_dict['TE'].append((edge,TE(train_graph, edge,time_list, data_type)))
                
                # print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
                # print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
                for index in auc_dict:
                    auc_dict[index].append(evaluation_auc(real_link_dict[index], fake_link_dict[index]))
                    pre_dict[index].append(evaluation_precision(real_link_dict[index], fake_link_dict[index], len(test_graph.edges())))
            print(auc_dict)
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(np.arange(1,20),auc_dict[index],COLOR[i] + MARK[i] + '-')
                plt.title(data_name)
            plt.subplot(1,2,2)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(np.arange(1,20),pre_dict[index],COLOR[i] + MARK[i] + '-')
            plt.show()
        elif data_type == 'communication':
            for t in range(1,86400,300):
                time_list = dl.generate_time_list(data, t)
                real_link_dict = {key:[] for key in ['MI','TE']}
                fake_link_dict = {key:[] for key in ['MI','TE']}
                for edge in tqdm(test_graph.edges()):
                    # real_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # real_link_dict['TMIn_CN'].append((edge,TMIn_CN(train_graph, edge)))
                    real_link_dict['MI'].append((edge,MI(train_graph, edge,time_list, data_type)))
                    real_link_dict['TE'].append((edge,TE(train_graph, edge,time_list, data_type)))
                    
                for edge in tqdm(no_list):
                    # fake_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # fake_link_dict['TMIn_CN'].append((edge,TMIn_CN(train_graph, edge)))
                    fake_link_dict['MI'].append((edge,MI(train_graph, edge,time_list, data_type)))
                    fake_link_dict['TE'].append((edge,TE(train_graph, edge,time_list, data_type)))
                
                # print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
                # print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
                for index in auc_dict:
                    auc_dict[index].append(evaluation_auc(real_link_dict[index], fake_link_dict[index]))
                    pre_dict[index].append(evaluation_precision(real_link_dict[index], fake_link_dict[index], len(test_graph.edges())))
