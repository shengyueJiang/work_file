# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:50:14 2022

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

DATA_LIST = {
    # 'science':['BMJ'],
    'communication':[ 'SD03']
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
        
        
        auc_dict = {key:[] for key in ['MICN','TECN']}
        pre_dict = {key:[] for key in ['MICN','TECN']}
        
        
        
        if data_type == 'science':
            for t in range(1,21):
                print(t)
                time_list = dl.generate_time_list(data, t)
                real_link_dict = {key:[] for key in ['MICN','TECN']}
                fake_link_dict = {key:[] for key in ['MICN','TECN']}
                for edge in tqdm(test_graph.edges()):
                    # real_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # real_link_dict['TCN_min_CN'].append((edge,TCN_min_CN(train_graph, edge)))
                    real_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type)))
                    real_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type)))
                    
                for edge in tqdm(no_list):
                    # fake_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # fake_link_dict['TCN_min_CN'].append((edge,TCN_min_CN(train_graph, edge)))
                    fake_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type)))
                    fake_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type)))
                
                # print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
                # print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
                for index in auc_dict:
                    auc_dict[index].append(evaluation_auc(real_link_dict[index], fake_link_dict[index]))
                    pre_dict[index].append(evaluation_precision(real_link_dict[index], fake_link_dict[index], len(test_graph.edges())))
            print(auc_dict)
            x = [str(i) for i in np.arange(1,21)]
            plt.figure(figsize=(15, 5))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.subplot(1,2,1)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(x,auc_dict[index],COLOR[i] + MARK[i] + '-')
                plt.xlabel('时间/年', size = 10)
                plt.ylabel('AUC', size = 10)
            for index in auc_dict:
                for auc in auc_dict[index]:
                    if auc < 0.64:
                        line_mi = auc_dict[index].index(auc)
                        break
            plt.vlines(line_mi, 0, 1, colors = 'b', linestyles = "dotted")
            plt.subplot(1,2,2)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(x,pre_dict[index],COLOR[i] + MARK[i] + '-', label = index)
                plt.xlabel('时间/年', size = 10)
                plt.ylabel('precision', size = 10)
            for index in pre_dict:
                for pre in pre_dict[index]:
                    if pre < 0.459:
                        line_mi = pre_dict[index].index(pre)
                        break
            plt.vlines(line_mi, 0, 1, colors = 'b', linestyles = "dotted") 

            plt.legend(fontsize = 10)
            plt.show()
        elif data_type == 'communication':
 
            x = [str(i) for i in range(1, 289)]
            for t in range(300,86401,300):
                print(t)
                time_list = dl.generate_time_list(data, t)
                real_link_dict = {key:[] for key in ['MICN','TECN']}
                fake_link_dict = {key:[] for key in ['MICN','TECN']}
                for edge in tqdm(test_graph.edges()):
                    # real_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # real_link_dict['TCN_min_CN'].append((edge,TCN_min_CN(train_graph, edge)))
                    real_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type)))
                    real_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type)))
                    
                for edge in tqdm(no_list):
                    # fake_link_dict['TCN_mean_CN'].append((edge,TCN_mean_CN(train_graph, edge)))
                    # fake_link_dict['TCN_min_CN'].append((edge,TCN_min_CN(train_graph, edge)))
                    fake_link_dict['MICN'].append((edge,CN_MI(train_graph, edge,time_list, data_type)))
                    fake_link_dict['TECN'].append((edge,CN_TE(train_graph, edge,time_list, data_type)))
                
                # print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
                # print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
                for index in auc_dict:
                    auc_dict[index].append(evaluation_auc(real_link_dict[index], fake_link_dict[index]))
                    pre_dict[index].append(evaluation_precision(real_link_dict[index], fake_link_dict[index], len(test_graph.edges())))
            plt.figure(figsize=(15, 5))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.subplot(1,2,1)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(x,auc_dict[index],COLOR[i] + MARK[i] + '-')
                
                plt.xlabel('时间/秒', size = 10)
                plt.ylabel('AUC', size = 10)
       
  
            plt.subplot(1,2,2)
            for index,i in zip(auc_dict,[i for i in range(2)]):
                plt.plot(x,pre_dict[index],COLOR[i] + MARK[i] + '-', label = index)
                plt.xlabel('时间/秒', size = 10)
                plt.ylabel('precision', size = 10)
            plt.legend(fontsize = 10)
            plt.show()
        print(auc_dict)

# 当时间尺度到达一定阈值时，基于信息论所刻画的同步信息的作用已经消失了，会达到和CN一样的精确度。
