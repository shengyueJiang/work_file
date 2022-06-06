# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:25:32 2022

@author: ShengyueJiang
"""
import pandas as pd
from tqdm import tqdm
import dataLoader as dl
from collections import Counter
import math
from informationIndex import TE,MI
from operator import eq
from evaluation import evaluation_auc,evaluation_precision
from diffDraw import draw
import numpy as np
from informationIndex import getTime, te_cal, getnodeTime
from sklearn.metrics.cluster import normalized_mutual_info_score as mi_cal
from behaviorIndex import TCN_mean, TCN_min
from pyinform.transferentropy import *
import networkx as nx
import matplotlib.pyplot as plt



def  analysisIndex(real, fake):
    true_link = []
    false_link = []
    mid_link = []
    for link_real, link_fake in zip(real, fake):
        if link_real[1] < link_fake[1]:
            false_link.append((link_real[0], link_fake[0]))
        elif link_real[1] > link_fake[1]:
            true_link.append((link_real[0], link_fake[0]))
        elif link_real[1] == link_fake[1]:
            mid_link.append((link_real[0], link_fake[0]))
        else:
            print('----')
    return set(true_link), set(false_link), set(mid_link)

def analysisDiff(data_name, data_type):

    # 科学家合作网络
    data = dl.data_load(data_name, type_ = data_type)
    G,train_graph, test_graph = dl.data_div(data, index_type = 'topo')
    no_list = dl.generate_no_list(train_graph, test_graph)
    
    if data_type == 'science':
        time_list = dl.generate_time_list(data, 1)
    else:
        time_list = dl.generate_time_list(data, 300)

# 计算部分
# =============================================================================
#     te_real_score = []
#     te_fake_score = []
#     # tex_real_score = []
#     # tex_fake_score = []
#     mi_real_score = []
#     mi_fake_score = []
#     tcn_mean_real_score = []
#     tcn_mean_fake_score = []
#     tcn_min_real_score = []
#     tcn_min_fake_score = []
#     
#     for edge in tqdm(test_graph.edges()):
#         te_real_score.append((edge,TE(train_graph, edge, time_list)))
#         # tex_real_score.append((edge,TE_xiu(train_graph, edge, time_list)))
#         mi_real_score.append((edge,MI(train_graph, edge, time_list)))
#         tcn_mean_real_score.append((edge,TCN_mean(train_graph, edge)))
#         tcn_min_real_score.append((edge,TCN_min(train_graph, edge)))
#         
#     for edge in tqdm(no_list):
#         te_fake_score.append((edge,TE(train_graph, edge, time_list)))
#         # tex_fake_score.append((edge,TE_xiu(train_graph, edge, time_list)))
#         mi_fake_score.append((edge,MI(train_graph, edge, time_list)))
#         tcn_mean_fake_score.append((edge,TCN_mean(train_graph, edge)))
#         tcn_min_fake_score.append((edge,TCN_min(train_graph, edge)))
#     
#     te_true_link, te_false_link,te_mid_link= analysisIndex(te_real_score, te_fake_score)
#     mi_true_link, mi_false_link,mi_mid_link= analysisIndex(mi_real_score, mi_fake_score)
#     tcn_mean_true_link, tcn_mean_false_link,tcn_mean_mid_link= analysisIndex(tcn_mean_real_score, tcn_mean_fake_score)
# =============================================================================

    
# 绘图部分
# =============================================================================
#     data_te_mi = np.array([[len(te_true_link.intersection(mi_true_link)), len(te_true_link.intersection(mi_mid_link)), len(te_true_link.intersection(mi_false_link))],
#                       [len(te_mid_link.intersection(mi_true_link)), len(te_mid_link.intersection(mi_mid_link)), len(te_mid_link.intersection(mi_false_link))],
#                       [len(te_false_link.intersection(mi_true_link)), len(te_false_link.intersection(mi_mid_link)), len(te_false_link.intersection(mi_false_link))]])
#     draw(data_te_mi, data_name, ty = data_type, label = ['te', 'mi'])
#     
#     data_te_tcn = np.array([[len(te_true_link.intersection(tcn_mean_true_link)), len(te_true_link.intersection(tcn_mean_mid_link)), len(te_true_link.intersection(tcn_mean_false_link))],
#                       [len(te_mid_link.intersection(tcn_mean_true_link)), len(te_mid_link.intersection(tcn_mean_mid_link)), len(te_mid_link.intersection(tcn_mean_false_link))],
#                       [len(te_false_link.intersection(tcn_mean_true_link)), len(te_false_link.intersection(tcn_mean_mid_link)), len(te_false_link.intersection(tcn_mean_false_link))]])
#     draw(data_te_tcn, data_name, ty = data_type, label = ['te', 'tcn_mean'])
#     
#     data_mi_tcn = np.array([[len(mi_true_link.intersection(tcn_mean_true_link)), len(mi_true_link.intersection(tcn_mean_mid_link)), len(mi_true_link.intersection(tcn_mean_false_link))],
#                       [len(mi_mid_link.intersection(tcn_mean_true_link)), len(mi_mid_link.intersection(tcn_mean_mid_link)), len(mi_mid_link.intersection(tcn_mean_false_link))],
#                       [len(mi_false_link.intersection(tcn_mean_true_link)), len(mi_false_link.intersection(tcn_mean_mid_link)), len(mi_false_link.intersection(tcn_mean_false_link))]])
#     draw(data_mi_tcn, data_name, ty = data_type, label = ['mi', 'tcn_mean'])
# =============================================================================

    
    # print('te_auc:', evaluation_auc(te_real_score, te_fake_score))
    # print('mi_auc:', evaluation_auc(mi_real_score, mi_fake_score))
    # print('tcn_mean_auc:', evaluation_auc(tcn_mean_real_score, tcn_mean_fake_score))

    # diff = te_false_link.intersection(mi_true_link)
    
#通讯数据构建子图

# =============================================================================
#     m_cut = []
#     for i in range(0,8641, 288):
#         m_cut.append(i)
# 
#     SON_G = nx.Graph()
#     son_edges = []
#     son_edges.extend(G.edges(5))
#     for link in G.edges(5):
#         son_edges.extend(G.edges(link[1]))
#         
#     # print(son_edges)
#     SON_G.add_edges_from(son_edges)
#     nx.draw(SON_G, with_labels=True)
#     
#     time_list = dl.generate_time_list(data, 300)
#     divide = int(len(time_list))
#     # print(len(time_list))
#     
#     for node in list(SON_G.nodes):
#         print(node)
#         node_cut = getnodeTime(G, time_list, divide, node)
#         # print(train_graph.edges(node))
#         # print(node_vector.count(1))
#        
#         print(sorted(pd.cut(node_cut, m_cut, labels=False)))
#         print('-' * 10)
#     
#     # 1: 5 2:1327 3:6 4:39 5:47 
#     #6:3167 7:48 8:220 9:554 10:377
#     print(G.edges(220))
#     print(G.edges(554))
# =============================================================================

    

    
    
            
    
    
    
    #科学家合作网络构建子图
    SON_G = nx.Graph()
    son_edges = []
    son_edges.extend(G.edges(10))
    for link in G.edges(10):
        if link[1] != 11:
            son_edges.extend(G.edges(link[1]))
    
    # replace_dict = {10: 1, 926: 2, 387: 3, 927: 4, 928: 5, 929: 6, 388: 7, 389: 8, 1674: 9, 3243: 10}
    
    # print(son_edges)
    
    SON_G.add_edges_from(son_edges)
    SON_G.remove_node(11)
    nx.draw_networkx(SON_G, with_labels=True, font_weight='bold', pos=nx.spring_layout(SON_G))
    # print(SON_G.edges)
    
    time_list = dl.generate_time_list(data, 1)
    divide = int(len(time_list))
    
#通讯数据集
# =============================================================================
#     #mi预测成功TE预测失败
#     mi_ture_te_false = []
#     for real_link in tqdm(SON_G.edges):
#         vector_real = getTime(train_graph, time_list, divide, real_link)
#         real_mi = mi_cal(vector_real[0], vector_real[1])
#         real_te = (te_cal(vector_real[0], vector_real[1], 1) + te_cal(vector_real[1], vector_real[0], 1)) / 2
# 
#         
#         for fake_link in nx.non_edges(SON_G):
#             vector_fake = getTime(train_graph, time_list, divide, fake_link)
#             fake_mi = mi_cal(vector_fake[0], vector_fake[1])
#             fake_te = (te_cal(vector_fake[0], vector_fake[1], 1) + te_cal(vector_fake[1], vector_fake[0], 1)) / 2
#             # print(real_te, fake_te)
#             # print(real_mi, fake_mi)
#             if real_mi > fake_mi and real_te < fake_te:
#                 mi_ture_te_false.append((real_link, fake_link))
#                 
#     # print(mi_ture_te_false)
# 
# 
# #筛选连边
#     link_fin = []
#     for links in mi_ture_te_false:
#         if TCN_mean(train_graph, links[0]) < TCN_mean(train_graph, links[1]):
#             if TCN_min(train_graph, links[0]) > TCN_min(train_graph, links[1]):
#                 link_fin.append(links)
#     if len(link_fin) > 0:
#         for link in link_fin:
#             print(link)
# =============================================================================


# 通讯网络连边筛选结果
# =============================================================================
#     real_links = {'U4->U9':(39, 554),'U2->U6':(1327, 3167)}
#     fake_links = {'U1->U5':(5, 47),'U1->U7':(5, 48)}
#     
#     for key in real_links:
#         
#         vector = getTime(train_graph, time_list, divide, real_links[key])
#         
#         mi = mi_cal(vector[0], vector[1])
#         te_1 = te_cal(vector[0], vector[1], 1)
#         te_2 = te_cal(vector[1], vector[0], 1)
#         te = (te_1 + te_2) / 2
#         min_ = TCN_min(train_graph, real_links[key])
#         mean_ = TCN_mean(train_graph, real_links[key])
#         
#         print('{0}:\nMI:{1}\nTE(X->Y):{2}\nTE(Y->X):{3}\nTE:{4}\nMIN:{5}\nMEAN:{6}'.format(key,mi,te_1, te_2, te, min_, mean_))
#         print('-' * 10)
#         
#     for key in fake_links:
#         
#         vector = getTime(train_graph, time_list, divide, fake_links[key])
#         
#         mi = mi_cal(vector[0], vector[1])
#         te_1 = te_cal(vector[0], vector[1], 1)
#         te_2 = te_cal(vector[1], vector[0], 1)
#         te = (te_1 + te_2) / 2
#         min_ = TCN_min(train_graph, fake_links[key])
#         mean_ = TCN_mean(train_graph, fake_links[key])
#         
#         print('{0}:\nMI:{1}\nTE(X->Y):{2}\nTE(Y->X):{3}\nTE:{4}\nMIN:{5}\nMEAN:{6}'.format(key,mi,te_1, te_2, te, min_, mean_))
#         print('-' * 10)
# =============================================================================
    
                        


# 科学家合作网络连边筛选结果
    real_links = {'SC1->SC3':(10, 387),'SC2->SC4':(926, 927)}
    fake_links = {'SC1->SC10':(10, 3243),'SC10->SC2':(3243, 926)}
    
    print(SON_G.edges(10))
    
    for key in real_links:
        
        vector = getTime(train_graph, time_list, divide, real_links[key], 'science')
        print(vector)
        mi = mi_cal(vector[0], vector[1])
        te_1 = te_cal(vector[0], vector[1], 1)
        te_2 = te_cal(vector[1], vector[0], 1)
        te = (te_1 + te_2) / 2
        min_ = TCN_min(train_graph, real_links[key])
        mean_ = TCN_mean(train_graph, real_links[key])
        
        print('{0}:\nMI:{1}\nTE(X->Y):{2}\nTE(Y->X):{3}\nTE:{4}\nMIN:{5}\nMEAN:{6}'.format(key,mi,te_1, te_2, te, min_, mean_))
        print('-' * 10)
        
    for key in fake_links:
        
        vector = getTime(train_graph, time_list, divide, fake_links[key], 'science')
        
        mi = mi_cal(vector[0], vector[1])
        te_1 = te_cal(vector[0], vector[1], 1)
        te_2 = te_cal(vector[1], vector[0], 1)
        te = (te_1 + te_2) / 2
        min_ = TCN_min(train_graph, fake_links[key])
        mean_ = TCN_mean(train_graph, fake_links[key])
        
        print('{0}:\nMI:{1}\nTE(X->Y):{2}\nTE(Y->X):{3}\nTE:{4}\nMIN:{5}\nMEAN:{6}'.format(key,mi,te_1, te_2, te, min_, mean_))
        print('-' * 10)
    
    

#构建散点图数据集
# =============================================================================
#     node_time_list = []
#     #构建散点图数据集
#     flag = 0
#     for node in SON_G.nodes:
#         # print(node)
#         time = [str(G[i[0]][i[1]]['t']) for i in list(G.edges(node))]
#         node_time_list.append(pd.Series(time, name = node))
#         flag += 1
#     #需要其它类型的数据结构
#     # print(node_time_list)
#     time_data = {str(key):[] for key in range(1984, 2004)}
#     for node in node_time_list:
#         for year in node.values:
#             if year in time_data.keys():
#                 time_data[year].append(int(node.name))
#     # print(time_data)
#     
# =============================================================================
    
    # print(df)
    
#从整体分析差异
# =============================================================================
#     # for links_i in list(diff):
#         # real_link = links[0]
#         # fake_link = links[1]
#         # print(real_link)
#         # print(fake_link)
#         # divide = int(len(time_list))
#         # vector_real = getTime(train_graph, time_list, divide, real_link)
#         # vector_fake = getTime(train_graph, time_list, divide, fake_link)
#         # real_te_1 = te_cal(vector_real[0], vector_real[1], 1)
#         # real_te_2 = te_cal(vector_real[1], vector_real[0], 1)
#         # fake_te_1 = te_cal(vector_fake[1], vector_fake[0], 1)
#         # fake_te_2 = te_cal(vector_fake[1], vector_fake[0], 1)
#         # real_te = real_te_1 + real_te_2 / 2
#         # fake_te = fake_te_1 + fake_te_2 / 2
#         # print('real_te_1', real_te)
#         # print('real_te_2', real_te)
#         # print('real_te', real_te)
#         # print('fake_te_1', fake_te_1)
#         # print('fake_te_2', fake_te_2)
#         # print('fake_te', fake_te)
#         # print('real_H:',Entropy(vector_real[0]),Entropy(vector_real[1]))
#         # print('fake_H:',Entropy(vector_fake[0]),Entropy(vector_fake[1]))
# 
#         # real_mi = mi_cal(vector_real[0], vector_real[1])
#         # fake_mi = mi_cal(vector_fake[0], vector_fake[1])
#         # print('')
#         # print('real_Mi', real_mi)
#         # print('fake_Mi', fake_mi)
#         # print('-' * 10)
# =============================================================================

diff = analysisDiff('BMJ', 'science')
# diff = analysisDiff('SD03', 'communication')
