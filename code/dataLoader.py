# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:39:18 2022

@author: ShengyueJiang
"""

from sklearn.model_selection import train_test_split
import networkx as nx
from random import choice, randint,seed
import pandas as pd

def data_load(name, type_ = 'science'):
    if type_ == 'science':
        data_path = '../data/{}.csv'.format(name)
    else:
        data_path = '../data/{}.csv'.format(name)
    data = pd.read_csv(data_path, sep=',')
    if type_ == 'science':
        data = data[(data['t'] < data.loc[0,'t'] + 21)]
    elif type_ == 'communication':
        data = data[(data['t'] < data.loc[0,'t'] + 2592000)]
        #3235949
    data['w'] = 1
    # 此部分权重会打乱数据，不能应用在互信息预测中
    # data = data.groupby(data.columns.tolist()).size().reset_index().rename(columns={0:'w'})
    return data

def data_div(data, index_type='topo'):
    
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=300)
    
    if index_type == 'topo' or index_type == 'weight 'or index_type == 'time' or index_type == 'information' :
        G = nx.from_pandas_edgelist(data, 's', 'e', ['t', 'w'], create_using=nx.Graph())
        train_graph = nx.from_pandas_edgelist(train_data, 's', 'e', ['t', 'w'], create_using=nx.Graph())
        test_graph = nx.from_pandas_edgelist(test_data, 's', 'e', ['t', 'w'], create_using=nx.Graph())
        
    if index_type == 'behavior':
        G = nx.from_pandas_edgelist(data, 's', 'e', ['t', 'w'], create_using=nx.Graph())
        train_graph = nx.from_pandas_edgelist(train_data, 's', 'e', ['t', 'w'], create_using=nx.DiGraph())
        test_graph = nx.from_pandas_edgelist(test_data, 's', 'e', ['t', 'w'], create_using=nx.DiGraph())
    

    for node in list(test_graph.nodes()):
        if node not in train_graph.nodes():
            test_graph.remove_node(node)
    
    # for node in G.nodes:
    #     if len(G.neighbors(node)) == 0:
    #         G.remove(node)
    
    return G,train_graph, test_graph

def generate_no_list(train_graph, test_graph):

    no_list = []
    while len(no_list) < len(list(test_graph.edges())):
        # seed(1545)
        # randint(0, len(list(train_graph.nodes())))
        # randint(0, len(list(train_graph.nodes())))
        index_1 = choice(list(train_graph.nodes()))
        index_2 = choice(list(train_graph.nodes()))
        try:
            train_graph[index_1][index_2] > 0
        except:
            if index_1 != index_2:
                no_list.append((min(index_1, index_2), max(index_1, index_2)))
    
    return no_list

def generate_time_list(data, step):
    
    time_list = [t for t in range(int(data.iloc[0]['t']), int(data.iloc[-1]['t']) + 1, step)]
    
    if int(data.iloc[-1]['t']) not in time_list:
        time_list.append(int(data.iloc[-1]['t']))
    
    # while time_list[-1] <= 3592000:
    #     time_list.append(time_list[-1] +300)
        
    return time_list

