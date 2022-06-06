# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:30:08 2022

@author: ShengyueJiang
"""

import random
import dataLoader as dl
from gensim.models import Word2Vec
import numpy as np
from evaluation import evaluation_auc,evaluation_precision
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from embeddingIndex import deepWalk, node2vec
from tqdm import tqdm
import networkx as nx

DATA_LIST = {
    'science':['BMJ','nature','science','jama','england'],
    'communication':['msg', 'email', 'SD01', 'SD02', 'SD03']
    }

for data_type in DATA_LIST:
    
    for data_name in DATA_LIST[data_type]:
    
    
        data = dl.data_load(data_name, data_type)
        
        print('---building {0} net---'.format(data_name))
        
        G,train_graph, test_graph = dl.data_div(data)
        
        print(nx.average_clustering(G))
        print('node:', len(G.nodes))
        print('edge:', len(G.edges))
        if data_type == 'science':
            print(data['t'].min())
            print(data['t'].max())
        
        
        print('---generate no list {0} net---'.format(data_name))
        no_list = dl.generate_no_list(train_graph, test_graph)
        
        
        real_model = deepWalk(train_graph, test_graph.edges)
        fake_model = deepWalk(train_graph, no_list)
        
        # real_model = node2vec(train_graph, test_graph.edges)
        # fake_model = node2vec(train_graph, no_list)
        
        real = []
        fake = []
        
        for s,t in tqdm(test_graph.edges):
            if real_model.wv.has_index_for(s) and real_model.wv.has_index_for(t):
                real.append(((s, t), real_model.wv.similarity(s, t)))
            else:
                real.append(((s, t), 0))
        
        for s,t in tqdm(no_list):
            if fake_model.wv.has_index_for(s) and fake_model.wv.has_index_for(t):
                fake.append(((s, t), fake_model.wv.similarity(s, t)))
            else:
                fake.append(((s, t), 0))
                
        # print(len(real[:min(len(real), len(fake))]))
        # print(len(fake[:min(len(real), len(fake))]))
        
        print(data_name, evaluation_auc(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))]))
        print(data_name, evaluation_precision(real[:min(len(real), len(fake))], fake[:min(len(real), len(fake))], len(test_graph.edges())))
