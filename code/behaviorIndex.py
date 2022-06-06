# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:36:00 2022

@author: ShengyueJiang
"""
import networkx as nx
from topoIndex import CN

DATA_LIST = ['msg', 'email', 'SD01', 'SD02', 'SD03']  

def mean(l):
    s = 0.0
    for i in l:
        s += i
    return s / len(l)

def list_min(main_list, target):
    
    temp_dif = []
    for i in main_list:
        temp_dif.append((abs(i - target)))
    return min(temp_dif)
        

def TCN_mean(G, edge):
    
    node_x = edge[0]
    node_y  = edge[1]
    node_x_action = []
    node_y_action = []
    
    cn = CN(G, edge)
    
    node_x_action = G.edges(node_x)
    node_y_action = G.edges(node_y)
        
    
    if len(node_x_action) == 0 or len(node_y_action) == 0: 
        return cn

    action_x_mean = []
    action_y_mean = []
    
    for action in node_x_action:
        if len(node_y_action) == 0:
            return 0
        action_x_mean.append(list_min([G[action[0]][action[1]]['t'] for action in node_y_action], G[action[0]][action[1]]['t']))
    for action in node_y_action:
        if len(node_y_action) == 0:
            return 0 
        action_y_mean.append(list_min([G[action[0]][action[1]]['t'] for action in node_x_action], G[action[0]][action[1]]['t']))

    d_xy = mean(action_x_mean)
    d_yx = mean(action_y_mean)

    
    return (-(d_xy + d_yx) / 2.0)

def TCN_min(G, edge):
    
    node_x = edge[0]
    node_y  = edge[1]
    node_x_action = []
    node_y_action = []
    
    cn = CN(G, edge)
    
    node_x_action = G.edges(node_x)
    node_y_action = G.edges(node_y)
        
    
    if len(node_x_action) == 0 or len(node_y_action) == 0: 
        return cn
        
    action_x_min = []
    action_y_min = []
    
    for action in node_x_action:
        if len(node_y_action) == 0:
            return cn          
        action_x_min.append(list_min([G[action[0]][action[1]]['t'] for action in node_y_action], G[action[0]][action[1]]['t']))

        
    for action in node_y_action:       
        if len(node_y_action) == 0:
            return cn     
        action_y_min.append(list_min([G[action[0]][action[1]]['t'] for action in node_x_action], G[action[0]][action[1]]['t']))
        
    d_xy = min(action_x_min)
    d_yx = min(action_y_min)
    
    return (-(d_xy + d_yx) / 2.0)

def TCN_mean_CN(G, nodeij, a = 0.5):
    return a * TCN_mean(G, nodeij) + (1-a) * CN(G, nodeij)
    
def TCN_min_CN(G, nodeij, a = 0.5):
    return a * TCN_min(G, nodeij) + (1-a) * CN(G, nodeij)