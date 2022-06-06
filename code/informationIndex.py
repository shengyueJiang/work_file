# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:14:50 2022

@author: ShengyueJiang
"""
from ctypes import c_double
from ctypes import c_int
from ctypes import byref
from ctypes import CDLL
from ctypes import POINTER
from ctypes import c_long
import dataLoader as dl
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
from pyinform.transferentropy import *
import topoIndex

def getTime(G, time_list, divide, nodeij, type_):
    
    x = nodeij[0]
    y = nodeij[1]
    
    x_vector = [0 for i in time_list]
    y_vector = [0 for i in time_list]
    #所有关于x的时间和关于y的时间
    time_x = [G[i[0]][i[1]]['t'] for i in list(G.edges(x))]
    # print(time_x)
    time_y = [G[i[0]][i[1]]['t'] for i in list(G.edges(y))]
    # print(list(G.edges(y)))
    # print(time_y)

    # print(time_list)
    x_cut = pd.cut(time_x, time_list, labels=False)
    # print(x_cut)
    y_cut = pd.cut(time_y, time_list, labels=False)
    
    if type_ == 'communication':

        
        # x_vector = [0 for i in range(divide)]
        
        for v in x_cut:

            try:
                x_vector[int(v)] = 1
            except Exception:
                pass
        
        y_vector = [0 for i in range(divide)]
        for v in y_cut:
        
            try:
                y_vector[int(v)] = 1
            except Exception:
                pass  
        return x_vector, y_vector
        # for t in time_list:
        #     if t in time_x:
        #         x_vector[time_list.index(t)] = 1
        #     else:
        #         x_vector[time_list.index(t)] = 0
        
        # for t in time_list:
        #     if t in time_y:
        #         y_vector[time_list.index(t)] = 1
        #     else:
        #         y_vector[time_list.index(t)] = 0

        # return x_vector, y_vector
    
    else:
        
        # for t in time_list:
        #     if t in time_x:
        #         x_vector[time_list.index(t)] = 1
        #     else:
        #         x_vector[time_list.index(t)] = 0
        
        # for t in time_list:
        #     if t in time_y:
        #         y_vector[time_list.index(t)] = 1
        #     else:
        #         y_vector[time_list.index(t)] = 0
        
        for s in range(divide):
            if s in x_cut:
                x_vector.append(1)
            else:
                x_vector.append(0)

        for s in range(divide):
            if s in y_cut:
                y_vector.append(1)
            else:
                y_vector.append(0)
        
        return x_vector, y_vector
    
def getnodeTime(G, time_list, divide, node):
    
    x = node

    
    x_vector = [0 for i in time_list]

    #所有关于x的时间和关于y的时间
    time_x = [G[i[0]][i[1]]['t'] for i in list(G.edges(x))]
    # print(time_x)
    
    x_cut = pd.cut(time_x, time_list, labels=False)
    
    x_vector = [0 for i in range(divide)]
    
    for v in x_cut:

        x_vector[v] = 1
    

    return x_cut

def te_cal(X, Y, s):
    
    num = len(X)
    len_ = c_double(len(X))
    s = c_int(s)

    a_list = (c_int * len(X))()
    for i in range(len(X)):
        a_list[i] = X[i] 

    b_list = (c_int * len(Y))()
    for i in range(len(Y)):
        b_list[i] = Y[i] 

    dll = CDLL('te_cal_0.dll')
    dll.te.restype = c_double
    dll.te.argtypes = (POINTER(c_long * num), POINTER(c_long * num), c_int, c_double)

    te = dll.te(byref(a_list), byref(b_list), s, len_)
    return te

def MI(G, nodeij, time_list, type_):
    
    divide = int(len(time_list))
    x_vector, y_vector = getTime(G, time_list, divide, nodeij, type_)
    return normalized_mutual_info_score(x_vector, y_vector)


def TE(G, nodeij, time_list, type_):
    
    divide = int(len(time_list))
    x, y = getTime(G, time_list, divide, nodeij, type_)
    # return (_te_cal(x_vector, y_vector,1) + _te_cal(y_vector, x_vector,1))/2
    # if x == y:
    #     return 1
    # else:
    te_x = te_cal(x, y,1)
    te_y = te_cal(y, x,1)
    # if te_x == 0:
    #     te_x = 1
    # elif te_y == 0:
    #     te_y = 1
        
    return (te_x + te_y) / 2
    # return min(te_x, te_y)


def CN_TE(G, nodeij, time_list, type_, a = 0.5):

    return a * TE(G, nodeij, time_list, type_)+ (1-a) * topoIndex.CN(G, nodeij)

def CN_MI(G, nodeij, time_list, type_, a = 0.5):
    
    return a * MI(G, nodeij, time_list, type_) + (1-a) * topoIndex.CN(G, nodeij)

