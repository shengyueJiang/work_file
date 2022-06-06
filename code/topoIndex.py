# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 08:42:33 2022

@author: ShengyueJiang
"""
import numpy as np

def CN(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    num_cn = len(neigh_ij)
    
    return num_cn

def AA(G, nodeij):

    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    aa = 0.0
    if len(neigh_ij) > 0:
        for k in neigh_ij:
            degree_k = G.degree(k)
            if degree_k > 1:
                aa = aa + 1 / np.math.log10(degree_k)
    return aa

def RA(G, nodeij):

    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    ra = 0.0
    if len(neigh_ij) > 0:
        for k in neigh_ij:
            degree_k = G.degree(k)
            if degree_k > 0:
                ra = ra + 1.0 / degree_k
    return ra

def JACC(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    
    return CN(G, nodeij) / len(neigh_i | neigh_j)

def LHN(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
          
    return CN(G, nodeij) / len(neigh_i) * len(neigh_j)
