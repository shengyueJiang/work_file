# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:48:07 2022

@author: ShengyueJiang
"""
import networkx as nx
from math import log
# 'WCN', 'WAA', 'WRA', 'rWCN', 'rWAA', 'rWRA'
# 验证无向网络，取一条边的概念

def WCN(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    wcn = 0.0
    for z in neigh_ij:
        wcn += G[node_i][z]['w'] + G[node_j][z]['w']
    return wcn

def WAA(G, nodeij):
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    waa = 0.0
    for z in neigh_ij:
        waa += (G[node_i][z]['w'] + G[node_j][z]['w']) / log(1 + G.degree(z))
    return waa

def WRA(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    wra = 0.0
    for z in neigh_ij:
        degree = G.degree(z)
        if degree == 1 :
            degree = 2
        wra += (G[node_i][z]['w'] + G[node_j][z]['w']) / log(degree)
    return wra

def rWCN(G, nodeij):
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    rwcn = 0.0
    for z in neigh_ij:
        rwcn += G[node_i][z]['w'] * G[node_j][z]['w']
    return rwcn

def rWAA(G, nodeij):
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    rwaa = 0.0
    for z in neigh_ij:
        rwaa += (G[node_i][z]['w'] * G[node_j][z]['w']) / log(1 + G.degree(z))
    return rwaa

def rWRA(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    rwra = 0.0
    for z in neigh_ij:
        degree = G.degree(z)
        if degree == 1 :
            degree = 2
        rwra += (G[node_i][z]['w'] * G[node_j][z]['w']) / log(degree)
    return rwra
