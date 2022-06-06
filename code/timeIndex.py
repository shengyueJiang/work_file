# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:11:20 2022

@author: ShengyueJiang
"""

def TCN(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    tcn = 0.0
    for z in neigh_ij:
        tcn += abs(1.0 / G[node_i][z]['t'] - G[node_j][z]['t'])
    return tcn

def TRA(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j)
    tra = 0.0
    for z in neigh_ij:
        tra += 1.0 / G.degree(z) * abs(G[node_i][z]['t'] - G[node_j][z]['t'])
    return tra

def TLHN(G, nodeij):
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    return TCN(G, nodeij) / (len(neigh_i) * len(neigh_j))

def TJACC(G, nodeij):
    #return self.index_tcn() / len(self.total_neighbors())

    node_i = nodeij[0]
    node_j = nodeij[1]
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i | neigh_j
    return TCN(G, nodeij) / len(neigh_ij)