# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 19:48:36 2022

@author: ShengyueJiang
"""
from random import choice

def evaluation_auc(real_edges,fake_edges):
    AUC_result = 0.0
    for i in range(len(real_edges)):   
        if real_edges[i][1] > fake_edges[i][1]:
            AUC_result = AUC_result + 1
        elif real_edges[i][1] == fake_edges[i][1]:
            AUC_result = AUC_result + 0.5
            
    auc = round(AUC_result / len(real_edges), 3)
    # print('AUC:',auc)
    return auc


def evaluation_precision(real_edges, fake_edges, l):
    top_l = []
    i = 0
    j = 0
    m = 0 
    cn_real = sorted(real_edges, key=lambda x: x[1], reverse=True)
    cn_false = sorted(fake_edges, key=lambda x: x[1], reverse=True)

    while len(top_l) <= l:
        if cn_real[i][1] > cn_false[j][1]:
            top_l.append(cn_real[i])
            i += 1
        elif cn_real[i][1] < cn_false[j][1]:
            top_l.append(cn_false[j])
            j += 1
        else:
            same = [cn_real[i], cn_false[j]]
            a = choice(same)
            top_l.append(a)
            same.remove(a)
            top_l.append(same)
            i += 1
            j += 1
    for i in range(l):
        if top_l[i] in cn_real[0:l - 1]:
            m = m + 1
    p = round(m / l, 3)
    # print('Precision:', p)
    return p