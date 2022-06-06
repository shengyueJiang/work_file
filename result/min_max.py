# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:06:46 2022

@author: ShengyueJiang
"""

import pandas as pd

auc = pd.read_csv('auc.csv', index_col='index')
pre = pd.read_csv('precision.csv', index_col='index')



for data_name in auc:
    print((auc[data_name]['CN_MI'] - auc[data_name].min()) / auc[data_name].min())

        
# for data_name in pre:
#     print((auc[data_name]['CN_MI'] - auc[data_name].min()) / auc[data_name].min())
#     if pre[data_name]['CN_MI'] != pre[data_name].max():
#         print('2')
#         print(data_name)