# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:00:14 2022

@author: ShengyueJiang
"""

import pandas as pd
import matplotlib.pyplot as plt

def boxDraw(data_list, label):
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 8))
    plt.grid(True)  # 显示网格
    plt.boxplot(data_list,labels=label)
    plt.xlabel('DATA', fontsize = 15) 
    plt.ylabel('H', fontsize = 15)
    plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=15)
           # patch_artist = True,
           # boxprops = {'color':'black'},

           # flierprops = {'marker':'o', 'markerfacecolor':'red', 'color':'black'},
           # meanprops = {'marker':'D', 'markerfacecolor':'indianred'},
           # medianprops = {'linestyle':'--', 'color':'orange'})  # 绘制箱形图，设置异常点大小、样式等
    plt.show()  # 


DATA_LIST = {
    'science':['BMJ','nature','science','jama','england'],
    'communication':['msg', 'email'],
    'messge': ['SD01', 'SD02', 'SD03']
    }


for data_type in DATA_LIST:
    data_list = []
    for data_name in DATA_LIST[data_type]:
        
        data_path = '../result/analysis/{}_H.csv'.format(data_name)
        data = pd.read_csv(data_path, index_col=0)
        # print(data_name, round(data.mean(), 3))
        print(data_name, data.var())
        data_list.append(data['H'])
    # print(data_list)
    boxDraw(data_list, label = DATA_LIST[data_type])