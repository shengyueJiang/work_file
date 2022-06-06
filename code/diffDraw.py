# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:24:45 2022

@author: ShengyueJiang
"""

import numpy as np
import matplotlib.pyplot as plt
from palettable.cartocolors.sequential import Sunset_3, TealGrn_3

def draw(data, data_name, ty='', label = []):
    te = ['True', 'Mid', 'False']
    mi = ['True', 'Mid', 'False']
    
    fig, ax = plt.subplots()  
    #将元组分解为fig和ax两个变量 
    if ty == 'science':
        im = ax.imshow(data, cmap=Sunset_3.mpl_colormap)
    else:
        im = ax.imshow(data, cmap=Sunset_3.mpl_colormap)
    #显示图片
    
    
    ax.set_xticks(np.arange(len(mi)))    
    #设置x轴刻度间隔
    ax.set_yticks(np.arange(len(te)))    
    #设置y轴刻度间隔
    ax.set_xticklabels(mi)        
    #设置x轴标签'''
    ax.set_yticklabels(te)     
    ax.set_xlabel(str.upper(label[1]))
    ax.set_ylabel(str.upper(label[0]))
    
    plt.figure(dpi=600)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    
    for i in range(len(te)):
        for j in range(len(mi)):
            if i == 0 and j == 1:
                text = ax.text(j, i, str(data[i, j]) + '*',
                               ha="center", va="center", color="black")
            elif i == 0 and j == 2:
                text = ax.text(j, i, str(data[i, j]) + '*',
                               ha="center", va="center", color="black")
            elif i == 2 and j == 0:
                text = ax.text(j, i, str(data[i, j]) + '*',
                               ha="center", va="center", color="black")
            elif i == 1 and j == 0:
                text = ax.text(j, i, str(data[i, j]) + '*',
                               ha="center", va="center", color="black")
            else:
                text = ax.text(j, i, str(data[i, j]),
                               ha="center", va="center", color="black")
    ax.set_title(data_name)      
    fig.tight_layout()  #自动调整子图参数,使之填充整个图像区域。
    plt.show()      #图像展示
