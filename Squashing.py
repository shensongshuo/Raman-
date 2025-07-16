# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:09:02 2019

@author: Administrator
"""


import numpy as np
import math
import matplotlib.pyplot as plt
# import crossvalidation  as cr
def squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range (row):
        for j in range (col):
            sqData[i][j] =( 1-math.cos(Data[i][j]*math.pi))/2
    return sqData
'''
Data , _=cr.getdata(["E:\\数据\\用于中期答辩作图"])
wavenumbers = np.loadtxt("wavenumbers.txt")
#Data = Data.reshape((1 , 2000))

s = squashing(Data)
#s = s.reshape((2000 , 1))
for i in range (s.shape[0]):
    plt.figure(1)
    plt.plot(wavenumbers , s[i])
    plt.title('MCF-10A')
    plt.xlabel('Raman Shift(cm^{-1})')
    plt.ylabel('Intensity(a.u.)')
    plt.legend
    plt.show
    plt.figure(2)
    plt.plot(wavenumbers ,  Data[i])
    plt.title('MCF-10A')
    plt.xlabel('Raman Shift(cm^{-1})')
    plt.ylabel('Intensity(a.u.)')
    plt.legend
    plt.show
 '''