# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:21:11 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import Backtest as bt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

# Option Market Size vs Underlying
### SET WHICH ASSET TO BE IMPORTED #######################################################
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load data
SPXDataAll = pd.read_csv(loadloc + "SPXAggregateData.csv") #SPX
SPYDataAll = pd.read_csv(loadloc + "SPYAggregateData.csv") #SPY


    
