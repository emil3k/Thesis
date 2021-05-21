# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:32:57 2021

@author: ekblo
"""


import numpy as np
import pandas as pd
import Backtest as bt

UnderlyingTicker      = "NDX"
UnderlyingTicker2     = "QQQ"
loadlocAgg            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
loadlocOpt            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
loadlocSpot           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/SpotData/SpotData.xlsx"



IndexOptionsData    = pd.read_csv(loadlocOpt + UnderlyingTicker + "OptionDataClean.csv")
ETFOptionsData      = pd.read_csv(loadlocOpt + UnderlyingTicker2 + "OptionDataClean.csv")

startDate = 20190102
endDate   = 20190202

IndexOptionDataTr   = bt.trimToDates(IndexOptionsData, IndexOptionsData["date"], startDate, endDate + 1)
ETFOptionDataTr   = bt.trimToDates(ETFOptionsData, ETFOptionsData["date"], startDate, endDate + 1)

Index = IndexOptionDataTr[(IndexOptionDataTr["exdate"] == 20190118)]
ETF = ETFOptionDataTr[(ETFOptionDataTr["exdate"] == 20190118)]

    