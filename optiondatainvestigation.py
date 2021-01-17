# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:19:09 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import Backtest as bt
import matplotlib.pyplot as plt
import sys

## Option Data Investigation
OptionData        = pd.read_csv(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\CleanData\SPXOptionDataClean.csv")
OptionDataToTrade = pd.read_csv(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\CleanData\SPXOptionDataToTrade.csv")
UnderlyingData    = pd.read_csv(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\CleanData\SPXUnderlyingData.csv")
UnderlyingAssetName = "S&P 500 Index"

startDate = 19960102
endDate   = 20200101

OptionData  = bt.trimToDates(OptionData, OptionData["date"], startDate, endDate)
print(OptionData.head())
print(OptionData.tail())

UnderlyingDataTr = bt.trimToDates(UnderlyingData, UnderlyingData["Dates"], startDate, endDate)
print(UnderlyingData.head())
print(UnderlyingData.tail())

OptionDataToTrade = bt.trimToDates(OptionDataToTrade, OptionDataToTrade["date"], startDate, endDate)
print(OptionDataToTrade.head())
print(OptionDataToTrade.tail())

#Delete options that violate arbitrage bounds
gamma         = OptionData["gamma"].to_numpy()
non_violating = np.isfinite(gamma)
OptionDataTr  = OptionData.loc[non_violating, :]

#Underlying data
UnderlyingDates  = UnderlyingDataTr["Dates"].to_numpy()
UnderlyingPrices = UnderlyingDataTr["Price"].to_numpy()
#UnderlyingDates  = UnderlyingDates[1:]
#UnderlyingPrices = UnderlyingPrices[1:]

#Option Dates
OptionDates      = OptionDataTr["date"].to_numpy()
UniqueDates      = np.unique(OptionDates)
nOptionDays      = np.size(UniqueDates)

#Grab data needed
gamma        = OptionDataTr["gamma"].to_numpy()
delta        = OptionDataTr["delta"].to_numpy()
openInterest = OptionDataTr["open_interest"].to_numpy()
volume       = OptionDataTr["volume"].to_numpy()
callFlag     = OptionDataTr["cp_flag"].to_numpy()


#Compute timeseries of daily aggregate statistics
netGamma                = np.zeros((nOptionDays, 1))
netGamma_alt            = np.zeros((nOptionDays, 1))
aggOpenInterest         = np.zeros((nOptionDays, 1))
netOpenInterest         = np.zeros((nOptionDays, 1))
deltaAdjOpenInterest    = np.zeros((nOptionDays, 1))
deltaAdjNetOpenInterest = np.zeros((nOptionDays, 1))
aggVolume               = np.zeros((nOptionDays, 1))
deltaAdjVolume          = np.zeros((nOptionDays, 1))

for i in np.arange(0, nOptionDays):
    day = UniqueDates[i] #grab day
    isRightDay = (OptionDates == day) #boolean for right day
    
    #Grab values for relevant day
    rel_callFlag     = callFlag[isRightDay]
    rel_gamma        = gamma[isRightDay]
    rel_delta        = delta[isRightDay]
    rel_openInterest = openInterest[isRightDay]
    rel_volume       = volume[isRightDay]
    
    #Gamma Exposure
    rel_gamma_call  = rel_gamma[rel_callFlag == 1] * rel_openInterest[rel_callFlag == 1]*100
    rel_gamma_put   = rel_gamma[rel_callFlag == 0] * rel_openInterest[rel_callFlag == 0]*100
    netGamma[i]     = np.sum(rel_gamma_call) - np.sum(rel_gamma_put)
    netGamma_alt[i] = np.sum(rel_gamma_call) + np.sum(rel_gamma_put)
    
    #Agg and net open interest
    aggOpenInterest[i]  = np.sum(rel_openInterest)*100
    netOpenInterest[i]  = np.sum(rel_openInterest[rel_callFlag == 1]) - np.sum(rel_openInterest[rel_callFlag == 0])
    
    #delta adjusted open interest
    deltaAdjOpenInterest[i]    = np.sum(np.abs(rel_delta) * 100 * rel_openInterest)
    deltaAdjNetOpenInterest[i] = np.sum(rel_delta * 100 * rel_openInterest)
    
    #volume
    aggVolume[i] = np.sum(rel_volume)
    deltaAdjVolume[i] = np.sum(np.abs(rel_delta)*100 * rel_volume)
    
    
#Construct array with aggregate data
aggregateData = np.concatenate((UniqueDates.reshape(nOptionDays, 1), netGamma, netGamma_alt, aggOpenInterest,\
                netOpenInterest, deltaAdjOpenInterest, deltaAdjNetOpenInterest, aggVolume, deltaAdjVolume), axis = 1)    

    
#Fill missing values of aggregate data w/ previous day value
nRows = np.size(UnderlyingDates)
nCols = np.size(aggregateData, 1)

syncMat    = np.zeros((nRows, nCols))
date_bool  = np.in1d(UnderlyingDates, UniqueDates) #Dates where option data is recorded
date_shift = np.concatenate((date_bool[1:], np.ones((1,))), axis = 0) == 1 

syncMat[date_bool, :] = aggregateData
syncMat[(date_bool == 0), 1:] = syncMat[(date_shift == 0), 1:]    
syncMat[(date_bool == 0), 0]  = UnderlyingDates[(date_bool == 0)]     
aggregateData = syncMat #overwrite    


returns = np.concatenate((np.zeros((1,)), UnderlyingPrices[1:] / UnderlyingPrices[0:-1] - 1), axis = 0)
dates4fig = pd.to_datetime(aggregateData[:, 0], format = '%Y%m%d')


#Gamma Exposure Plot
lag = 1 
plt.figure()
plt.scatter(aggregateData[0:-lag, 1], np.abs(returns[lag:]), color = "blue", s = 3)
plt.title("Gamma Exposure vs Absolute Returns, Lag = " + str(lag) + " Day(s)")
plt.xlabel("Market Maker Net Gamma Exposure")
plt.ylabel("Absolute Returns")


#Open Interest
plt.figure()
plt.plot(dates4fig, aggregateData[:, 3], label = "Aggregate Open Interest")
plt.title("Aggregate Open Interest, " + str(UnderlyingAssetName) + " Options")
plt.ylabel("Net Daily Open Interest")


#Delta Adjusted Volume
plt.figure()
plt.plot(dates4fig, aggregateData[:, -1], label = "Delta Adjusted Volume")
plt.title("Delta Adjusted Volume, " + str(UnderlyingAssetName) + " Options")
plt.ylabel("Delta Adjusted Volume ($)")
    
    
    
    
    
    
    
    
    
