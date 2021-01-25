# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:09:23 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Backtest as bt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import sys

#Trading Strategy Development
### SET IMPORT PARAMS ####################################################################
UnderlyingAssetName   = "SPX Index"
UnderlyingTicker      = "SPX"
loadlocAgg            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateDataTr/"
loadlocOpt            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData   = pd.read_csv(loadlocAgg + UnderlyingTicker + "AggregateDataTr.csv")
AggregateDates  = AggregateData["Dates"].to_numpy()
startDate = AggregateDates[0]
endDate   = AggregateDates[-1]

#Trim Options data for match aggregate data
OptionsData    = pd.read_csv(loadlocOpt + UnderlyingTicker + "OptionDataToTrade.csv")
OptionDataTr   = bt.trimToDates(OptionsData, OptionsData["date"], startDate, endDate + 1)
OptionDates    = OptionDataTr["date"].to_numpy()
UniqueDates    = np.unique(OptionDates)

#Signal
netGamma = AggregateData["netGamma"].to_numpy()

#Trading strategy 1)
#Buy front month ATM straddles whenever gamma is negative
#Buy back month if front month has less than 1-week to expiry
#Hold these options until gamma turns positive
#Roll 1-week before expiration

OptionCols      = OptionDataTr.columns
OptionDataArr   = OptionDataTr.to_numpy()
FrontATMCalls = np.zeros((1, np.size(OptionCols)))
FrontATMPuts  = np.zeros((1, np.size(OptionCols)))

nDays           = np.size(UniqueDates)
currentlyInvested = 0 #initialize no position
putPortfolio = np.zeros((nDays, ))
error = []
for i in np.arange(0, nDays):
            
    day = UniqueDates[i] #grab day
    isRightDay      = (OptionDates == day) #right day boolean
    rightDayOptions = OptionDataArr[isRightDay, :] #grab options for trading day
    
    expirations    = rightDayOptions[:, 1] 
    frontExpiry    = np.unique(expirations)[0] #grab front month expiration
    backExpiry     = np.unique(expirations)[1] #grab back month expiration
    daysToMatFront = frontExpiry - day #days to maturity of front contract
    
    #Set expiration for options strategy will trade
    if daysToMatFront > 7:
        useFrontMonth = True
        exdate = frontExpiry 
    else:
        useFrontMonth = False
        exdate = backExpiry
    
    isRightExp = (expirations == exdate)
    isATM      = (rightDayOptions[:, -1] == 1)

    optionsToGet = rightDayOptions[isRightExp * isATM, :]    
    
    if np.size(optionsToGet, 0) < 2 and i > 0:
        #raise ValueError("Not enough ATM options to create straddle")
        error.append(i)
        FrontATMCalls = np.concatenate((FrontATMCalls, CallsToGet), axis = 0)
        FrontATMPuts  = np.concatenate((FrontATMPuts, PutsToGet), axis = 0)   
        continue
        
    if np.size(optionsToGet, 0) > 2: #grab the options closest to ATM
       strikes  = optionsToGet[:, 3] #grab strikes
       spot     = optionsToGet[:, 19][0] #grab spot
       toGetIdx = ((strikes - spot) == np.min(strikes - spot)) #index of options closest to spot
       toGet    = optionsToGet[toGetIdx, :]
       CallsToGet = toGet[(toGet[:,2] == 1), :]
       PutsToGet  = toGet[(toGet[:,2] == 0), :]
    
    FrontATMCalls = np.concatenate((FrontATMCalls, CallsToGet), axis = 0)
    FrontATMPuts  = np.concatenate((FrontATMPuts, PutsToGet), axis = 0)   
    
   
   #callstrikes = optionsToGet[:, 3][optionsToGet[:, 2] == 1]
   #putstrikes  = optionsToGet[:, 3][optionsToGet[:, 2] == 0]
   #callToGet = ((callstrikes - spot) ==  np.min(callstrikes - spot))
   #puyToGet = ((putstrikes - spot) ==  np.min(putstrikes - spot))

#What do with more options

#cp_flag = rightDayOptions[:, 2]
#rightDayCalls = rightDayOptions[(cp_flag == 1), :] #grab calls
#rightDayPuts  = rightDayOptions[(cp_flag == 0), :] #grab puts

   
     












































