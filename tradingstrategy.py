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

currentlyInvested = 0 #initialize no position

frontError   = []
backError    = []

#Set params for loop
nDays           = np.size(UniqueDates)
rollDay  = 7
rollover = np.zeros((nDays, 1))
frontATMCalls = np.zeros((nDays, np.size(OptionCols)))
frontATMPuts  = np.zeros((nDays, np.size(OptionCols)))
backATMCalls   = np.zeros((nDays, np.size(OptionCols)))
backATMPuts    = np.zeros((nDays, np.size(OptionCols)))

#Grab front and back month ATM Options
for i in np.arange(0, nDays):
            
    day = UniqueDates[i] #grab day
    isRightDay      = (OptionDates == day) #right day boolean
    rightDayOptions = OptionDataArr[isRightDay, :] #grab options for trading day
    
    expirations    = rightDayOptions[:, 1] 
    frontExpiry    = np.unique(expirations)[0] #grab front month expiration
    backExpiry     = np.unique(expirations)[1] #grab back month expiration
    daysToMatFront = frontExpiry - day #days to maturity of front contract
 
    
    if daysToMatFront < rollDay:
        rollover[i] = 1
        
    
    isFrontExp = (expirations == frontExpiry)
    isBackExp  = (expirations == backExpiry)
    isATM      = (rightDayOptions[:, -1] == 1)
    
    FrontToGet = rightDayOptions[isFrontExp * isATM, :]    
    BackToGet  = rightDayOptions[isBackExp * isATM, :]
    
    #Check number of calls and puts
    nCallsFront = np.sum(FrontToGet[:, 2] == 1)
    nPutsFront  = np.sum(FrontToGet[:, 2] == 0)
    nCallsBack  = np.sum(BackToGet[:, 2] == 1)
    nPutsBack   = np.sum(BackToGet[:, 2] == 0)
    
    
    def getATMOptions(options):
        strikes    = options[:, 3]      #grab strikes
        spot       = options[:, 19][0]  #grab spot
        cpflag     = options[:, 2]      #grab call/put flag
        
        callstrikes = strikes[cpflag == 1]
        calloptions = options[cpflag == 1, :]
        callsToGetIdx   = ((callstrikes - spot) == np.min(callstrikes - spot))
        
        putstrikes = strikes[cpflag == 0]
        putoptions = options[cpflag == 0, :]
        putsToGetIdx   = ((putstrikes - spot) == np.min(putstrikes - spot))
        
        ATMCalls = calloptions[callsToGetIdx, :]
        ATMPuts  = putoptions[putsToGetIdx, :]
        
        #ATMOptions = options[toGetIdx, :] #grab the options closest
        #ATMCalls   = ATMOptions[(cpflag[toGetIdx] == 1), :] #separate calls
        #ATMPuts    = ATMOptions[(cpflag[toGetIdx] == 0), :] #separate puts
        
        return ATMCalls, ATMPuts
    
    
    
    ### NEED TO GET PREVIOUS DAY PRICE TO COMPUTE RETURNS
    
    #Grab Front ATM Options
    if np.size(FrontToGet, 0) >= 2: #grab front
        frontCallsToGet, frontPutsToGet = getATMOptions(FrontToGet)
        frontATMCalls[i, :] = frontCallsToGet
        frontATMPuts[i, :]  = frontPutsToGet
    
    else: #not enough ATM options, use previous day options and  save error for investigation
        frontError.append(i)
        frontATMCalls[i, :] = frontATMCalls[i-1, :] #fill with previous
        frontATMPuts[i, :]  = frontATMPuts[i-1, :] #fill with previous
        
    #Grab Back ATM Options
    if np.size(BackToGet, 0) >= 2: #grab back
        backCallsToGet, backPutsToGet = getATMOptions(BackToGet)
        backATMCalls[i, :] = backCallsToGet
        backATMPuts[i, :]  = backPutsToGet
    
    else:
        backError.append(i)
        backATMCalls[i, :]  = backATMCalls[i-1, :] #fill with previous
        backATMCalls[i, :]  = backATMCalls[i-1, :] #fill with previous 



## Error investigation

errorDate = UniqueDates[frontError[26]]
isErrorDate = (OptionDates == errorDate)
errorDayOptions = OptionDataArr[isErrorDate, :]
errorDayOptions = errorDayOptions[:, [0, 1, 2, 3, 19, -1]]


back = errorDayOptions[:, 1] == errorDate
atm  = errorDayOptions[:, -1] == 1

tt  = errorDayOptions[back*atm, :]































