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
UnderlyingBBGTicker   = "SPX Index"
UnderlyingTicker      = "SPX"
loadlocAgg            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
loadlocOpt            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
loadlocSpot           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/SpotData/SpotData.xlsx"
prefColor             = '#0504aa'
##########################################################################################

## Load Data
#Aggregate Data
AggregateData   = pd.read_csv(loadlocAgg + UnderlyingTicker + "AggregateData.csv")
AggregateDates  = AggregateData["Dates"].to_numpy()
startDate = AggregateDates[0]
endDate   = AggregateDates[-1]

#Underlying
Underlying      = AggregateData[UnderlyingTicker].to_numpy()
UnderlyingDates = AggregateData["Dates"].to_numpy()
RfDaily         = AggregateData["Rf Daily"].to_numpy()


#Option Data
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
frontIdError = []
backIdError  = []

#Set params for loop
nDays           = np.size(UniqueDates)
rollDay  = 7
isRollover    = np.zeros((nDays, ))
frontATMCalls = np.zeros((nDays, np.size(OptionCols)))
frontATMPuts  = np.zeros((nDays, np.size(OptionCols)))
backATMCalls   = np.zeros((nDays, np.size(OptionCols)))
backATMPuts    = np.zeros((nDays, np.size(OptionCols)))

frontCallReturns = np.zeros((nDays, ))
frontPutReturns  = np.zeros((nDays, ))
backCallReturns  = np.zeros((nDays, ))
backPutReturns   = np.zeros((nDays, ))

def getATMOptions(options):
    strikes    = options[:, 3]      #grab strikes
    spot       = options[:, 19][0]  #grab spot
    cpflag     = options[:, 2]      #grab call/put flag
    
    callstrikes = strikes[cpflag == 1]
    calloptions = options[cpflag == 1, :]
    callToGetIdx   = (np.abs(callstrikes - spot) == np.min(np.abs(callstrikes - spot)))
    
    putstrikes = strikes[cpflag == 0]
    putoptions = options[cpflag == 0, :]
    putToGetIdx   = (np.abs(putstrikes - spot) == np.min(np.abs(putstrikes - spot)))
    
    ATMCall = calloptions[callToGetIdx, :]
    ATMPut  = putoptions[putToGetIdx, :]
    
    #ATMOptions = options[toGetIdx, :] #grab the options closest
    #ATMCalls   = ATMOptions[(cpflag[toGetIdx] == 1), :] #separate calls
    #ATMPuts    = ATMOptions[(cpflag[toGetIdx] == 0), :] #separate puts
    
    return ATMCall, ATMPut

#match option across trading days
def matchOption(option, optionsToMatch, matchClosest = False):
    #grab unique features of option
    exdate = option[0,1] 
    cpflag = option[0,2]
    strike = option[0,3]
    
    #construct booleans
    isRightExp    = (optionsToMatch[:, 1] == exdate)
    isRightType   = (optionsToMatch[:, 2] == cpflag)
    isRightStrike = (optionsToMatch[:, 3] == strike)
    
    if matchClosest == True and np.sum(isRightStrike) == 0:
        matchStrikes  = optionsToMatch[:, 3]
        strikeDiff    = np.abs(matchStrikes - strike)
        closestStrike = matchStrikes[(strikeDiff == np.min(strikeDiff))]
        isRightStrike = (optionsToMatch[:, 3] == closestStrike) 
                                     
        
    #combine to one boolean and extract
    matchIndex  = isRightExp * isRightType * isRightStrike
    matchOption = optionsToMatch[matchIndex, :]
    
    return matchOption


#Grab front and back month ATM Options
for i in np.arange(0, nDays - 1):        
    day = UniqueDates[i] #grab day
    nextDay = UniqueDates[i + 1]
    isRightDay      = (OptionDates == day) #right day boolean
    rightDayOptions = OptionDataArr[isRightDay, :] #grab options for trading day
    
    #Grab next day options
    isNextDay      = (OptionDates == nextDay)
    nextDayOptions = OptionDataArr[isNextDay, :]
    
    
    expirations    = rightDayOptions[:, 1] 
    frontExpiry    = np.unique(expirations)[0] #grab front month expiration
    backExpiry     = np.unique(expirations)[1] #grab back month expiration
    daysToMatFront = frontExpiry - day #days to maturity of front contract
 
    
    if daysToMatFront < rollDay:
        isRollover[i] = 1
        
    
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
          
    
    #Compute Front Month and Back Month ATM Options Returns
    #For Calls and Puts
    #Assume daily Rebalancing
    
    if np.size(FrontToGet, 0) >= 2: #grab front
        frontCallToGet, frontPutToGet = getATMOptions(FrontToGet)
        nextDayCall = matchOption(frontCallToGet, nextDayOptions, matchClosest = True)
        nextDayPut  = matchOption(frontPutToGet, nextDayOptions, matchClosest = True)
        
        if len(nextDayCall) > 0:
            frontCallReturns[i + 1] = nextDayCall[0, 15] / frontCallToGet[0, 15] - 1 #mid  
        else:
            frontIdError.append(i)              
        if len(nextDayPut) > 0:
            frontPutReturns[i + 1]  = nextDayPut[0, 15] /  frontPutToGet[0, 15] - 1  #mid
        else:
           frontIdError.append(i) 
        
        frontATMCalls[i, :] = frontCallToGet[0, :] #store front ATM call (take the first if two are ATM)
        frontATMPuts[i, :]  = frontPutToGet[0, :]  #store front ATM put
    
    else: #not enough ATM options, use previous day options and  save error for investigation
        frontError.append(i)
        
        #set return to zero if not enough front contracts         
        frontCallReturns[i + 1] = 0
        frontPutReturns[i + 1]  = 0
                
        frontATMCalls[i, :] = frontATMCalls[i-1, :] #fill with previous
        frontATMPuts[i, :]  = frontATMPuts[i-1, :] #fill with previous
        
    #Grab Back ATM Options
    if np.size(BackToGet, 0) >= 2: #grab back
        backCallToGet, backPutToGet = getATMOptions(BackToGet)
        
        nextDayCall = matchOption(backCallToGet, nextDayOptions, matchClosest = True)
        nextDayPut  = matchOption(backPutToGet,  nextDayOptions, matchClosest = True)
        
        if len(nextDayCall) > 0:
            backCallReturns[i + 1] = nextDayCall[0, 15] / backCallToGet[0, 15] - 1 #mid  
        else:
            backIdError.append(i) 
        
        if len(nextDayPut) > 0:                
            backPutReturns[i + 1]  = nextDayPut[0, 15] / backPutToGet[0, 15] - 1   
        else:
            backIdError.append(i) 
        
        backATMCalls[i, :] = backCallToGet[0, :]
        backATMPuts[i, :]  = backPutToGet[0, :]
    
    else:
        backError.append(i)
        
        backCallReturns[i + 1] = 0                
        backPutReturns[i + 1]  = 0
        
        backATMCalls[i, :]  = backATMCalls[i-1, :] #fill with previous
        backATMCalls[i, :]  = backATMCalls[i-1, :] #fill with previous 



## Error investigation


# errorDate = UniqueDates[idError[0]-1]
# isErrorDate = (OptionDates == errorDate)
# errorDayOptions = OptionDataArr[isErrorDate, :]
# errorDayOptions = errorDayOptions[:, [0, 1, 2, 3, 19, -1]]


#Roll Returns
scale = 1
rolledCallReturns = ((1 - isRollover)* frontCallReturns + isRollover*backCallReturns)*scale
rolledPutReturns  = ((1 - isRollover)* frontPutReturns + isRollover*backPutReturns)*scale
straddleReturns   = 0.5*(rolledCallReturns + rolledPutReturns)

optionReturns = np.concatenate((UniqueDates.reshape(-1, 1), rolledCallReturns.reshape(-1, 1), rolledPutReturns.reshape(-1, 1), straddleReturns.reshape(-1, 1)), axis = 1)

#Returns of underlying 
#### TO BE REPLACED BY TOTAL RETURN INDEX RETURNS
underlyingTotalReturns   = Underlying[1:] / Underlying[0:-1] - 1
underlyingTotalReturns   = np.concatenate((np.zeros((1,1)), underlyingTotalReturns.reshape(-1, 1)), axis = 0)
RfDaily[0] = 0 #set rf the first day to zero
underlyingXsReturns = underlyingTotalReturns[:, 0] - RfDaily #excess returns

underlyingReturns = np.concatenate((UnderlyingDates.reshape(-1, 1), underlyingTotalReturns.reshape(-1, 1), underlyingXsReturns.reshape(-1,1)), axis = 1)


def SyncData(array1, array2, fillPrevious = False):
    #Construct empty SyncMat
    nRows = max(len(array1), len(array2))
    nCols_1 = np.size(array1, 1)
    nCols_2 = np.size(array2, 1)
    nCols   = nCols_1 + nCols_2 - 1
    syncMat = np.zeros((nRows, nCols)) #matrix where synced data wil be stored

    #Assume dates are in first column
    date_bool  = np.in1d(array1[:, 0], array2[:, 0]) #Dates where option data is recorded

    #Check if dates are mismatched both ways
    date_bool2  = np.in1d(array2[:, 0],array1[:, 0]) #Dates where option data is recorded
    if np.sum(date_bool2 == 0) != 0:
        raise ValueError("Dates are mismatched both ways")


    syncMat[:, 0:nCols_1] = array1
    syncMat[date_bool, nCols_1:] = array2[:, 1:]

    if fillPrevious == True:
        date_shift = np.concatenate((date_bool[1:], np.ones((1,))), axis = 0) == 1 
        syncMat[(date_bool == 0), nCols_1:] = syncMat[(date_shift == 0), nCols_1:]  

    return syncMat

ReturnMat = SyncData(underlyingReturns, optionReturns, fillPrevious = False)

#Gamma Strategy
signal = (netGamma < 0) #long signal
lag    = 1
GammaStraddleXsReturns = signal[0:-lag] * ReturnMat[lag:, -1]
GammaStraddleXsReturns = np.concatenate((np.zeros(1,), GammaStraddleXsReturns), axis = 0)

#Hedge Overlay
hedgeWeight = 0.05
GammaStraddleOverlayXsReturns = hedgeWeight*GammaStraddleXsReturns + (1 - hedgeWeight)*ReturnMat[:, 2]


#Cumulative Returns
cumCallReturns             = np.cumprod(1 + ReturnMat[:, 3])
cumPutReturns              = np.cumprod(1 + ReturnMat[:, 4])
cumStraddleReturns         = np.cumprod(1 + ReturnMat[:, 5])
cumUnderlyingXsReturns     = np.cumprod(1 + ReturnMat[:, 2])
cumUnderlyingTotalReturns  = np.cumprod(1 + ReturnMat[:, 1])
cumGammaStraddleXsReturns  = np.cumprod(1 + GammaStraddleXsReturns)
cumGammaOverlayXsReturns   = np.cumprod(1 + GammaStraddleOverlayXsReturns)



#Plot Returns
dates4fig = pd.to_datetime(ReturnMat[:, 0], format = '%Y%m%d')

plt.figure()
plt.plot(dates4fig, cumCallReturns, color = prefColor, label = "ATM Calls")
plt.plot(dates4fig, cumUnderlyingXsReturns, color = "red", alpha = 0.7, label = UnderlyingBBGTicker)
plt.title("Front ATM Call Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")

plt.figure()
plt.plot(dates4fig, cumPutReturns, color = prefColor, label = "ATM Puts")
plt.plot(dates4fig, cumUnderlyingXsReturns, color = "red", alpha = 0.7, label = UnderlyingBBGTicker)
plt.title("Front ATM Put Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")


plt.figure()
plt.plot(dates4fig, cumStraddleReturns, color = prefColor, label = "ATM Straddles")
plt.plot(dates4fig, cumUnderlyingXsReturns, color = "red", alpha = 0.7, label = UnderlyingBBGTicker)
plt.title("Front ATM Straddle Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")

plt.figure()
plt.plot(dates4fig, cumGammaStraddleXsReturns, color = prefColor, label = "Gamma Timed Straddles")
plt.plot(dates4fig, cumUnderlyingXsReturns, color = "red", alpha = 0.7, label = UnderlyingBBGTicker)
plt.title("Front ATM Straddle Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")

plt.figure()
plt.plot(dates4fig, cumGammaStraddleXsReturns, color = prefColor, label = "Gamma Timed ATM Straddles")
plt.plot(dates4fig, cumStraddleReturns, color = "red", alpha = 0.7, label = "ATM Straddles")
plt.title("Front ATM Straddle Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")

plt.figure()
plt.plot(dates4fig, cumGammaOverlayXsReturns, color = prefColor, label = "Gamma Straddle Overlay")
plt.plot(dates4fig, cumUnderlyingXsReturns, color = "red", alpha = 0.7, label = UnderlyingBBGTicker)
plt.title("Front ATM Straddle Returns - daily rebalancing")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
#plt.yscale("log")



tt= bt.ComputePerformance(GammaStraddleXsReturns.reshape(-1, 1), RfDaily, 0, 255)









