# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:08:07 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Backtest as bt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import sys

#VIX Strategy
### SET IMPORT PARAMS ####################################################################
UnderlyingBBGTicker   = "SPX Index"
UnderlyingTicker      = "SPX"
loadlocAgg            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
loadlocSpot           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/SpotData/SpotData.xlsx"
loadlocFutures        = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/FuturesData/"
prefColor             = '#0504aa'
##########################################################################################

#Load data
#VIX Futures
#VIXFutures        = pd.read_excel(loadlocFutures + "VIXFuturesData.xlsx", sheet_name = "Prices")
VIXFuturesPrices  = pd.read_excel(loadlocFutures + "VIXFuturesDataUnrolled.xlsx", sheet_name = "Prices")
VIXFuturesTickers = pd.read_excel(loadlocFutures + "VIXFuturesDataUnrolled.xlsx", sheet_name = "Tickers")
  
#Aggregate Data
AggregateData   = pd.read_csv(loadlocAgg + UnderlyingTicker + "AggregateData.csv")
AggregateDates  = AggregateData["Dates"].to_numpy()
netGamma        = AggregateData["netGamma"].to_numpy()

#Compute Daily Rf
def computeRfDaily(data, withDates = False):
        dates            = data["Dates"].to_numpy()
        dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
        daycount         = bt.dayCount(dates4fig)
        
        Rf               = data["LIBOR"].to_numpy() / 100
        RfDaily          = np.zeros((np.size(Rf, 0), ))
        RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360
        
        if withDates == True:
            RfDaily = np.concatenate((dates.reshape(-1,1), RfDaily.reshape(-1,1)), axis = 1)
        
        return RfDaily
RfDaily = computeRfDaily(AggregateData, withDates = False)

#Store important variables in array to sync w/ VIX data
syncArr = np.concatenate((AggregateDates.reshape(-1,1), netGamma.reshape(-1,1), RfDaily.reshape(-1,1)), axis = 1)

#VIX Dates
VIXDates   = VIXFuturesPrices["Dates"]
datesTime  = pd.to_datetime(VIXDates, format = '%Y-%m-%d')
daycount   = bt.dayCount(datesTime) #count days between each obs.
VIXDates   = bt.yyyymmdd(datesTime) #get to right format

#Roll futures returns
futPrices     = VIXFuturesPrices.iloc[:, 1:-1].to_numpy()
futTickers    = VIXFuturesTickers.iloc[:, 1:].to_numpy()
nContracts    = np.size(futPrices, 1) - 1
RolledReturns = np.zeros((len(futPrices), nContracts))
RolledReturns[1:, :], isRollover = bt.RolloverFutures(futPrices[:, 0:-1], futPrices[:, 1:], futTickers, returnRoll = True)
VIXFuturesReturns = np.concatenate((VIXDates.reshape(-1,1), RolledReturns), axis = 1)  


#VIX Futures
#VIXFutures["Dates"] = VIXDates
#VIXFutures = VIXFutures.to_numpy()

#Trim, sync and align data
startDate = VIXDates[0] 
endDate   = VIXDates[-1] + 1

AggregateDataTr = bt.trimToDates(syncArr, syncArr[:, 0], startDate, endDate)
#SyncMat        = bt.SyncData(VIXFutures[1:, :], AggregateDataTr, fillPrevious = True, removeNonOverlapping = False)
SyncMat         = bt.SyncData(VIXFuturesReturns, AggregateDataTr, fillPrevious = True, removeNonOverlapping = False)


#Compute Returns
#VIXReturns = np.zeros((len(SyncMat), 5))
#VIXReturns[1:, :] = SyncMat[1:, 1:6] / SyncMat[0:-1, 1:6] - 1
RfDaily    = SyncMat[:, -1]
VIXReturns = SyncMat[:, 1:nContracts]


#Timing strategy
lag = 1
scale = 0.2

#Long strategies
gammaSignal     = (SyncMat[:, -2] < 0) #long when gamma is negative
timedFrontReturns   = gammaSignal[0:-lag] * VIXReturns[lag:, 0] * scale
timedBackReturns    = gammaSignal[0:-lag] * VIXReturns[lag:, 1] * scale
untimedFrontReturns = VIXReturns[lag:, 0] * scale
untimedBackReturns  = VIXReturns[lag:, 1] * scale

#Short Strategies
shortSignal = (SyncMat[:, -2] > 0)
timedShortFrontReturns = (-1)*shortSignal[0:-lag]*VIXReturns[lag:, 0]*scale #gamma timed front
timedShortBackReturns  = (-1)*shortSignal[0:-lag]*VIXReturns[lag:, 1]*scale #gamma timed back

#Combo Strategies
timedLSFrontReturns = timedFrontReturns + timedShortFrontReturns#long/short gamma-timed front
timedLSBackReturns  = timedBackReturns + timedShortBackReturns #long/short gamma-timed back
timedLSTermReturns  = timedBackReturns +    timedShortFrontReturns   #long back, short front gamma-timed


#Cumulative Returns
#Long
cumTimedFront   = np.cumprod(1 + timedFrontReturns)
cumTimedBack    = np.cumprod(1 + timedBackReturns)
cumUntimedFront = np.cumprod(1 + untimedFrontReturns)
cumUntimedBack  = np.cumprod(1 + untimedBackReturns)

#Short
cumTimedShortFront   = np.cumprod(1 + timedShortFrontReturns)
cumTimedShortBack    = np.cumprod(1 + timedShortBackReturns)
cumUntimedShortFront =  np.cumprod(1 - untimedFrontReturns)
cumUntimedShortBack  = np.cumprod(1 - untimedBackReturns)

#Long/Short Combination
cumTimedLSFront = np.cumprod(1 + timedLSFrontReturns)
cumTimedLSBack  = np.cumprod(1 + timedLSBackReturns)
cumTimedLSTerm  = np.cumprod(1 + timedLSTermReturns)



#Plots
dates4fig = pd.to_datetime(SyncMat[lag:, 0], format = "%Y%m%d")

#Long strategies
plt.figure()
plt.plot(dates4fig, cumTimedFront, "k", label = "Gamma-timed Front")
plt.plot(dates4fig, cumTimedBack, "b", label = "Gamma-timed Back")
plt.plot(dates4fig, cumUntimedFront, "--k", label = "Untimed Front")
plt.plot(dates4fig, cumUntimedBack, "--b", label = "Untimed Back")
plt.title("Cumulative Returns")
plt.ylabel("Cumulative Excess Returns")
plt.legend()

#Short strategies
plt.figure()
plt.plot(dates4fig, cumTimedShortFront, "k", label = "Gamma-timed Front")
plt.plot(dates4fig, cumTimedShortBack, "b", label = "Gamma-timed Back")
#plt.plot(dates4fig, cumUntimedShortFront, "--k", label = "Untimed Front")
#plt.plot(dates4fig, cumUntimedShortBack, "--b", label = "Untimed Back")
plt.title("Cumulative Returns Short Strategies")
plt.ylabel("Cumulative Excess Returns")
plt.legend()

#Long/Short combinations
plt.figure()
plt.plot(dates4fig, cumTimedLSFront, "k", label = "Gamma-timed L/S Front")
plt.plot(dates4fig, cumTimedLSBack, "b", label = "Gamma-timed L/S Back")
plt.plot(dates4fig, cumTimedLSTerm, "r", label = "Gamma-timed L/S Term")
plt.title("Cumulative Returns Short Strategies")
plt.ylabel("Cumulative Excess Returns")
plt.legend()
















