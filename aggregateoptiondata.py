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

## Aggregate Option Data to daily series

### SET WHICH ASSET TO BE IMPORTED #######################################################
UnderlyingAssetName   = "IWM US Equity"
UnderlyingTicker      = "IWM"
VolIndexTicker        = "RVX Index"
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
equity_index          = False
##########################################################################################

#Load data
OptionData        = pd.read_csv(loadloc + UnderlyingTicker + "OptionDataClean.csv")
UnderlyingData    = pd.read_csv(loadloc + UnderlyingTicker + "UnderlyingData.csv")


#Risk free rate
Rf           = pd.read_excel("C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/SpotData/SpotData.xlsx", sheet_name = "Rf")
RfDates      = Rf["Dates"].to_numpy()
RfDates      = pd.to_datetime(RfDates, format = "%Y-%m-%d")
RfDates      = bt.yyyymmdd(RfDates)

#Trim Data
startDate = 19960102
endDate   = 20200101

OptionData  = bt.trimToDates(OptionData, OptionData["date"], startDate, endDate)    
print(OptionData.head())
print(OptionData.tail())

UnderlyingDataTr = bt.trimToDates(UnderlyingData, UnderlyingData["Dates"], startDate, endDate)
print(UnderlyingDataTr.head())
print(UnderlyingDataTr.tail())


#Trim Rf
startDate = int(UnderlyingDataTr.iloc[0, 0])
endDate   = int(UnderlyingDataTr.iloc[-1, 0]) + 1
RfTr      = bt.trimToDates(Rf, RfDates, startDate, endDate)
print(RfTr.head())
print(RfTr.tail())

#Compute daily rf
datesTime   = pd.to_datetime(RfTr["Dates"].to_numpy(), format = '%Y-%m-%d')
daycount    = bt.dayCount(datesTime)
Rf          = RfTr["US0003M Index"].to_numpy() / 100 #Adjust to pct.points
RfDaily     = np.zeros((np.size(RfTr, 0), ))
RfDaily[1:] = Rf[0:-1] * daycount[1:]/360 

#VIX Futures
if UnderlyingTicker == "VIX":
    futPrices         = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesData.xlsx', sheet_name = "Prices")
    futVolume         = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesData.xlsx', sheet_name = "Volume")
    futOpenInterest   = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesData.xlsx', sheet_name = "Open Interest")
    futPricesUnrolled = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesDataUnrolled.xlsx', sheet_name = "Prices")
    
    futDates   = futPrices["Dates"]
    futDates   = pd.to_datetime(futDates, '%Y-%m-%d')
    futDates   = bt.yyyymmdd(futDates)
    
    futDatesUnrolled   = futPricesUnrolled["Dates"]
    futDatesUnrolled   = pd.to_datetime(futDatesUnrolled, '%Y-%m-%d')
    futDatesUnrolled   = bt.yyyymmdd(futDatesUnrolled)
    
    
    futPrices         = bt.trimToDates(futPrices, futDates, startDate, endDate)
    futVolume         = bt.trimToDates(futVolume, futDates, startDate, endDate)
    futOpenInterest   = bt.trimToDates(futOpenInterest.iloc[:, 0:5], futDates, startDate, endDate)
    futPricesUnrolled = bt.trimToDates(futPricesUnrolled.iloc[:, 0:5], futDatesUnrolled, startDate, endDate)    
    
    
    frontPrices  = futPrices.iloc[:, 1].to_numpy()
    backPrices   = futPrices.iloc[:, 2].to_numpy()
    frontVolume  = futVolume.iloc[:, 1].to_numpy()
    backVolume   = futVolume.iloc[:, 2].to_numpy()

    #Market Cap "Estimation" for VIX
    VIXMarketCap = np.sum((futOpenInterest.iloc[:, 1:].to_numpy()) * (futPricesUnrolled.iloc[:, 1:].to_numpy()), 1)
    

################################
## Compute and aggregate data ##

#Delete options that violate arbitrage bounds
gamma         = OptionData["gamma"].to_numpy()
non_violating = np.isfinite(gamma)
OptionDataTr  = OptionData.loc[non_violating, :]

#Underlying data
UnderlyingDates     = UnderlyingDataTr["Dates"].to_numpy()
UnderlyingPrices    = UnderlyingDataTr["Price"].to_numpy()
UnderlyingVolume    = UnderlyingDataTr["Volume"].to_numpy()

if UnderlyingTicker != "VIX":
    UnderlyingVolIndex  = UnderlyingDataTr[VolIndexTicker].to_numpy()

if UnderlyingTicker == "VIX":
    UnderlyingMarketCap = VIXMarketCap
else:    
    UnderlyingMarketCap = UnderlyingDataTr["Market Cap"].to_numpy()

if equity_index == True:
    UnderlyingTR    = UnderlyingDataTr["TR Index"].to_numpy()

if UnderlyingTicker != "VIX": #Backfill market cap unless underlying is VIX
    #Backfill Market Cap
    UnderlyingReturns = bt.computeReturns(UnderlyingPrices) #Compute underlying returns
    refInd     = int(np.nonzero(np.isfinite(UnderlyingMarketCap))[0][0]) #Find first finite value index
    refVal     = UnderlyingMarketCap[refInd]     #Grab first finite value
    refReturns = UnderlyingReturns[0:refInd + 1] #Grab returns up to and including first finite value
    refCumRet  = np.cumprod(1 + refReturns)      #Compute cumulative returns up to first finite value
    startVal   = refVal / refCumRet[-1]          #Compute starting point
    fillSeries = startVal * refCumRet[0:-1]      #Create price series
    UnderlyingMarketCap[0:refInd] = fillSeries   #Add filled market cap values


#MA dollar volume
lookback = 90
nDays          = np.size(UnderlyingVolume)
MADollarVolume = np.zeros((nDays,))
MAVolume       = np.zeros((nDays,))
ILLIQ          = np.zeros((nDays,))

if equity_index == True:
    UnderlyingReturns = bt.computeReturns(UnderlyingTR)
else:
    UnderlyingReturns = bt.computeReturns(UnderlyingPrices)

for i in np.arange(lookback, nDays):
    #grab volume
    if UnderlyingTicker == "VIX":
        volume = frontVolume[i - lookback:i]
        #Should have futures price here, estimate with spot price
    else:
        volume = UnderlyingVolume[i - lookback:i] 
    
    price         = UnderlyingPrices[i - lookback:i] #grab price
    returns       = UnderlyingReturns[i - lookback:i] #grab returns
    abs_returns   = np.abs(returns)
    dollar_volume = volume*price #compute dollar volume
    daily_illiq   = abs_returns / dollar_volume
    
    ILLIQ[i]          = np.mean(daily_illiq)
    MAVolume[i]       = np.mean(volume)
    MADollarVolume[i] = np.mean(dollar_volume)

    
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
impliedVol   = OptionDataTr["impl_volatility"]
strikes      = OptionDataTr["strike_price"]
prices       = OptionDataTr["mid_price"]
OTMFlag      = OptionDataTr["OTM_flag"]

#Compute timeseries of daily aggregate statistics
netGamma                = np.zeros((nOptionDays, 1))
netGamma_alt            = np.zeros((nOptionDays, 1))
callGamma               = np.zeros((nOptionDays, 1))
putGamma                = np.zeros((nOptionDays, 1))
aggOpenInterest         = np.zeros((nOptionDays, 1))
netOpenInterest         = np.zeros((nOptionDays, 1))
deltaAdjOpenInterest    = np.zeros((nOptionDays, 1))
deltaAdjNetOpenInterest = np.zeros((nOptionDays, 1))
aggVolume               = np.zeros((nOptionDays, 1))
deltaAdjVolume          = np.zeros((nOptionDays, 1))
IVOL                    = np.zeros((nOptionDays, 1))
weightedIVOL            = np.zeros((nOptionDays, 1))



for i in np.arange(0, nOptionDays):
    day = UniqueDates[i] #grab day
    isRightDay = (OptionDates == day) #boolean for right day
    
    #Grab values for relevant day
    rel_callFlag     = callFlag[isRightDay]
    rel_gamma        = gamma[isRightDay]
    rel_delta        = delta[isRightDay]
    rel_openInterest = openInterest[isRightDay]
    rel_volume       = volume[isRightDay]
    rel_impliedVol   = impliedVol[isRightDay]
    rel_prices       = prices[isRightDay]
    rel_strikes      = strikes[isRightDay]
    rel_OTMFlag      = OTMFlag[isRightDay]
    
    #Gamma Exposure
    rel_gamma_call  = rel_gamma[rel_callFlag == 1] * rel_openInterest[rel_callFlag == 1]*100
    rel_gamma_put   = rel_gamma[rel_callFlag == 0] * rel_openInterest[rel_callFlag == 0]*100
    netGamma[i]     = np.sum(rel_gamma_call) - np.sum(rel_gamma_put)
    netGamma_alt[i] = np.sum(rel_gamma_call) + np.sum(rel_gamma_put)
    callGamma[i]    = np.sum(rel_gamma_call)
    putGamma[i]     = np.sum(rel_gamma_put)
    
    #Agg and net open interest
    aggOpenInterest[i]  = np.sum(rel_openInterest)*100
    netOpenInterest[i]  = np.sum(rel_openInterest[rel_callFlag == 1]) - np.sum(rel_openInterest[rel_callFlag == 0])
    
    #delta adjusted open interest
    deltaAdjOpenInterest[i]    = np.sum(np.abs(rel_delta) * 100 * rel_openInterest)
    deltaAdjNetOpenInterest[i] = np.sum(rel_delta * 100 * rel_openInterest)
    
    #volume
    aggVolume[i]      = np.sum(rel_volume)
    deltaAdjVolume[i] = np.sum(np.abs(rel_delta)*100 * rel_volume)
    
    #Implied Volatility
    IVOL[i] = np.mean(rel_impliedVol) #Naive average implied vol
    
    #Variance Swap Approach
    #Calls
    #otmCallImpliedVol  = rel_impliedVol[(rel_callFlag == 1) * (rel_OTMFlag == 1)]
    #otmCallStrikes     = rel_strikes[(rel_callFlag == 1) * (rel_OTMFlag == 1)]
    #call_leg = (1 / otmCallStrikes**2) * otmCallImpliedVol 
    
    #Puts
    #otmPutImpliedVol = rel_impliedVol[(rel_callFlag == 0)*(rel_OTMFlag == 1)]
    #otmPutStrikes    = rel_strikes[(rel_callFlag == 0) * (rel_OTMFlag == 1)]
    #put_leg = (1 / otmPutStrikes**2) * otmPutImpliedVol

    #weightedIVOL[i] = np.sum(call_leg) + np.sum(put_leg)
    
    
    
#Construct array with aggregate data
aggregateData = np.concatenate((UniqueDates.reshape(nOptionDays, 1), netGamma, netGamma_alt, callGamma, putGamma, aggOpenInterest,\
                netOpenInterest, deltaAdjOpenInterest, deltaAdjNetOpenInterest, aggVolume, deltaAdjVolume, IVOL), axis = 1)    

    
#Save unsynced (to underlying) data for strategy use
#transfrom to datafram
#cols              = np.array(["Dates", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
#                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume"])
#aggregateUnsynced =  pd.DataFrame.from_records(aggregateData, columns = cols)
#saveloc           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateDataTr/"
#aggregateUnsynced.to_csv(path_or_buf = saveloc + UnderlyingTicker + "AggregateDataTr.csv" , index = False)   


#Fill missing values of aggregate data w/ previous day value
UnderlyingDollarVolume  = UnderlyingPrices * UnderlyingVolume   

if equity_index == True:
    UnderlyingData          = np.concatenate((UnderlyingDates.reshape(-1, 1), UnderlyingPrices.reshape(-1, 1),\
                                UnderlyingVolume.reshape(-1, 1), UnderlyingDollarVolume.reshape(-1, 1),\
                                (Rf*100).reshape(-1, 1), MAVolume.reshape(-1,1), MADollarVolume.reshape(-1,1), \
                                UnderlyingMarketCap.reshape(-1, 1), ILLIQ.reshape(-1,1), UnderlyingTR.reshape(-1,1), UnderlyingVolIndex.reshape(-1,1)), axis = 1) #add price and volume 
    cols = np.array(["Dates", UnderlyingTicker, UnderlyingTicker + " Volume", UnderlyingTicker + " Dollar Volume",\
                 "LIBOR", "MAVolume", "MADollarVolume", "Market Cap", "ILLIQ", "TR Index", VolIndexTicker, "netGamma", "netGamma_alt", "gamma_call", "gamma_put", "aggOpenInterest", "netOpenInterest",\
                     "deltaAdjOpenInterest", "deltaAdjNetOpenInterest", "aggVolume", "deltaAdjVolume", "IVOL"])

else:
        UnderlyingData          = np.concatenate((UnderlyingDates.reshape(-1, 1), UnderlyingPrices.reshape(-1, 1),\
                                UnderlyingVolume.reshape(-1, 1), UnderlyingDollarVolume.reshape(-1, 1),\
                                (Rf*100).reshape(-1, 1), MAVolume.reshape(-1,1), MADollarVolume.reshape(-1,1), UnderlyingMarketCap.reshape(-1, 1), \
                                ILLIQ.reshape(-1,1), UnderlyingVolIndex.reshape(-1,1)), axis = 1)    
            
        cols = np.array(["Dates", UnderlyingTicker, UnderlyingTicker + " Volume", UnderlyingTicker + " Dollar Volume",\
                 "LIBOR", "MAVolume", "MADollarVolume", "Market Cap", "ILLIQ", VolIndexTicker, "netGamma", "netGamma_alt", "gamma_call", "gamma_put", "aggOpenInterest", "netOpenInterest",\
                     "deltaAdjOpenInterest", "deltaAdjNetOpenInterest", "aggVolume", "deltaAdjVolume", "IVOL"])
            
#Set columns for data frame
if UnderlyingTicker == "VIX":
    UnderlyingData = np.concatenate((UnderlyingData, frontPrices.reshape(-1, 1), backPrices.reshape(-1, 1), frontVolume.reshape(-1, 1), backVolume.reshape(-1, 1)), axis = 1)
    
    cols = np.array(["Dates", UnderlyingTicker, UnderlyingTicker + " Volume", UnderlyingTicker + " Dollar Volume",\
                 "LIBOR", "MAVolume", "MADollarVolume", "Market Cap", "ILLIQ", "frontPrices", "backPrices", "frontVolume", "backVolume", "netGamma", "netGamma_alt", "gamma_call", "gamma_put", "aggOpenInterest", "netOpenInterest",\
                     "deltaAdjOpenInterest", "deltaAdjNetOpenInterest", "aggVolume", "deltaAdjVolume", "IVOL"])
         
   
#Sync Data of underlying and aggregate
aggregateDataSynced = bt.SyncData(UnderlyingData, aggregateData, removeNonOverlapping = True)    

#Transform to data frame
aggregateDf  =  pd.DataFrame.from_records(aggregateDataSynced, columns = cols)
aggregateDf  = aggregateDf.iloc[lookback:, :] #Kill lookback period

#plt.figure()
#plt.plot(UnderlyingVolIndex)
#plt.plot(IVOL*100)


## EXPORT DATA TO EXCEL ##
saveloc = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
aggregateDf.to_csv(path_or_buf = saveloc + UnderlyingTicker + "AggregateData.csv" , index = False)


    
    
    
    
