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
UnderlyingAssetName   = "SPX Index"
UnderlyingTicker      = "SPX"
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
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
    futPrices  = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesData.xlsx', sheet_name = "Prices")
    futVolume  = pd.read_excel(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\FuturesData\VIXFuturesData.xlsx', sheet_name = "Volume")
    futDates   = futPrices["Dates"]
    futDates   = pd.to_datetime(futDates, '%Y-%m-%d')
    futDates   = bt.yyyymmdd(futDates)
    
    futPrices    = bt.trimToDates(futPrices, futDates, startDate, endDate)
    futVolume    = bt.trimToDates(futVolume, futDates, startDate, endDate)
    
    frontPrices  = futPrices.iloc[:, 1].to_numpy()
    backPrices   = futPrices.iloc[:, 2].to_numpy()
    frontVolume  = futVolume.iloc[:, 1].to_numpy()
    backVolume   = futVolume.iloc[:, 2].to_numpy()



################################
## Compute and aggregate data ##

#Delete options that violate arbitrage bounds
gamma         = OptionData["gamma"].to_numpy()
non_violating = np.isfinite(gamma)
OptionDataTr  = OptionData.loc[non_violating, :]

#Underlying data
UnderlyingDates  = UnderlyingDataTr["Dates"].to_numpy()
UnderlyingPrices = UnderlyingDataTr["Price"].to_numpy()
UnderlyingVolume = UnderlyingDataTr["Volume"].to_numpy()

#MA dollar volume
lookback = 90
nDays          = np.size(UnderlyingVolume)
MADollarVolume = np.zeros((nDays,))
MAVolume       = np.zeros((nDays,))

for i in np.arange(lookback, nDays):
    #grab volume
    if UnderlyingTicker == "VIX":
        volume = frontVolume[i - lookback:i]
        #Should have futures price here, estimate with spot price
    else:
        volume = UnderlyingVolume[i - lookback:i] 
    
    price  = UnderlyingPrices[i - lookback:i] #grab price
    dollar_volume = volume*price #compute dollar volume
    
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
    aggVolume[i]      = np.sum(rel_volume)
    deltaAdjVolume[i] = np.sum(np.abs(rel_delta)*100 * rel_volume)
    
    
#Construct array with aggregate data
aggregateData = np.concatenate((UniqueDates.reshape(nOptionDays, 1), netGamma, netGamma_alt, aggOpenInterest,\
                netOpenInterest, deltaAdjOpenInterest, deltaAdjNetOpenInterest, aggVolume, deltaAdjVolume), axis = 1)    

#Save unsynced (to underlying) data for strategy use
#transfrom to datafram
#cols              = np.array(["Dates", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
#                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume"])
#aggregateUnsynced =  pd.DataFrame.from_records(aggregateData, columns = cols)
#saveloc           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateDataTr/"
#aggregateUnsynced.to_csv(path_or_buf = saveloc + UnderlyingTicker + "AggregateDataTr.csv" , index = False)   


#Fill missing values of aggregate data w/ previous day value
UnderlyingDollarVolume  = UnderlyingPrices * UnderlyingVolume   
UnderlyingData          = np.concatenate((UnderlyingDates.reshape(-1, 1), UnderlyingPrices.reshape(-1, 1),\
                                UnderlyingVolume.reshape(-1, 1), UnderlyingDollarVolume.reshape(-1, 1),\
                                (Rf*100).reshape(-1, 1), MAVolume.reshape(-1,1), MADollarVolume.reshape(-1,1)), axis = 1) #add price and volume 

if UnderlyingTicker == "VIX":
    UnderlyingData = np.concatenate((UnderlyingData, frontPrices.reshape(-1,1), backPrices(-1,1), frontVolume.reshape(-1,1), backVolume.reshape(-1,1)), axis = 1)
    cols  = np.array(["Dates", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker, UnderlyingTicker + " Volume",\
                            UnderlyingTicker + " Dollar Volume", "LIBOR", "MAVolume", "MADollarVolume", "frontPrices", "backPrices", "frontVolume", "backVolume"])
else:
    cols = np.array(["Dates", UnderlyingTicker, UnderlyingTicker + " Volume", UnderlyingTicker + " Dollar Volume",\
                 "LIBOR", "MAVolume", "MADollarVolume", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest",\
                     "deltaAdjOpenInterest", "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume"])  
    
    # cols = np.array(["Dates", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
    #                     "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker, UnderlyingTicker + " Volume",\
    #                         UnderlyingTicker + " Dollar Volume", "LIBOR", "MAVolume", "MADollarVolume"])

    
aggregateDataSynced = bt.SyncData(UnderlyingData, aggregateData, removeNoneOverlapping = True)    
    

# nRows = np.size(UnderlyingDates)
# nCols = np.size(aggregateData, 1)

# syncMat    = np.zeros((nRows, nCols))
# date_bool  = np.in1d(UnderlyingDates, UniqueDates) #Dates where option data is recorded
# date_shift = np.concatenate((date_bool[1:], np.ones((1,))), axis = 0) == 1 

# syncMat[date_bool, :] = aggregateData
# syncMat[(date_bool == 0), 1:] = syncMat[(date_shift == 0), 1:]    
# syncMat[(date_bool == 0), 0]  = UnderlyingDates[(date_bool == 0)]

#aggregateData = np.concatenate((syncMat, UnderlyingPrices.reshape(nRows, 1),\
#                                UnderlyingVolume.reshape(nRows, 1), UnderlyingDollarVolume.reshape(nRows, 1)), axis = 1) #add price and volume      
    
#transfrom to datafram
#cols        = np.array(["Dates", "netGamma", "netGamma_alt", "aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
#                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker, UnderlyingTicker + " Volume",\
#                            UnderlyingTicker + " Dollar Volume"])



    
aggregateDf =  pd.DataFrame.from_records(aggregateDataSynced, columns = cols)

#aggregateDf["LIBOR"]    = Rf*100  #add LIBOR
#aggregateDf["Rf Daily"] = RfDaily #add daily Rf
#aggregateDf["MAVolume"] = MAVolume
#aggregateDf["MADollarVolume"] = MADollarVolume

# if UnderlyingTicker == "VIX":
#     aggregateDf["frontPrices"] = frontPrices
#     aggregateDf["backPrices"]  = backPrices
#     aggregateDf["frontVolume"] = frontVolume
#     aggregateDf["backVolume"]  = backVolume

aggregateDf  = aggregateDf.iloc[lookback:, :]


## EXPORT DATA TO EXCEL ##
saveloc = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
aggregateDf.to_csv(path_or_buf = saveloc + UnderlyingTicker + "AggregateData.csv" , index = False)








sys.exit()












returns   = np.concatenate((np.zeros((1,)), UnderlyingPrices[1:] / UnderlyingPrices[0:-1] - 1), axis = 0)
dates4fig = pd.to_datetime(aggregateData[:, 0], format = '%Y%m%d')

##################################
# Plots #
#Gamma Exposure Plot
lag = 1 
plt.figure()
plt.scatter(aggregateData[0:-lag, 1], np.abs(returns[lag:]), color = "blue", s = 3)
plt.title("Gamma Exposure vs Absolute Returns, Lag = " + str(lag) + " Day(s)")
plt.xlabel("Market Maker Net Gamma Exposure")
plt.ylabel("Absolute Returns")

lag = 1 
plt.figure()
plt.scatter(aggregateData[0:-lag, 1], returns[lag:], color = "blue", s = 3)
plt.title("Gamma Exposure vs Underlying Returns, Lag = " + str(lag) + " Day(s)")
plt.xlabel("Market Maker Net Gamma Exposure")
plt.ylabel("Underlying Returns")

lag = 1 
plt.figure()
plt.scatter(aggregateData[0:-lag, 1], returns[lag:]**2, color = "blue", s = 3)
plt.title("Gamma Exposure vs Underlying Squared Returns, Lag = " + str(lag) + " Day(s)")
plt.xlabel("Market Maker Net Gamma Exposure")
plt.ylabel("Squared Returns")




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
    
    
    
    
    
    
    
    
    
