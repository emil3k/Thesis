# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:46:28 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import Backtest as bt
import matplotlib.pyplot as plt
import sys

data = pd.read_csv(r'C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\AggregateData\SPXAggregateData.csv')
dates            = data["Dates"].to_numpy()
dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')

UnderlyingTicker = "SPX"

UnderlyingPrice  = data[UnderlyingTicker].to_numpy()
nDays            = np.size(dates)
Returns = UnderlyingPrice[1:] / UnderlyingPrice[0:-1] - 1
Returns = np.concatenate((np.zeros((1,)), Returns), axis = 0) #add zero return for day 1

#Moving average smoothing for visualizations
smoothCols = np.array(["aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker + " Volume",\
                            UnderlyingTicker + " Dollar Volume"])

dataToSmooth  = data[smoothCols].to_numpy()
nCols         = np.size(dataToSmooth, 1)

lookback           = 200 #20 day moving average
aggregateSmooth    = np.zeros((nDays, nCols)) #preallocate

#compute moving average smoothing
for i in np.arange(lookback, nDays):
    periodData = dataToSmooth[i - lookback:i, :]
 
    #Smooth data
    aggregateSmooth[i, :] = np.mean(periodData, 0)
   
#Trim
aggregateSmooth = aggregateSmooth[lookback:, :]
smoothdates     = dates4fig[lookback:, ]
#plot Buckets function
#returns should be same format as netGamma (i.e 0 in first row)
def plotBucketStats(netGamma, Returns, lag, nBuckets):
    nDays       = np.size(netGamma)
    quantileInd = np.floor(np.linspace(0, nDays - lag, nBuckets + 1)) #Index of buckets
    
    netGammaTr  = netGamma[0:-lag] #trim to lag
    ReturnsTr   = Returns[lag:]    #trim to lag
    
    sortMat     = np.concatenate((netGammaTr.reshape(nDays-lag, 1), ReturnsTr.reshape(nDays-lag, 1)), axis = 1) #concatenate
    sortDf      = pd.DataFrame.from_records(sortMat, columns = ["netGamma", "Returns"]) #transform to pandas dataframe for clean sorting
    sortDf      = sortDf.sort_values("netGamma", ascending = True) #sort
    sortMat     = sortDf.to_numpy() #back to numpy for bucketing

    bucketMeans    = np.zeros((nBuckets, ))
    bucketAbsMeans = np.zeros((nBuckets, ))
    bucketStd      = np.zeros((nBuckets, ))
    #Construct buckets
    for i in np.arange(0, nBuckets):
        bucketStart = int(quantileInd[i])
        bucketEnd   = int(quantileInd[i + 1])
        
        bucketReturns  = sortMat[bucketStart:bucketEnd, 1]
        
        bucketMeans[i]    = np.mean(bucketReturns)
        bucketAbsMeans[i] = np.mean(np.abs(bucketReturns))
        bucketStd[i]      = np.std(bucketReturns)
    
    #Plot bucket results
    x = np.arange(1, nBuckets + 1)
    width = 0.7
    plt.figure()
    plt.bar(x, bucketAbsMeans, width = width, color = "blue", label = UnderlyingTicker)  
    plt.title("Average Absolute Returns by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Average Absolute Returns")
    plt.legend()
    plt.show()
    
    x = np.arange(1, nBuckets + 1)
    plt.figure()
    plt.bar(x, bucketMeans, width = width, color = "blue", label = UnderlyingTicker)  
    plt.title("Average Returns by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Average Returns")
    plt.legend()
    plt.show()
        
    x = np.arange(1, nBuckets + 1)
    plt.figure()
    plt.bar(x, bucketStd, width = width, color = "blue", label = UnderlyingTicker)  
    plt.title("Standard Deviation by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.show()

    return bucketMeans, bucketAbsMeans, bucketStd

netGamma    = data["netGamma"].to_numpy()
[bMeans, bAbsMeans, bStd] = plotBucketStats(netGamma, Returns, lag = 1, nBuckets = 6)



sys.exit()

#Bucket by net gamma
nBuckets    = 6
lag         = 1
quantileInd = np.floor(np.linspace(0, nDays - lag, nBuckets + 1)) #Index of buckets
netGamma    = data["netGamma"].to_numpy()
netGammaTr  = netGamma[0:-lag] #trim to lag
ReturnsTr   = Returns[lag:]    #trim to lag
sortMat     = np.concatenate((netGammaTr.reshape(nDays-lag, 1), ReturnsTr.reshape(nDays-lag, 1)), axis = 1) #concatenate
sortDf      = pd.DataFrame.from_records(sortMat, columns = ["netGamma", "Returns"]) #transform to pandas dataframe for clean sorting
sortDf      = sortDf.sort_values("netGamma", ascending = True) #sort
sortMat     = sortDf.to_numpy() #back to numpy for bucketing

bucketMeans    = np.zeros((nBuckets, ))
bucketAbsMeans = np.zeros((nBuckets, ))
bucketStd      = np.zeros((nBuckets, ))
#Construct buckets
for i in np.arange(0, nBuckets):
    bucketStart = int(quantileInd[i])
    bucketEnd   = int(quantileInd[i + 1])
    
    bucketReturns  = sortMat[bucketStart:bucketEnd, 1]
    
    bucketMeans[i]    = np.mean(bucketReturns)
    bucketAbsMeans[i] = np.mean(np.abs(bucketReturns))
    bucketStd[i]      = np.std(bucketReturns)
    
#Plot bucket results
x = np.arange(1, nBuckets + 1)
width = 0.7
plt.figure()
plt.bar(x, bucketAbsMeans, width = width, color = "blue")  
plt.title("Average Absolute Returns by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
plt.ylabel("Average Absolute Returns")

x = np.arange(1, nBuckets + 1)
plt.figure()
plt.bar(x, bucketMeans, width = width, color = "blue")  
plt.title("Average Returns by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
plt.ylabel("Average Returns")

x = np.arange(1, nBuckets + 1)
plt.figure()
plt.bar(x, bucketStd, width = width, color = "blue")  
plt.title("Standard Deviation by Sorted Gamma Exposure, Lag = " + str(lag) + " day")
plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
plt.ylabel("Standard Deviation")


#Plots
plt.figure()
plt.plot(smoothdates, aggregateSmooth[:, -3] / 1000000, color = "blue", label = "Delta Adjusted Option Volume")
plt.plot(smoothdates, aggregateSmooth[:, -2]/ 1000000, color = "black", label = "Volume Underlying")
plt.title(str(lookback) + "-day MA Volume for Options and Underlying")
plt.ylabel("Volume")
plt.yscale("log")























