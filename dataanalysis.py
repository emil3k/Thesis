# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:46:28 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import Backtest as bt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

### SET WHICH ASSET TO BE IMPORTED #######################################################
UnderlyingAssetName   = "SPY Index"
UnderlyingTicker      = "SPY"
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load data
data    = pd.read_csv(loadloc + UnderlyingTicker + "AggregateData.csv") #assetdata
SPXData = pd.read_csv(loadloc + "SPXAggregateData.csv")                 #SPX data for reference

dates            = data["Dates"].to_numpy()
dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
UnderlyingPrice  = data[UnderlyingTicker].to_numpy()
nDays            = np.size(dates)

#Grab futures returns and dates if underlying is VIX
if UnderlyingTicker == "VIX":
    frontPrices      = data["frontPrices"].to_numpy()
    frontXsReturns   = frontPrices[1:, :] / frontPrices[0:-1, :] - 1
    frontXsReturns   = np.concatenate((np.zeros((1, )), frontXsReturns), axis = 0)
    

#trim SPX data to match underlying
startDate = dates[0]
endDate   = dates[-1] + 1
SPXDataTr = bt.trimToDates(SPXData, SPXData["Dates"], startDate, endDate)
SPXPrice  = SPXDataTr["SPX"].to_numpy()


#Compute Returns
if UnderlyingTicker == "VIX":
    Returns = frontXsReturns
else:
    Returns    = UnderlyingPrice[1:] / UnderlyingPrice[0:-1] - 1
    Returns    = np.concatenate((np.zeros((1,)), Returns), axis = 0) #add zero return for day 1

SPXReturns = SPXPrice[1:] / SPXPrice[0:-1] - 1
SPXReturns = np.concatenate((np.zeros((1,)), SPXReturns), axis = 0) #add zero return for day 1

#Extract net gamma
netGamma        = data["netGamma"].to_numpy()
netGamma_alt    = data["netGamma_alt"].to_numpy()
netGammaSPXTr   = SPXDataTr["netGamma"].to_numpy()
netGammaSPX     = SPXData["netGamma"].to_numpy()

#Investigate proporties of gamma
def computeNegGammaStreaks(netGamma, UnderlyingTicker = None , hist = True, color = '#0504aa', histtype = 'stepfilled'):
    nDays  = np.size(netGamma)
    streak = 0
    negGammaStreaks = []    
    for i in np.arange(0, nDays):
        if netGamma[i] < 0: #Start new streak
           streak = streak + 1
        elif streak > 0:
           negGammaStreaks.append(streak)
           streak = 0 #reset streak
       
    negGammaStreaks = np.transpose(np.array([negGammaStreaks]))
    
    if hist == True: #plot histogram
        plt.figure()
        n, bins, patches = plt.hist(x = negGammaStreaks, bins='auto', color=color, histtype = histtype)
        plt.xlabel('Streak Size')
        plt.ylabel('# of Days')
        plt.title('Streaks of Negative Gamma Exposure (# of Days) for ' + UnderlyingTicker)
        plt.show()
        
    return negGammaStreaks
def computeGammaStats(netGamma, UnderlyingTicker = None, hist = True, color = '#0504aa', histtype = 'stepfilled'):    
    nDays = np.size(netGamma)
    negGammaStreaks    = computeNegGammaStreaks(netGamma, hist = False)
    avgStreakLength    = np.round(np.mean(negGammaStreaks), decimals = 2)
    nDaysNegative      = np.sum(netGamma < 0)
    nDaysPositive      = np.sum(netGamma > 0)
    negGammaFraction   = np.round(nDaysNegative / nDays, decimals = 2)
    avgNetGamma        = np.round(np.mean(netGamma)/1000, decimals = 2)
    
    legend = np.array(["Total No. of Days", "No. of Negative Net Gamma Days", "No. of Positive Net Gamma Days", "Negative Net Gamma Fraction", "Average Net Gamma Exposure (1000s)", "Average Cond. Neg. Gamma Streak"])
    netGammaStats = np.array([nDays, nDaysNegative, nDaysPositive, negGammaFraction, avgNetGamma, avgStreakLength])
    
    if hist == True: #Plot netGamma histogram
        plt.figure()
        n, bins, patches = plt.hist(x = netGamma, bins='auto', color=color, histtype = histtype)
        plt.xlabel('MM Net Gamma Exposure')
        plt.ylabel('# of Days')
        plt.title('MM Net Gamma Exposure Distribution for ' + UnderlyingTicker)
        plt.show()
        
    
    return legend, netGammaStats
    
[legend, gammaStats]    = computeGammaStats(netGamma, UnderlyingTicker, hist = True)
[legend, gammaStatsSPX] = computeGammaStats(netGammaSPX, "SPX", hist = True, color = "red")

#Store in dataframe
gammaStatsDf = pd.DataFrame()
gammaStatsDf["Statistics"]       = legend
gammaStatsDf[UnderlyingTicker]   = gammaStats
gammaStatsDf["SPX"]              = gammaStatsSPX

negGammaStreaks    = computeNegGammaStreaks(netGamma, UnderlyingTicker)
negGammaStreaksSPX = computeNegGammaStreaks(netGammaSPX, "SPX", color = "red")


sys.exit()



# #Set up Regression
lag = 1
# X   = netGamma[0:-lag]
# X   = sm.add_constant(X)
# y   = np.abs(Returns[lag:])

# regression = sm.OLS(y, X).fit()
# print(regression.summary())



#Normalize Price Series
netGamma_norm    = (netGamma - np.mean(netGamma)) / np.std(netGamma)
netGammaSPX_norm = (netGammaSPXTr - np.mean(netGammaSPXTr)) / np.std(netGammaSPXTr)

dataToSmooth  = np.concatenate((netGamma.reshape(nDays, 1), netGammaSPXTr.reshape(nDays, 1),\
                netGamma_norm.reshape(nDays, 1), netGammaSPX_norm.reshape(nDays, 1)  ), axis = 1)

def smoothData(dataToSmooth, dates, lookback):
    nCols         = np.size(dataToSmooth, 1)
    nDays         = np.size(dataToSmooth, 0)
    smoothData    = np.zeros((nDays, nCols)) #preallocate

    #compute moving average smoothing
    for i in np.arange(lookback, nDays):
        periodData = dataToSmooth[i - lookback:i, :]
     
        #Smooth data
        smoothData[i, :] = np.mean(periodData, 0)
       
    #Trim
    smoothData      = smoothData[lookback:, :]
    smoothDates     = dates4fig[lookback:, ]

    return smoothData, smoothDates

lookback = 100
[smoothGamma, smoothDates] = smoothData(dataToSmooth, dates, lookback)

## Plot net Gamma of underlying and SPX
#  Smooth and normalized
plt.figure()
plt.plot(smoothDates, smoothGamma[:, 2], color = "blue", label = UnderlyingTicker)
plt.plot(smoothDates, smoothGamma[:, 3], color = "black", label = "SPX")
plt.title(str(lookback) + "-day Normalized MovAvg MM Net Gamma Exposure")
plt.ylabel("Net Gamma Exposure")
plt.legend()

#Normalized
plt.figure()
plt.plot(dates4fig, netGamma_norm, color = "blue", label = UnderlyingTicker)
plt.plot(dates4fig, netGammaSPX_norm, color = "black", label = "SPX")
plt.title("MM Net Gamma Exposure for " + UnderlyingTicker + " and SPX")
plt.ylabel("Net Gamma Exposure")
plt.legend()

## Scatter plot
plt.figure()
plt.scatter(netGamma[0:-lag], Returns[lag:], color = "blue", s = 0.7)
plt.title("Returns vs Gamma for " + UnderlyingTicker + ", lag = " + str(lag) + " day")
plt.ylabel(UnderlyingTicker + " Returns")
plt.xlabel("Market Maker Net Gamma Exposure")
plt.legend()

## Scatter plot
plt.figure()
plt.scatter(netGammaSPX[0:-lag], Returns[lag:], color = "blue", s = 0.7)
plt.title("Returns vs MM net Gamma for SPX, lag = " + str(lag) + " day")
plt.ylabel(UnderlyingTicker + " Futures Returns")
plt.xlabel("Market Maker Net SPX Gamma Exposure")
plt.legend()


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

#plot Buckets
[bMeans, bAbsMeans, bStd] = plotBucketStats(netGamma, Returns, lag = 1, nBuckets = 6)


## Open Interest and Volume Investigation
#Moving average smoothing for visualizations
smoothCols = np.array(["aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker + " Volume",\
                            UnderlyingTicker + " Dollar Volume"])

dataToSmooth = data[smoothCols].to_numpy()
lookback = 100
[aggregateSmooth, smoothDates] = smoothData(dataToSmooth, dates, lookback)


#Plots
plt.figure()
plt.plot(smoothDates, aggregateSmooth[:, -3] / 1000000, color = "blue", label = "Delta Adjusted Option Volume")
plt.plot(smoothDates, aggregateSmooth[:, -2]/ 1000000, color = "black", label = "Volume " + UnderlyingTicker)
plt.title(str(lookback) + "-day MA Volume for " + UnderlyingTicker + " and "+ UnderlyingTicker + " Options")
plt.ylabel("Volume (log scale)")
plt.legend()
plt.yscale("log")

#Plots
plt.figure()
plt.plot(smoothDates, aggregateSmooth[:, 2] / 1000000,color = "blue", label = "Delta Adjusted Open Interest")
#plt.plot(smoothDates, aggregateSmooth[:, -2]/ 1000000, color = "black", label = "Volume " + UnderlyingTicker)
plt.title(str(lookback) + "-day MA Delta Adj. Open Interest for "+ UnderlyingTicker + " Options")
plt.ylabel("Delta Adj. Open Interest (in Millions)")
plt.legend()
#plt.yscale("log")
























