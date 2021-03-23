# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:25:48 2021

@author: ekblo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Function to compute negative gamma streaks and plot them as bars

def computeNegGammaStreaks(netGamma, UnderlyingTicker = None , hist = True, color = '#0504aa', histtype = 'stepfilled', periodLabel = " "):

    nDays  = np.size(netGamma)
    streak = 0
    negGammaStreaks = []    
    for i in np.arange(0, nDays):
        if netGamma[i] < 0: #Start new streak
           streak = streak + 1
        elif streak > 0:
           negGammaStreaks.append(streak) #save streak
           streak = 0 #reset streak
       
    negGammaStreaks = np.transpose(np.array([negGammaStreaks]))
    
    if hist == True: #plot histogram
        plt.figure()
        n, bins, patches = plt.hist(x = negGammaStreaks, bins='auto', color=color, histtype = histtype)
        plt.xlabel('Streak Size')
        plt.ylabel('# of Days')
        plt.title('Streaks of Negative Gamma Exposure (# of Days) for ' + UnderlyingTicker + " (" + periodLabel + ")")
        plt.show()
        
    return negGammaStreaks



#Function for computing gamma stats
def computeGammaStats(netGamma, UnderlyingTicker = None, printTable = False, hist = True, color = '#0504aa', histtype = 'stepfilled', periodLabel = " "):    
        nDays = np.size(netGamma)
        negGammaStreaks    = computeNegGammaStreaks(netGamma, hist = False)
        avgStreakLength    = np.round(np.mean(negGammaStreaks), decimals = 2)
        nDaysNegative      = np.sum(netGamma < 0)
        nDaysPositive      = np.sum(netGamma > 0)
        negGammaFraction   = np.round(nDaysNegative / nDays, decimals = 2)
        avgNetGamma        = np.round(np.mean(netGamma)/1000, decimals = 2)
        longestStreak      = np.max(negGammaStreaks)
        
        legend = np.array(["Total No. of Days", "No. of Negative Net Gamma Days", "No. of Positive Net Gamma Days", "Negative Net Gamma Fraction",\
                           "Average Net Gamma Exposure (1000s)", "Average Cond. Neg. Gamma Streak", "Longest Streak"])
        netGammaStats = np.array([nDays, nDaysNegative, nDaysPositive, negGammaFraction, avgNetGamma, avgStreakLength, longestStreak])
        
        if hist == True: #Plot netGamma histogram
            plt.figure()
            n, bins, patches = plt.hist(x = netGamma, bins='auto', color=color, histtype = histtype)
            plt.xlabel('MM Net Gamma Exposure')
            plt.ylabel('# of Days')
            plt.title('MM Net Gamma Exposure Distribution for ' + UnderlyingTicker + " (" + periodLabel + ")")
            plt.show()
        
        if printTable == True:
           table = np.concatenate((legend.reshape(-1,1), netGammaStats.reshape(-1,1)), axis = 1) 
           cols  = np.array(["Stats", UnderlyingTicker + " (" + periodLabel + ")"])
           Df  = pd.DataFrame.from_records(table, columns = cols)
           print(Df.to_latex(index=False))
        
        return legend, netGammaStats
    

#Functions for smoothing data on lookback window  
def smoothData(dataToSmooth, dates, lookback, dates4fig = 0):
    
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
    
    if np.dim(dates4fig) > 0:
        smoothDates     = dates4fig[lookback:, ]
    else:
        smoothDates = dates[lookback:, ]     

    return smoothData, smoothDates
    

#plot Buckets function
#returns should be same format as netGamma (i.e 0 in first row)
def plotBucketStats(netGamma, Returns, lag, nBuckets, UnderlyingTicker = " ", periodLabel = " ", ):
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
    plt.bar(x, bucketAbsMeans, width = width, color = '#0504aa', label = UnderlyingTicker)  
    plt.title("Avg. Absolute Returns by Gamma Exposure, Lag = " + str(lag) + " day" + " (" + periodLabel + ")")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Average Absolute Returns")
    plt.legend()
    plt.show()
    
    x = np.arange(1, nBuckets + 1)
    plt.figure()
    plt.bar(x, bucketMeans, width = width, color = '#0504aa', label = UnderlyingTicker)  
    plt.title("Avg. Returns by Gamma Exposure, Lag = " + str(lag) + " day" + " (" + periodLabel + ")")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Average Returns")
    plt.legend()
    plt.show()
        
    x = np.arange(1, nBuckets + 1)
    plt.figure()
    plt.bar(x, bucketStd, width = width, color = '#0504aa', label = UnderlyingTicker)  
    plt.title("Std. by Gamma Exposure, Lag = " + str(lag) + " day" + " (" + periodLabel + ")")
    plt.xlabel("Gamma Exposure (1 is the lowest quantile)")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.show()

    return bucketMeans, bucketAbsMeans, bucketStd

      
    