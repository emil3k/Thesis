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
UnderlyingAssetName   = "SPX Index"
UnderlyingTicker      = "SPX"
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load data
dataAll    = pd.read_csv(loadloc + UnderlyingTicker + "AggregateData.csv") #assetdata
SPXDataAll = pd.read_csv(loadloc + "SPXAggregateData.csv")                 #SPX data for reference

#Sample Split
datesAll         = dataAll["Dates"].to_numpy()
SPXdatesAll      = SPXDataAll["Dates"].to_numpy()
fullSampleOnly   = True

if fullSampleOnly == True:
    startDates       = [datesAll[0]]
    endDates         = [datesAll[-1] + 1]
    periodLabels     = [str(int(np.floor(datesAll[0] / 10000))) + " - " + str(int(np.floor(datesAll[-1] / 10000)) + 1)]
else:
    startDates       = [datesAll[0], 20090101, datesAll[0]]
    endDates         = [20081231, datesAll[-1] + 1, datesAll[-1] + 1]
    periodLabels     = [str(int(np.floor(datesAll[0] / 10000))) + " - 2009", "2009 - " + str(int(np.floor(datesAll[-1] / 10000)) + 1), \
                        str(int(np.floor(datesAll[0] / 10000))) + " - " + str(int(np.floor(datesAll[-1] / 10000)) + 1)]

    
nSplits = np.size(startDates)


for i in np.arange(0, nSplits):
   
    #Split data sample
    data        = bt.trimToDates(dataAll, datesAll, startDates[i], endDates[i])
    SPXData     = bt.trimToDates(SPXDataAll, SPXdatesAll, startDates[i], endDates[i])
    periodLabel = periodLabels[i]
    
    #Print for check
    print(data.head())
    print(data.tail())
    
    print(SPXData.head())
    print(SPXData.tail())
    
    
    dates            = data["Dates"].to_numpy()
    dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
    daycount         = bt.dayCount(dates4fig)
    
    UnderlyingPrice  = data[UnderlyingTicker].to_numpy()
    Rf               = data["LIBOR"].to_numpy() / 100
    RfDaily          = np.zeros((np.size(Rf, 0), ))
    RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
    
    nDays            = np.size(dates)
    
    #Grab futures returns and dates if underlying is VIX
    if UnderlyingTicker == "VIX":
        frontPrices      = data["frontPrices"].to_numpy()
        frontXsReturns   = frontPrices[1:] / frontPrices[0:-1] - 1
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
    
    #Extract market cap
    marketCap = data["Market Cap"].to_numpy()
    marketCapSPX = SPXData["Market Cap"].to_numpy()
    
 
    
    #Investigate proporties of gamma
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
        
    [legend, gammaStats]    = computeGammaStats(netGamma, UnderlyingTicker, printTable = True, hist = True, periodLabel = periodLabel)
    [legend, gammaStatsSPX] = computeGammaStats(netGammaSPX, "SPX", hist = True, color = "red", periodLabel = periodLabel)
    
   
    #Store in dataframe
    gammaStatsDf = pd.DataFrame()
    gammaStatsDf["Statistics"]       = legend
    gammaStatsDf[UnderlyingTicker]   = gammaStats
    gammaStatsDf["SPX"]              = gammaStatsSPX
    
    negGammaStreaks    = computeNegGammaStreaks(netGamma, UnderlyingTicker, periodLabel = periodLabel)
    negGammaStreaksSPX = computeNegGammaStreaks(netGammaSPX, "SPX", color = "red", periodLabel = periodLabel)
    
    
    #Normalize Series
    lag = 1
    netGamma_norm    = (netGamma - np.mean(netGamma)) / np.std(netGamma)
    netGammaSPX_norm = (netGammaSPXTr - np.mean(netGammaSPXTr)) / np.std(netGammaSPXTr)
   
    netGamma_scaled    = netGamma / marketCap
    netGammaSPX_scaled = netGammaSPXTr / marketCap
   
    #Collect gamma measures to smooth
    dataToSmooth     = np.concatenate((netGamma.reshape(-1, 1), netGammaSPXTr.reshape(-1, 1),\
                       netGamma_norm.reshape(-1, 1), netGammaSPX_norm.reshape(-1, 1), netGamma_scaled.reshape(-1,1), \
                       netGammaSPX_scaled.reshape(-1,1) ), axis = 1)
    
    #Smooth data and plot    
    def smoothData(dataToSmooth, dates, lookback, plotResults = False, UnderlyingTicker = ""):
        
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
             
        if plotResults == True:
            plt.figure()
            plt.plot(smoothDates, smoothData[:, -2], color = '#0504aa', label = UnderlyingTicker + "")
            if UnderlyingTicker != "SPX":
                plt.plot(smoothDates, smoothData[:, -1], color = "black", label = "SPX")
            plt.title(str(lookback) + "-day MA Net Gamma Exposure" + " (" + periodLabel + ")")
            plt.ylabel("Net Gamma Exposure / Market Cap")
            plt.legend()
            plt.show()
    
        return smoothData, smoothDates
    
    #Plot gamma scatter plots and normalized time series
    def gammaPlots(netGamma, netGamma_scaled, Returns, lag, UnderlyingTicker = "", periodLabel = ""):
        
        
        #Market Cap Scaled
        plt.figure()
        plt.plot(dates4fig, netGamma_scaled, color = '#0504aa')
        plt.title("MM Net Gamma Exposure for " + UnderlyingTicker + " (" + periodLabel + ")")
        plt.ylabel("Net Gamma Exposure / Market Cap")
        plt.legend()
        plt.show()
              
        ## Scatter plot
        plt.figure()
        plt.scatter(netGamma[0:-lag], Returns[lag:], color = '#0504aa', s = 0.7)
        plt.title("Returns vs Net Gamma for " + UnderlyingTicker + ", lag = " + str(lag) + " day" + " (" + periodLabel + ")")
        plt.ylabel(UnderlyingTicker + " Returns")
        plt.xlabel("Market Maker Net Gamma Exposure")
        plt.legend()
        
        if UnderlyingTicker != "SPX":
            ## Scatter plot
            plt.figure()
            plt.scatter(netGammaSPX[0:-lag], Returns[lag:], color = '#0504aa', s = 0.7)
            plt.title("Returns vs Net Gamma for SPX, lag = " + str(lag) + " day" + " (" + periodLabel + ")")
            plt.ylabel(UnderlyingTicker + " Futures Returns")
            plt.xlabel("Market Maker Net SPX Gamma Exposure")
            plt.legend()
    
      
       
    lookback = 100
    [smoothGamma, smoothDates] = smoothData(dataToSmooth, dates, lookback, plotResults = True, UnderlyingTicker = UnderlyingTicker)
    gammaPlots(netGamma, netGamma_scaled, Returns, lag = 1, UnderlyingTicker = UnderlyingTicker, periodLabel = periodLabel)
    
    ######### BUCKETS ##############
    #plot Buckets function
    #returns should be same format as netGamma (i.e 0 in first row)
    def plotBucketStats(netGamma, Returns, lag, nBuckets, periodLabel = " "):
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
    
    #plot Buckets
    [bMeans, bAbsMeans, bStd] = plotBucketStats(netGamma, Returns, lag = 1, nBuckets = 6, periodLabel = periodLabel)
    
    #VIX Returns for SPX gamma
    VIXandSPX = plotBucketStats(netGammaSPX, Returns, lag = 1, nBuckets = 6, periodLabel = "SPX Gamma")
    
    ###################################
    
    
    ## Open Interest and Volume Investigation
    #Moving average smoothing for visualizations
   
    deltaAdjNetOpenInterest = data["deltaAdjOpenInterest"].to_numpy()   
    deltaAdjNetOpenInterest_scaled = (deltaAdjNetOpenInterest * UnderlyingPrice) / marketCap
    
    
    smoothCols = np.array(["aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
                            "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker + " Volume",\
                                UnderlyingTicker + " Dollar Volume"])
    
    dataToSmooth = data[smoothCols].to_numpy()
    dataToSmooth = np.concatenate((dataToSmooth, deltaAdjNetOpenInterest_scaled.reshape(-1,1)), axis = 1) #add deltaadjusted open interest scaled
    
    
    lookback = 100
    [aggregateSmooth, smoothDates] = smoothData(dataToSmooth, dates, lookback)
    
    
    ## Volume and open interest over time
    plt.figure()
    plt.plot(smoothDates, aggregateSmooth[:, 6] / 1000000, color = '#0504aa', label = "Delta Adjusted Option Volume")
    plt.plot(smoothDates, aggregateSmooth[:, 7]/ 1000000, color = "black", label = "Volume " + UnderlyingTicker)
    plt.title(str(lookback) + "-day MA Volume for " + UnderlyingTicker + " and "+ UnderlyingTicker + " Options" + " (" + periodLabel + ")")
    plt.ylabel("Volume (log scale)")
    plt.legend()
    plt.yscale("log")
    
    plt.figure()
    plt.plot(smoothDates, aggregateSmooth[:, 2] / 1000000,color = '#0504aa', label = "Delta Adjusted Open Interest")
    #plt.plot(smoothDates, aggregateSmooth[:, -2]/ 1000000, color = "black", label = "Volume " + UnderlyingTicker)
    plt.title(str(lookback) + "-day MA Delta Adj. Open Interest for "+ UnderlyingTicker + " Options" + " (" + periodLabel + ")")
    plt.ylabel("Delta Adj. Open Interest (in Millions)")
    plt.legend()
    #plt.yscale("log")
    
    ## Volume and open interest over time
    plt.figure()
    plt.plot(smoothDates, aggregateSmooth[:, -1] , color = '#0504aa', label = "Delta Adjusted Open Interest")
    plt.title(str(lookback) + "-day MA Net Open Interest for " + UnderlyingTicker + " (" + periodLabel + ")")
    plt.ylabel("Net Open Interest (in USD, Market Cap Adjusted)")
    plt.legend()
    
   
    
    
    
   
    
    
    
    
    ############# REVERSALS #################
    
    def computeReversalBars(netGamma, Returns, lag = 1):
        isNegGamma     = (netGamma[0:-lag] < 0)
        sameDayReturns = Returns[0:-lag]
        nextDayReturns = Returns[lag:] 
    
        #Same Day vs Next Day Reversals
        negNegSameDay = isNegGamma * (sameDayReturns < 0)
        negPosSameDay = isNegGamma * (sameDayReturns > 0)
        posNegSameDay = (isNegGamma == 0) * (sameDayReturns < 0)
        posPosSameDay = (isNegGamma == 0) * (sameDayReturns > 0)
        
        afterNegNegSameDay = nextDayReturns[negNegSameDay] #Returns day after Negative Gamma, Negative Returns 
        afterNegPosSameDay = nextDayReturns[negPosSameDay] #Returns day after Negative Gamma, Positive Returns 
        afterPosNegSameDay = nextDayReturns[posNegSameDay] #Returns day after Positive Gamma, Negative Returns
        afterPosPosSameDay = nextDayReturns[posPosSameDay] #Returns day after Postivie Gamma, Positive Returns  
        
        bars = np.array([np.mean(afterNegNegSameDay), np.mean(afterNegPosSameDay), np.mean(afterPosNegSameDay), np.mean(afterPosPosSameDay)])
       
        return bars
    
    lag   = 1
    bars  = computeReversalBars(netGamma, Returns, lag = lag)
    ticks = np.array(["Neg-Neg", "Neg-Pos", "Pos-Neg", "Pos-Pos"])
    
    
    ## Unconditional Reversals
    # Uncond. on Gamma
    negSameDay      = (Returns[0:-lag] < 0)      #negative return boolean
    posSameDay      = (Returns[0:-lag] > 0)      #positive return boolean
    nextDayReturns  = Returns[lag:]              #next day (lag) return
    afterNegSameDay = nextDayReturns[negSameDay] #conditional mean
    afterPosSameDay = nextDayReturns[posSameDay] #conditional mean
    
    bars2  = np.array([np.mean(afterNegSameDay), np.mean(afterPosSameDay), np.mean(afterNegSameDay), np.mean(afterPosSameDay)])
    
    # Unconditional
    meanReturn = np.mean(Returns[lag:])
    bars3 = np.array([meanReturn, meanReturn, meanReturn, meanReturn])
    
    
    #Bar Plot
    barWidth = 0.3
    # Set position of bar on X axis
    r1 = np.arange(len(bars))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    plt.figure()
    plt.bar(r1, bars, width = barWidth, color = prefColor, label = "Conditional on gamma and return ")  
    plt.bar(r2, bars2, width = barWidth, color = "red", label = "Conditional on return")
    plt.bar(r3, bars3, width = barWidth, color = "black", label = "Unconditonal")
    plt.title("Average Returns By Previous Day Gamma and Return" + " (" + periodLabel + ")")
    plt.xlabel("Previous Day Net Gamma and Return Combinations")
    plt.ylabel("Average Daily Return")
    plt.xticks(r1 + barWidth/2, ticks)
    plt.legend()
    plt.show()


    #Conditional Autocorrelation
    nDays  = np.size(netGamma)
    streak = 0
    negGammaStreaks = []
    negGammaReturns = np.zeros((nDays, nDays))
    j = 0 
    for i in np.arange(0, nDays):
       if netGamma[i] < 0: #Start new streak
          negGammaReturns[i, j] = Returns[i] 
          streak = streak + 1

       elif streak > 0:
          negGammaStreaks.append(streak) #save streak
          negGammaReturns[i, j] = Returns[i] #Save return day after streak ends
          streak = 0 #reset streak
          j = j + 1 # jump to next column

    
    #Compute conditional autocorrelation
    negGammaAutocorr = []
    for j in np.arange(0, nDays):
        col  = negGammaReturns[:, j] #grab column
        ret  = col[np.nonzero(col)]
        if len(ret) > 2:
            corr = np.corrcoef(ret[0:-1],ret[1:])[1, 0] #compute correlation coefficient
            negGammaAutocorr.append(corr) #save

    #Finalize bar plot
    negGammaAutocorr    = np.transpose(np.array([negGammaAutocorr]))
    averageCondAutocorr = np.mean(negGammaAutocorr)
    uncondAutocorr      = np.corrcoef(Returns[0:-1], Returns[1:])[1,0]

    bars = np.array([averageCondAutocorr, uncondAutocorr])
    x    = np.arange(len(bars))
    ticks = np.array(["Cond. on Negative Gamma", "Unconditional"])
    plt.figure()
    plt.bar(x, bars, width = 0.7 , color = prefColor)  
    plt.title("Autocorrelation of Returns" + " (" + periodLabel + ")")
    plt.ylabel("Average Daily Return")
    plt.xticks(x, ticks)
    plt.legend()
    plt.show()











