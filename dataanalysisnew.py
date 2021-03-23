# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:04:18 2021

@author: ekblo
"""
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
import gammafunctions as gf
import sys

### SET WHICH ASSET TO BE IMPORTED #######################################################
UnderlyingAssetName   = "SPX Index"
UnderlyingTicker      = "SPX"

UnderlyingETFName     = "SPY US Equity"
UnderlyingETFTicker   = "SPY"

loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
load_etf = True
##########################################################################################

#Load data
IndexDataAll     = pd.read_csv(loadloc + UnderlyingTicker + "AggregateData.csv") #assetdata
if load_etf == True:
    ETFDataAll   = pd.read_csv(loadloc + UnderlyingETFTicker + "AggregateData.csv")             #SPX data for reference


fullSampleOnly   = True #Use only full sample for this script
indexDatesAll    = IndexDataAll["Dates"].to_numpy() #grab dates from index
ETFDatesAll      = ETFDataAll["Dates"].to_numpy()   #grab dates from etf

###################################################
#Start and end dates for trimming
indexStartDates      = indexDatesAll[0]
indexEndDates        = indexDatesAll[-1] + 1

ETFStartDates        = ETFDatesAll[0]
ETFEndDates          = ETFDatesAll[-1] + 1

periodLabelIndex  = str(int(np.floor(indexDatesAll[0] / 10000))) + " - " + str(int(np.floor(indexDatesAll[-1] / 10000)) + 1)
periodLabelETF    = str(int(np.floor(ETFDatesAll[0] / 10000))) + " - " + str(int(np.floor(ETFDatesAll[-1] / 10000)) + 1)
###################################################


#Split data sample
indexData   = bt.trimToDates(IndexDataAll, indexDatesAll, indexStartDates, indexEndDates)
ETFData     = bt.trimToDates(ETFDataAll, ETFDatesAll, ETFStartDates, ETFEndDates)

#Print for check
print(indexData.head())
print(indexData.tail())

print(ETFData.head())
print(ETFData.tail())

#Dates
indexDates      = indexData["Dates"].to_numpy()
indexDates4fig  = pd.to_datetime(indexDates, format = '%Y%m%d')
daycount        = bt.dayCount(indexDates4fig)

ETFDates        = ETFData["Dates"].to_numpy()
ETFDates4fig    = pd.to_datetime(ETFDates, format = '%Y%m%d')

#Prices
indexPrice       = indexData[UnderlyingTicker].to_numpy()
indexTRPrice     = indexData["TR Index"].to_numpy()
ETFPrice         = ETFData[UnderlyingETFTicker].to_numpy()

#Risk free rate
Rf               = indexData["LIBOR"].to_numpy() / 100
RfDaily          = np.zeros((np.size(Rf, 0), ))
RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 

#Compute Returns
IndexReturns    = indexTRPrice[1:] / indexTRPrice[0:-1] - 1
IndexReturns    = np.concatenate((np.zeros((1,)), IndexReturns), axis = 0) #add zero return for day 1

ETFReturns = ETFPrice[1:] / ETFPrice[0:-1] - 1
ETFReturns = np.concatenate((np.zeros((1,)), ETFReturns), axis = 0) #add zero return for day 1

#Extract net gamma
netGammaIndex = indexData["netGamma"].to_numpy()
netGammaETF   = ETFData["netGamma"].to_numpy()

#Extract market cap
marketCapIndex   = indexData["Market Cap"].to_numpy()
marketCapETF     = ETFData["Market Cap"].to_numpy()

#Compute Scaled Gamma 
netGammaIndex_scaled = netGammaIndex / marketCapIndex
netGammaETF_scaled   = netGammaIndex / marketCapETF


sys.exit()
#Investigate proporties of gamma
[legend, gammaStats]    = gf.computeGammaStats(netGamma, UnderlyingTicker, printTable = True, hist = True, periodLabel = periodLabel)
[legend, gammaStatsSPX] = gf.computeGammaStats(netGammaSPX, "SPX", hist = True, color = "red", periodLabel = periodLabel)

   
#Store in dataframe
gammaStatsDf = pd.DataFrame()
gammaStatsDf["Statistics"]       = legend
gammaStatsDf[UnderlyingTicker]   = gammaStats
gammaStatsDf["SPX"]              = gammaStatsSPX

negGammaStreaks    = gf.computeNegGammaStreaks(netGamma, UnderlyingTicker, periodLabel = periodLabel)
negGammaStreaksSPX = gf.computeNegGammaStreaks(netGammaSPX, "SPX", color = "red", periodLabel = periodLabel)


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

 
   
lookback = 100
[smoothGamma, smoothDates] = gf.smoothData(dataToSmooth, indexDates, lookback)

   
#Gamma plots 
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
    



######### BUCKETS ##############
[bMeans, bAbsMeans, bStd] = gf.plotBucketStats(netGamma, Returns, lag = 1, nBuckets = 6, UnderlyingTicker = UnderlyingTicker, periodLabel = periodLabel) #regular buckets


#VIX Returns for SPX gamma
VIXandSPX = gf.plotBucketStats(netGammaSPX, Returns, lag = 1, nBuckets = 6, periodLabel = "SPX Gamma")

###################################


## Open Interest and Volume Investigation
#Moving average smoothing for visualizations
   
deltaAdjNetOpenInterest = indexData["deltaAdjOpenInterest"].to_numpy()   
deltaAdjNetOpenInterest_scaled = (deltaAdjNetOpenInterest * indexPrice) / marketCap


smoothCols = np.array(["aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker + " Volume",\
                            UnderlyingTicker + " Dollar Volume"])

dataToSmooth = indexData[smoothCols].to_numpy()
dataToSmooth = np.concatenate((dataToSmooth, deltaAdjNetOpenInterest_scaled.reshape(-1,1)), axis = 1) #add deltaadjusted open interest scaled


lookback = 100
[aggregateSmooth, smoothDates] = smoothData(dataToSmooth, indexDates, lookback)


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











