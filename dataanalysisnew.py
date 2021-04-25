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

futuresOpenInterest = 2749010

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
dates4figIndex  = pd.to_datetime(indexDates, format = '%Y%m%d')
daycount        = bt.dayCount(dates4figIndex)

ETFDates        = ETFData["Dates"].to_numpy()
dates4figETF    = pd.to_datetime(ETFDates, format = '%Y%m%d')

#Prices
indexPrice       = indexData[UnderlyingTicker].to_numpy()
indexTRPrice     = indexData["TR Index"].to_numpy()
ETFPrice         = ETFData[UnderlyingETFTicker].to_numpy()

#Risk free rate
Rf               = indexData["LIBOR"].to_numpy() / 100
RfDaily          = np.zeros((np.size(Rf, 0), ))
RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 


########## Return and Gamma Computation #################

#Compute Returns
IndexReturns    = indexTRPrice[1:] / indexTRPrice[0:-1] - 1
IndexReturns    = np.concatenate((np.zeros((1,)), IndexReturns), axis = 0) #add zero return for day 1
IndexXsReturns  = IndexReturns - RfDaily

ETFReturns   = ETFPrice[1:] / ETFPrice[0:-1] - 1
ETFReturns   = np.concatenate((np.zeros((1,)), ETFReturns), axis = 0) #add zero return for day 1
ETFXsReturns = ETFReturns - RfDaily[-len(ETFReturns):]


#ETF Multiplier
if UnderlyingTicker == "NDX":
    ETFMultiplier = 40
else:
    ETFMultiplier = 10
    


#Extract net gamma
netGammaIndex = indexData["netGamma"].to_numpy()
netGammaETF   = ETFData["netGamma"].to_numpy()/(ETFMultiplier**2)

#Extract market cap
marketCapIndex   = indexData["Market Cap"].to_numpy()
marketCapETF     = ETFData["Market Cap"].to_numpy()

#Put and call gamma
putGammaIndex  = indexData["gamma_put"].to_numpy() / marketCapIndex
callGammaIndex = indexData["gamma_call"].to_numpy() / marketCapIndex
putGammaETF    = ETFData["gamma_put"].to_numpy() / ((ETFMultiplier**2)*marketCapIndex[-len(marketCapETF):])
callGammaETF   = ETFData["gamma_call"].to_numpy() / ((ETFMultiplier**2)*marketCapIndex[-len(marketCapETF):])





if UnderlyingETFTicker == "IWM": #replace min value of IWM Market Cap
    isMinVal = (marketCapETF == np.min(marketCapETF))
    minValIdx = int(np.nonzero(isMinVal)[0])  
    marketCapETF[minValIdx - 1:minValIdx + 1] = np.mean(marketCapETF[minValIdx - 5:minValIdx -1]) #replace w/ previos value
    

#Compute Scaled Gamma 
netGammaIndex_scaled = netGammaIndex / marketCapIndex
#netGammaETF_scaled   = netGammaETF / marketCapETF
netGammaETF_scaled   = netGammaETF / marketCapIndex[-len(netGammaETF):]

#Compute normalized gamma
netGammaIndex_scaledNorm = (netGammaIndex_scaled - np.mean(netGammaIndex_scaled)) / np.std(netGammaIndex_scaled)
netGammaETF_scaledNorm = (netGammaETF_scaled - np.mean(netGammaETF_scaled)) / np.std(netGammaETF_scaled)

#Compute aggregate measure of gamma
netGammaETFLong   = np.zeros((len(netGammaIndex),))
netGammaETFLong[-len(netGammaETF):] = netGammaETF
aggregateNetGamma = (netGammaIndex + netGammaETFLong) / marketCapIndex #Aggregate measure



##############################################################
########### #Investigate proporties of gamma ################
[legend, gammaStatsIndex]  = gf.computeGammaStats(netGammaIndex_scaled, UnderlyingTicker, hist = True, periodLabel = periodLabelIndex)
[legend, gammaStatsETF]    = gf.computeGammaStats(netGammaETF_scaled, UnderlyingETFTicker, hist = True, color = "red", periodLabel = periodLabelETF)


#Store in dataframe
gammaStatsDf = pd.DataFrame()
gammaStatsDf["Statistics"]        = legend
gammaStatsDf[UnderlyingTicker]    = gammaStatsIndex
gammaStatsDf[UnderlyingETFTicker] = gammaStatsETF
print(gammaStatsDf)
print(gammaStatsDf.to_latex(index=False))


negGammaStreaksIndex  = gf.computeNegGammaStreaks(netGammaIndex_scaled, UnderlyingTicker, periodLabel = periodLabelIndex)
negGammaStreaksETF    = gf.computeNegGammaStreaks(netGammaETF_scaled, UnderlyingETFTicker, color = "red", periodLabel = periodLabelETF)

#############################################################
############### Autocorrelation #############################
#Autocorrelation Plots
nLags    = 20
corrVecIndex = np.zeros((nLags, ))
corrVecETF   = np.zeros((nLags, ))

netGammaIndex_scaledTr = netGammaIndex_scaled[-len(netGammaETF_scaled):]

for i in np.arange(1, nLags):
    corrVecIndex[i] = np.corrcoef(netGammaIndex_scaledTr[0:-i], netGammaIndex_scaledTr[i:])[0,1]
    corrVecETF[i]   = np.corrcoef(netGammaETF_scaled[0:-i], netGammaETF_scaled[i:])[0,1]

#Autocorrelation plots
x     = np.arange(1, nLags)
width = 0.4
plt.figure()
plt.bar(x + width/2, corrVecIndex[1:], width = width, color = '#0504aa', label = UnderlyingTicker)  
plt.bar(x - width/2, corrVecETF[1:], width = width, color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Autocorrelation for different lags " + "(" + periodLabelETF + ")")
plt.xlabel("Lags in days")
plt.xticks(x, x, rotation='horizontal')
plt.ylabel("Autocorrelation coefficient")
plt.legend()
plt.show()



###############################################################
########### Time Series analysis ##############################
#Collect gamma measures to smooth
dataToSmoothIndex  = np.concatenate((netGammaIndex.reshape(-1, 1), netGammaIndex_scaled.reshape(-1, 1), netGammaIndex_scaledNorm.reshape(-1, 1), putGammaIndex.reshape(-1,1), callGammaIndex.reshape(-1,1)), axis = 1)
dataToSmoothETF    = np.concatenate((netGammaETF.reshape(-1, 1), netGammaETF_scaled.reshape(-1, 1), netGammaETF_scaledNorm.reshape(-1, 1), putGammaETF.reshape(-1,1), callGammaETF.reshape(-1,1)), axis = 1)

lookback = 100
[smoothGammaIndex, smoothDatesIndex] = gf.smoothData(dataToSmoothIndex, indexDates, lookback, dates4figIndex)
[smoothGammaETF, smoothDatesETF]     = gf.smoothData(dataToSmoothETF, ETFDates, lookback, dates4figETF)

#Construct vector of ETF smoothed gamma of equal size as Index  
netGammaETFLong = np.zeros((len(smoothGammaIndex), 3))
netGammaETFLong[0:, :] = np.nan    
netGammaETFLong[-len(smoothGammaETF):, :] = smoothGammaETF[:, 0:3]   

#Compute Correlation
GammaCorr = np.corrcoef(netGammaIndex_scaled[-len(netGammaETF_scaled):], netGammaETF_scaled)[0, 1]
print("Gamma Correlation")
print(GammaCorr)



##### Gamma Time Series Plots ########
#Scaled Gamma Index
plt.figure()
plt.plot(dates4figIndex, netGammaIndex_scaled, color = '#0504aa', alpha = 0.8)
plt.title("Net Gamma Exposure for " + UnderlyingTicker)
plt.ylabel(r'$netGamma_t%.5f$')
plt.legend()
plt.show()
      
#Scaled Gamma ETF
plt.figure()
plt.plot(dates4figETF, netGammaETF_scaled, color = "red", alpha = 0.8)
plt.title("Net Gamma Exposure for " + UnderlyingETFTicker)
plt.ylabel(r'$netGamma_t%.5f$')
plt.legend()
plt.show()


#Scaled Gamma Subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
if UnderlyingTicker == "SPX":
    fig.suptitle('Net Gamma Exposure Over Time')
ax1.plot(dates4figIndex, netGammaIndex_scaled, color = '#0504aa', alpha = 0.8, label = UnderlyingTicker)
ax2.plot(dates4figETF, netGammaETF_scaled, color = "red", alpha = 0.8, label = UnderlyingETFTicker)
ax1.set_ylabel("Market Maker Net Gamma")
fig.legend()


#Smoothed Gamma Index      
plt.figure()
plt.plot(smoothDatesIndex, smoothGammaIndex[:, 1], color = '#0504aa') 
plt.title("100 DMA Net Gamma Exposure for " + UnderlyingETFTicker)
plt.ylabel(r'$netGamma_t%.5f$')
plt.legend()
plt.show()

#Smoothed Gamma ETF  
plt.figure()
plt.plot(smoothDatesETF, smoothGammaETF[:, 1], color = "red", alpha = 0.8)
plt.title("100 DMA Net Gamma Exposure for " + UnderlyingETFTicker)
plt.ylabel(r'$netGamma_t%.5f$')
plt.legend()
plt.show()

#Normalize to compare the two
plt.figure()
plt.plot(smoothDatesIndex, smoothGammaIndex[:, 2], color = '#0504aa', label = UnderlyingTicker)
plt.plot(smoothDatesIndex, netGammaETFLong[:, 2], color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Net Gamma Exposure (100 DMA, Standardized)")
#plt.ylabel("MM Net Gamma Exposure (Standardized)")
plt.legend()
plt.show()


#Put and Call Gamma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
fig.suptitle('Put and Call Gamma Exposure Over Time')
ax1.plot(dates4figIndex, putGammaIndex,  color = '#0504aa', alpha = 0.8, label = "put " + UnderlyingTicker)
ax1.plot(dates4figIndex, callGammaIndex, color = 'silver',  alpha = 0.8, label = "call " + UnderlyingTicker)
ax2.plot(dates4figETF, putGammaETF,      color = "red",     alpha = 0.8, label = "put " + UnderlyingETFTicker)
ax2.plot(dates4figETF, callGammaETF,     color = "silver",  alpha = 0.8, label = "call " + UnderlyingETFTicker)
fig.legend()


#Put and Call Gamma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
if UnderlyingTicker == "SPX":
    fig.suptitle('Put and Call Gamma Exposure Over Time (100 DMA)', y = 0.98)
ax1.plot(smoothDatesIndex, smoothGammaIndex[:, -2], color = '#0504aa', alpha = 0.8, label = "put " + UnderlyingTicker)
ax1.plot(smoothDatesIndex, smoothGammaIndex[:, -1], color = 'silver',  alpha = 0.8, label = "call " + UnderlyingTicker)
ax1.set_ylabel("Aggregate Gamma")
#ax1.set_title(UnderlyingTicker)
ax1.legend()
ax2.plot(smoothDatesETF,   smoothGammaETF[:, -2],   color = "red",     alpha = 0.8, label = "put " + UnderlyingETFTicker)
ax2.plot(smoothDatesETF,   smoothGammaETF[:, -1],   color = "silver",  alpha = 0.8, label = "call " + UnderlyingETFTicker)
#ax2.set_title(UnderlyingETFTicker)
ax2.legend()
#fig.legend()



###### Scatter plot ##############
lag = 1
plt.figure()
plt.scatter(netGammaIndex_scaled[0:-lag], IndexXsReturns[lag:], color = '#0504aa', s = 5)
plt.title("Returns vs Net Gamma for " + UnderlyingTicker + ", lag = " + str(lag) + " day" + " (" + periodLabelIndex + ")")
plt.ylabel(UnderlyingTicker + " Excess Returns")
plt.xlabel(r'$netGamma_t%.5f$')
plt.legend()
    
lag = 1
plt.figure()
plt.scatter(netGammaETF_scaled[0:-lag], ETFXsReturns[lag:], color = "red", alpha = 0.8, s = 5)
plt.title("Returns vs Net Gamma for " + UnderlyingETFTicker + ", lag = " + str(lag) + " day" + " (" + periodLabelETF + ")")
plt.ylabel(UnderlyingTicker + " Excess Returns")
plt.xlabel(r'$netGamma_t%.5f$')
plt.legend()


####################################
######## Bucket Plots ##############

######### BUCKETS ##############
[bMeans, bAbsMeans, bStd] = gf.plotBucketStats(netGammaIndex_scaled, IndexXsReturns, lag = 1, nBuckets = 6, UnderlyingTicker = UnderlyingTicker, color = prefColor, alpha = 0.8, periodLabel = periodLabelIndex) #regular buckets
[bMeans, bAbsMeans, bStd] = gf.plotBucketStats(netGammaETF_scaled, ETFXsReturns, lag = 1, nBuckets = 6, UnderlyingTicker = UnderlyingETFTicker, color = "red", alpha = 0.8, periodLabel = periodLabelETF) #regular buckets

#VIX Returns for SPX gamma
#VIXandSPX = gf.plotBucketStats(netGammaSPX, Returns, lag = 1, nBuckets = 6, periodLabel = "SPX Gamma")
###################################


############ Open Interest and Volume Investigation ################
#Grab needed data
openInterestIndex            = indexData["aggOpenInterest"].to_numpy()
deltaAdjOpenInterestIndex    = indexData["deltaAdjOpenInterest"].to_numpy()  
volumeIndexOptions           = indexData["aggVolume"].to_numpy()
deltaAdjVolumeIndex          = indexData["deltaAdjVolume"].to_numpy()
volumeIndex                  = indexData[UnderlyingTicker + " Volume"].to_numpy()
dollarVolumeIndex            = indexData[UnderlyingTicker + " Dollar Volume"].to_numpy()


openInterestETF          = ETFData["aggOpenInterest"].to_numpy()
deltaAdjOpenInterestETF  = ETFData["deltaAdjOpenInterest"].to_numpy()  
volumeETFOptions         = ETFData["aggVolume"].to_numpy()
deltaAdjVolumeETF        = ETFData["deltaAdjVolume"].to_numpy()
volumeETF                = ETFData[UnderlyingETFTicker + " Volume"].to_numpy()
dollarVolumeETF          = ETFData[UnderlyingETFTicker + " Dollar Volume"].to_numpy()


#Combine volume data for index and etf


#Match size
volumeETFLong      = np.zeros((len(volumeIndex),))
volumeETFLong[0:]  = np.nan
volumeETFLong[-len(volumeETF):] = volumeETF

volumeETFLong2      = np.zeros((len(volumeIndex),))
volumeETFLong2[-len(volumeETF):] = volumeETF


deltaAdjVolumeETFLong      = np.zeros((len(deltaAdjVolumeIndex), ))
deltaAdjVolumeETFLong[0:]  = np.nan
deltaAdjVolumeETFLong[-len(deltaAdjVolumeETF):] = deltaAdjVolumeETF

#Compute total delta adjusted option volume
deltaAdjVolumeETFLong2      = np.zeros((len(deltaAdjVolumeIndex), ))
deltaAdjVolumeETFLong2[-len(deltaAdjVolumeETF):] = deltaAdjVolumeETF
totalDeltaAdjOptionVolume   = deltaAdjVolumeETFLong2/ETFMultiplier + deltaAdjVolumeIndex


openInterestETFLong      = np.zeros((len(openInterestIndex),))
openInterestETFLong[0:]  = np.nan
openInterestETFLong[-len(openInterestETF):] = openInterestETF / ETFMultiplier
openInterestETFShort = openInterestETF / ETFMultiplier

deltaAdjOpenInterestETFLong      = np.zeros((len(openInterestIndex),))
deltaAdjOpenInterestETFLong[0:]  = np.nan
deltaAdjOpenInterestETFLong[-len(deltaAdjOpenInterestETF):] = deltaAdjOpenInterestETF / ETFMultiplier


#Smooth data for nicer plots
volumeDataToSmooth = np.concatenate((volumeIndex.reshape(-1,1), volumeETFLong.reshape(-1,1), deltaAdjVolumeIndex.reshape(-1,1), deltaAdjVolumeETFLong.reshape(-1,1)), axis  = 1)
[smoothVolumeData, smoothVolumeDates] = gf.smoothData(volumeDataToSmooth, dates4figIndex, lookback = 100)

openInterestDataToSmooth = np.concatenate((openInterestIndex.reshape(-1,1), openInterestETFLong.reshape(-1,1), deltaAdjOpenInterestIndex.reshape(-1,1), deltaAdjOpenInterestETFLong.reshape(-1,1)), axis = 1)
[smoothOpenInterestData, smoothOpenInterestDates] = gf.smoothData(openInterestDataToSmooth, dates4figIndex, lookback = 100)

futuresOpenInterestVec = np.zeros((len(openInterestIndex), ))
futuresOpenInterestVec[0:] = np.nan
futuresOpenInterestVec[-1] = futuresOpenInterest



### Volume Plots ####
#Plot Volume Options
plt.figure()
plt.plot(dates4figIndex, deltaAdjVolumeIndex, color = prefColor, alpha = 0.8, label = UnderlyingTicker)
plt.plot(dates4figIndex, deltaAdjVolumeETFLong / ETFMultiplier, color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Delta Adjusted Volume")
plt.ylabel("Volume, (" + UnderlyingTicker + " Equivalent Shares)")
plt.legend()

#Plot Volume Underlying
plt.figure()
plt.plot(dates4figIndex, np.log(volumeIndex), color = prefColor, alpha = 0.8, label = UnderlyingTicker)
plt.plot(dates4figIndex, np.log(volumeETFLong / ETFMultiplier), color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Underlying Volume")
plt.ylabel("Volume, (" + UnderlyingTicker + " Equivalent Shares)")
plt.legend()

#Plot smooth data
plt.figure()
plt.plot(smoothVolumeDates, smoothVolumeData[:, 2], color = prefColor, alpha = 0.8, label = UnderlyingTicker)
plt.plot(smoothVolumeDates, smoothVolumeData[:, 3] / ETFMultiplier, color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Delta Adjusted Volume (100 DMA)")
plt.ylabel("Volume, (" + UnderlyingTicker + " Equivalent Shares)")
plt.legend()

#Plot Volume Underlying
plt.figure()
plt.plot(dates4figIndex, np.log(totalDeltaAdjOptionVolume), color = prefColor, alpha = 0.8, label = "Option Volume")
plt.plot(dates4figIndex, np.log(volumeIndex + volumeETFLong2/ETFMultiplier), color = "red", alpha = 0.8, label = "Index Volume")
plt.title("Underlying vs. Delta Adj. Option Volume, Russell 2000")
plt.ylabel("Volume Log Index Equivalent Units")
plt.legend()


### Open Interest Plots ###

#Smoothed open interest
plt.figure()
plt.plot(smoothOpenInterestDates, smoothOpenInterestData[:, 0] / 1000000, color = prefColor, alpha = 0.8, label = UnderlyingTicker)
plt.plot(smoothOpenInterestDates, smoothOpenInterestData[:, 1] / 1000000, color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.plot(smoothOpenInterestDates, futuresOpenInterestVec[100:]/1000000, color = "k", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Open Interest (100DMA)")
plt.ylabel("Open Interest in Million " + UnderlyingTicker + " Contracts")
plt.legend()

#Smoothed delta adjusted open interest
plt.figure()
plt.plot(smoothOpenInterestDates, smoothOpenInterestData[:, 2], color = prefColor, alpha = 0.8, label = UnderlyingTicker)
plt.plot(smoothOpenInterestDates, smoothOpenInterestData[:, 3], color = "red", alpha = 0.8, label = UnderlyingETFTicker)
plt.title("Delta Adjusted Open Interest (100 DMA)")
plt.ylabel("Open Interest in " + UnderlyingTicker + " Equivalent Shares)")
plt.legend()


#Not smoothed
plt.figure()
plt.plot(dates4figIndex, np.log(openInterestIndex), color = prefColor, alpha = 0.8, label = UnderlyingTicker + " Options")
plt.plot(dates4figIndex, np.log(openInterestETFLong), color = "red", alpha = 0.8, label = UnderlyingETFTicker + " Options")
#plt.scatter(dates4figIndex, np.log(futuresOpenInterestVec), color = "k", alpha = 1, label = "Futures Options")
plt.title("Aggregate Open Interest S&P 500 Instruments")
plt.ylabel("Open Interst in Log Index Equivalent Units")
plt.legend()

plt.figure()
plt.plot(dates4figETF, openInterestIndex[-len(openInterestETFShort):] / openInterestETFShort , color = prefColor, alpha = 0.8, label = UnderlyingTicker + " Options")
#plt.plot(dates4figIndex, np.log(openInterestETFLong), color = "red", alpha = 0.8, label = UnderlyingETFTicker + " Options")
#plt.scatter(dates4figIndex, np.log(futuresOpenInterestVec), color = "k", alpha = 1, label = "Futures Options")
plt.title("Open Interest " + UnderlyingTicker + "/" + UnderlyingETFTicker)
plt.ylabel("Open Interst Ratio Index Equivalent Units")
#plt.legend()




sys.exit()

#smoothCols = np.array(["aggOpenInterest", "netOpenInterest", "deltaAdjOpenInterest",\
#                        "deltaAdjNetOpenInterest", "aggVolum", "deltaAdjVolume", UnderlyingTicker + " Volume",\
#                            UnderlyingTicker + " Dollar Volume"])

#dataToSmooth = indexData[smoothCols].to_numpy()
#dataToSmooth = np.concatenate((dataToSmooth, deltaAdjNetOpenInterest_scaled.reshape(-1,1)), axis = 1) #add deltaadjusted open interest scaled

#lookback = 100
#[aggregateSmooth, smoothDates] = smoothData(dataToSmooth, indexDates, lookback)


  
   

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

lag = 1
CondReversalBarsIndex  = computeReversalBars(netGammaIndex_scaled, IndexXsReturns, lag = lag)
CondReversalBarsETF    = computeReversalBars(netGammaETF_scaled, ETFXsReturns, lag = lag)
ticks = np.array(["Neg-Neg", "Neg-Pos", "Pos-Neg", "Pos-Pos"])


#######################################
### Unconditional on Gamma Reversals ### 
#Index
negSameDayIndex      = (IndexXsReturns[0:-lag] < 0)         #negative return boolean
posSameDayIndex      = (IndexXsReturns[0:-lag] > 0)         #positive return boolean
nextDayReturnsIndex  = IndexXsReturns[lag:]                 #next day (lag) return
afterNegSameDayIndex = nextDayReturnsIndex[negSameDayIndex] #conditional mean
afterPosSameDayIndex = nextDayReturnsIndex[posSameDayIndex] #conditional mean
UncondReversalBarsIndex  = np.array([np.mean(afterNegSameDayIndex), np.mean(afterPosSameDayIndex), np.mean(afterNegSameDayIndex), np.mean(afterPosSameDayIndex)])

#ETF
negSameDayETF          = (ETFXsReturns[0:-lag] < 0)         #negative return boolean
posSameDayETF          = (ETFXsReturns[0:-lag] > 0)         #positive return boolean
nextDayReturnsETF      = ETFXsReturns[lag:]                 #next day (lag) return
afterNegSameDayETF     = nextDayReturnsETF[negSameDayETF]   #conditional mean
afterPosSameDayETF     = nextDayReturnsETF[posSameDayETF]   #conditional mean
UncondReversalBarsETF  = np.array([np.mean(afterNegSameDayETF), np.mean(afterPosSameDayETF), np.mean(afterNegSameDayETF), np.mean(afterPosSameDayETF)])


# Unconditional
meanXsReturnIndex   = np.mean(IndexXsReturns[lag:])
meanXsReturnETF     = np.mean(ETFXsReturns[lag:])
MeanReturnBarsIndex = np.array([meanXsReturnIndex, meanXsReturnIndex, meanXsReturnIndex, meanXsReturnIndex])
MeanReturnBarsETF   = np.array([meanXsReturnETF, meanXsReturnETF, meanXsReturnETF, meanXsReturnETF])


#Bar Plot
barWidth = 0.3
# Set position of bar on X axis
r1 = np.arange(len(CondReversalBarsIndex))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


#Bar Plot Index alternative color
plt.figure()
plt.bar(r1, CondReversalBarsIndex*100,   width = barWidth, color = prefColor, alpha = 0.8, label = "Conditional on gamma and return")  
plt.bar(r2, UncondReversalBarsIndex*100, width = barWidth, color = "red", alpha = 0.8,  label = "Conditional on return")
plt.bar(r3, MeanReturnBarsIndex*100,     width = barWidth, color = "silver",  alpha = 0.8, label = "Unconditonal")
plt.title("Avg. Returns By Prev. Day Gamma and Return, " + UnderlyingTicker)
plt.xlabel("Previous Day Net Gamma and Return Combinations" + " (" + periodLabelIndex + ")")
plt.ylabel("Average Daily Excess Return (%)")
plt.xticks(r1 + barWidth/2, ticks)
plt.legend()
plt.show()


#Bar Plot Index
plt.figure()
plt.bar(r1, (CondReversalBarsIndex - MeanReturnBarsIndex)*100,   width = barWidth, color = prefColor, alpha = 0.8, label = "Cond. on gamma and return")  
plt.bar(r2, (UncondReversalBarsIndex - MeanReturnBarsIndex)*100, width = barWidth, color = "red", alpha = 0.8,  label = "Cond. on return")
#plt.bar(r3, MeanReturnBarsIndex*100,     width = barWidth, color = "silver",  alpha = 0.8, label = "Unconditonal")
plt.title("Avg. Returns By Prev. Day Gamma and Return, " + UnderlyingTicker)
plt.xlabel("Previous Day Net Gamma and Return Combinations" + " (" + periodLabelIndex + ")")
plt.ylabel("Avg. Cond. Return minus Sample Average (%)")
plt.xticks(r1 + barWidth/2, ticks)
plt.axhline(y=0, color='k', linestyle='-', linewidth = 1)
plt.legend()
plt.show()


#Bar Plot ETF
plt.figure()
plt.bar(r1, (CondReversalBarsETF - MeanReturnBarsETF)*100,   width = barWidth, color = prefColor, alpha = 0.8, label = "Cond. on gamma and return")  
plt.bar(r2, (UncondReversalBarsETF - MeanReturnBarsETF) *100, width = barWidth, color = "red", alpha = 0.8,  label = "Cond. on return")
#plt.bar(r3, MeanReturnBarsETF*100,     width = barWidth, color = "silver",  alpha = 0.8, label = "Unconditonal")
plt.title("Avg. Returns By Prev. Day Gamma and Return, " + UnderlyingETFTicker)
plt.xlabel("Previous Day Net Gamma and Return Combinations" + " (" + periodLabelIndex + ")")
plt.ylabel("Avg. Cond. Return minus Sample Average (%)")
plt.xticks(r1 + barWidth/2, ticks)
plt.axhline(y=0, color='k', linestyle='-', linewidth = 1)
plt.legend()
plt.show()


############################
#Assumptions Score Bar plot
IndexScores = np.array([7, 4, 0.1, 1])
StockScores = np.array([2, 4, 5, 3])
labels      = np.array(["A1 Support", "A2 Support", "A1 Contradict", "A2 Contradict"])

#Autocorrelation plots
x     = np.arange(0, len(IndexScores))
width = 0.3
plt.figure()
plt.bar(x - width/2, IndexScores, width = width, color = '#0504aa', alpha = 0.8, label = "Index")  
plt.bar(x + width/2, StockScores, width = width, color = "red", alpha = 0.8, label = "Single Name")
plt.title("Assumption Scores for Index and Single Names")
plt.ylabel("Number of Papers Supporting/Contradicting")
plt.xticks(x, labels, rotation='horizontal')
plt.legend()
plt.show()










sys.exit()

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



##########################################
### Gamma Correlation Between Index and ETF
GammaCorrelation = np.array([0.74, 0.35, 0.42])
Pairs = np.array(["SPX/SPY", "NDX/QQQ", "RUT/IWM"])
nPairs = len(Pairs)

#Correlation plot
x     = np.arange(0, nPairs)
width = 0.5
plt.figure(figsize = (10, 4))
plt.bar(x, GammaCorrelation, width = width, color = '#0504aa', alpha = 0.8)  
plt.title("Net Gamma Correlation between Index and ETF")
plt.xticks(x, Pairs, rotation='horizontal')
plt.ylabel("Correlation Coefficient")
plt.legend()
plt.show()










