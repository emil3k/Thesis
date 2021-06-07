# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:54:41 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import Backtest as bt
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sn
import sys
from sklearn.linear_model import LassoCV
#Regressions

### SET IMPORT PARAMS ####################################################################
#UnderlyingAssetName   = ["SPX Index", "SPY US Equity", "NDX Index", "QQQ US Equity", "RUT Index", "IWM US Equity", "VIX Index"]
#UnderlyingTicker      = ["SPX", "SPY", "NDX", "QQQ", "RUT", "IWM", "VIX"]
#IsEquityIndex        = [True, False, True, False, True, False, False]
UnderlyingAssetName   = ["SPX US Equity"]
UnderlyingTicker      = ["SPX"]
IsEquityIndex         = [True]

loadloc               = "../Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData = []
for ticker in UnderlyingTicker:
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    AggregateData.append(data)


def computeRfDaily(data):
    dates            = data["Dates"].to_numpy()
    dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
    daycount         = bt.dayCount(dates4fig)
    
    Rf               = data["LIBOR"].to_numpy() / 100
    RfDaily          = np.zeros((np.size(Rf, 0), ))
    RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
    return RfDaily, dates4fig,daycount


#Compute returns, net Gamma measure and set up strategy for each asset
nAssets      = len(UnderlyingTicker)
TotalReturns = []
XsReturns    = []
for i in np.arange(0, nAssets):
    #Grab needed data
    data         = AggregateData[i]
    ticker       = UnderlyingTicker[i]
    eqIndexFlag  = IsEquityIndex[i]
    rf           = data["LIBOR"].to_numpy()
    dates        = data["Dates"].to_numpy()
    
    #Compute returns
    if ticker  != "VIX":
        #Use Total Return Index for index, and normal price for ETF
        if eqIndexFlag == True:
            price = data["TR Index"].to_numpy()   
        else:
            price    = data[ticker].to_numpy()
           
        rf, dates4fig, daycount = computeRfDaily(data)    
        #Compute returns 
        ret   = price[1:] / price[0:-1] - 1 
        ret   = np.concatenate((np.zeros((1, )), ret), axis = 0)
        xsret = ret - rf
        
        #Store
        TotalReturns.append(ret)
        XsReturns.append(xsret)

    #Use futures for VIX Returns
    else:
       frontPrices = data["frontPrices"].to_numpy()
       rf, dates4fig = computeRfDaily(data)
       
       frontXsReturns = frontPrices[1:] / frontPrices[0:-1] - 1
       frontXsReturns =  np.concatenate((np.zeros((1, )), frontXsReturns), axis = 0)
       frontTotalReturns = frontXsReturns + rf
       
       #Store
       TotalReturns.append(frontTotalReturns)
       XsReturns.append(frontXsReturns)
    
    
    #Gamma measures 
    netGamma       = data["netGamma"].to_numpy()
    marketCap      = data["Market Cap"].to_numpy()
    MAdollarVolume = data["MADollarVolume"].to_numpy() 
    ILLIQ          = data["ILLIQ"].to_numpy()
    ILLIQMedian    = np.median(ILLIQ)    
    ILLIQMean      = np.mean(ILLIQ) 
    
    #Use barbon measure for VIX
    if ticker != "VIX":
        netGamma_scaled = netGamma / marketCap
        netGamma_barbon = netGamma / MAdollarVolume
    else:
        netGamma_scaled = netGamma / MAdollarVolume
     
    #Weekly Strategy
    nDays    = len(rf)
    
    #get first and last day in week
    isFirstDay = np.zeros((nDays,))
    isLastDay  = np.zeros((nDays,))
    for i in np.arange(0, nDays):
        count = daycount[i]
        if count > 1:
            isFirstDay[i] = 1
            isLastDay[i - 1] = 1
    
    isFirstDay[0] = 1
    isLastDay[-1] = 1
    firstDayList = np.nonzero(isFirstDay)[0]
    lastDayList  = np.nonzero(isLastDay)[0]
    
    #Compute Weekly Returns and Weekly Gamma
    nWeeks = len(firstDayList)
    weeklyTotalReturns = np.zeros(nWeeks,)
    weeklyAvgGamma = np.zeros((nWeeks,))
    weeklyRf = np.zeros((nWeeks,))
    for i in np.arange(0, nWeeks):
        start = firstDayList[i]
        stop  = lastDayList[i]
        
        weekReturn = ret[start:stop + 1]
        weekGamma  = netGamma_scaled[start:stop + 1]
        weekRf = rf[start:stop + 1]
        
        weeklyTotalReturns[i] = np.prod(1 + weekReturn)
        weeklyAvgGamma[i]     = np.mean(weekGamma)
        weeklyRf[i]           = np.mean(1 + weekRf)
        
    weeklyXsReturns    = weeklyTotalReturns - weeklyRf
    weeklyGammaSignal  = 1*(weeklyAvgGamma < 0)
    weeklyReturnSignal = 1*(weeklyXsReturns < 0)
    weeklyCombinedSignal = weeklyGammaSignal * weeklyReturnSignal
    
    #Weekly Turnover
    turnoverWeeklyGamma    = np.abs(weeklyGammaSignal[1:] - weeklyGammaSignal[0:-1])
    turnoverWeeklyReturn   = np.abs(weeklyReturnSignal[1:] - weeklyReturnSignal[0:-1])
    turnoverWeeklyCombined = np.abs(weeklyCombinedSignal[1:] - weeklyCombinedSignal[0:-1])
  
    weeklyGammaXsReturns    = weeklyGammaSignal[0:-1] * weeklyXsReturns[1:]
    weeklyReturnXsReturns   = weeklyReturnSignal[0:-1] * weeklyXsReturns[1:]
    weeklyCombinedXsReturns = weeklyCombinedSignal[0:-1] * weeklyXsReturns[1:]
    
    c = 0.0010
    weeklyGammaXsReturnsTC    = weeklyGammaXsReturns - turnoverWeeklyGamma*c 
    weeklyReturnXsReturnsTC   = weeklyReturnXsReturns -  turnoverWeeklyReturn*c
    weeklyCombinedXsReturnsTC = weeklyCombinedXsReturns - turnoverWeeklyCombined*c
    
    #Cumulative Returns
    weeklyGammaCum    = np.cumprod(1 + weeklyGammaXsReturns)
    weeklyReturnCum   = np.cumprod(1 + weeklyReturnXsReturns)
    weeklyCombinedCum = np.cumprod(1 + weeklyCombinedXsReturns)
    weeklyBuyAndHoldCum = np.cumprod(1 + weeklyXsReturns[1:])
    
    #Cumulative Returns TC
    weeklyGammaCumTC    = np.cumprod(1 + weeklyGammaXsReturnsTC)
    weeklyReturnCumTC   = np.cumprod(1 + weeklyReturnXsReturnsTC)
    weeklyCombinedCumTC = np.cumprod(1 + weeklyCombinedXsReturnsTC)
   
    weeklyDates4fig = dates4fig[firstDayList]
    weeklyDates4fig = weeklyDates4fig[1:]
    
    plt.figure()
    plt.plot(weeklyDates4fig, weeklyGammaCum,  color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed")
    plt.plot(weeklyDates4fig, weeklyReturnCum,  color = "red",     alpha = 0.8, label = "Reversal")
    plt.plot(weeklyDates4fig, weeklyCombinedCum, color = "black", alpha = 0.8, label = "Combined Reversal")
    plt.plot(weeklyDates4fig, weeklyGammaCumTC,  linestyle = '--', color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed")
    plt.plot(weeklyDates4fig, weeklyReturnCumTC,  linestyle = '--', color = "red",     alpha = 0.8, label = "Reversal")
    plt.plot(weeklyDates4fig, weeklyCombinedCumTC, linestyle = '--', color = "black", alpha = 0.8, label = "Combined Reversal")
    plt.plot(weeklyDates4fig, weeklyBuyAndHoldCum, color = "silver", alpha = 0.8, label = "Buy And Hold")
    plt.title("Return Reversal Weekly Strategy, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend(loc = 'upper left')  
     
   
    

    #Z-score strategy
    #Generate signal only when Gamma or Returns are large
    lookback = 252
   
    gammaZscore = np.zeros(nDays,)
    returnZscore = np.zeros(nDays,)
    rollingMeanGammaVec = np.zeros(nDays,)
    for i in np.arange(lookback, nDays):
        rollingMeanGamma = np.mean(netGamma_scaled[i - lookback:i])
        rollingMeanGammaVec[i] = rollingMeanGamma
        rollingStdGamma  = np.std(netGamma_scaled[i - lookback:i])
        gammaZscore[i]   = (netGamma_scaled[i] - rollingMeanGamma) / rollingStdGamma
        
        rollingMeanRet = np.mean(xsret[i - lookback:i])
        rollingStdRet  = np.std(xsret[i - lookback:i])
        returnZscore[i]   = (xsret[i] - rollingMeanRet) / rollingStdRet

   
    #Signal Generation
    negGammaSignal    = (netGamma_scaled < 0)*1    
    posGammaSignal    = (netGamma_scaled > 0)*1
    posRetSignal      = (xsret > 0)*1
    negRetSignal       = (xsret < 0)*1
    
    threshold = 1
    gammaZscoreSignal       = 1*(gammaZscore < -threshold)
    returnZscoreSignal      = 1*(returnZscore < -threshold)
    returnZscoreShortSignal = 1*(returnZscore > threshold)
    
    combinedZscoreSignal      = gammaZscoreSignal * returnZscoreSignal
    combinedZscoreShortSignal = gammaZscoreSignal * returnZscoreShortSignal
    
    #Zscore strategies
    gammaZscoreXsReturns    = gammaZscoreSignal[lookback:-1] * xsret[lookback  + 1:]
    returnZscoreXsReturns   = returnZscoreSignal[lookback:-1] * xsret[lookback + 1:]
    combinedZscoreXsReturns = combinedZscoreSignal[lookback:-1]*xsret[lookback + 1:]
    
    returnZscoreShortXsReturns   = -returnZscoreShortSignal[lookback:-1] * xsret[lookback + 1:]
    combinedZscoreShortXsReturns = -combinedZscoreShortSignal[lookback:-1] * xsret[lookback + 1:]
   
    
    #Combined Signals
    negnegSignal  = negGammaSignal * negRetSignal
    negposSignal  = negGammaSignal * posRetSignal
    posnegSignal  = posGammaSignal * negRetSignal    
    posposSignal  = posGammaSignal * posRetSignal
    
    #Single Signal Strategy Returns
    negGammaXsReturns = negGammaSignal[lookback:-1] * xsret[lookback + 1:]
    posGammaXsReturns = posGammaSignal[lookback:-1] * xsret[lookback + 1:]
    negRetXsReturns   = negRetSignal[lookback:-1] * xsret[lookback + 1:]
    posRetXsReturns   = posRetSignal[lookback:-1] * xsret[lookback + 1:]
    
    #Combined Signal Returns
    negnegXsReturns = negnegSignal[lookback:-1] * xsret[lookback + 1:]
    negposXsReturns = negposSignal[lookback:-1] * xsret[lookback + 1:]
    posnegXsReturns = posnegSignal[lookback:-1] * xsret[lookback + 1:]
    posposXsReturns = posposSignal[lookback:-1] * xsret[lookback + 1:]
    
    
    #Long Short Strategies
    negGammaReversalXsReturns = negnegXsReturns - negposXsReturns #long after negative, short after positive return (and negative Gamma)
    posGammaReversalXsReturns = posnegXsReturns - posposXsReturns #long after negative, short after positive return (and positive Gamma)
    returnReversalsXsReturns  = negRetXsReturns - posRetXsReturns #Long after negative, short after positive return (unconditional on gamma) 
    
    
  
    
    
    #Transaction Costs
    #Long and short treated individually
    #Turnover
    turnoverNegNeg = np.abs(negnegSignal[lookback + 1:] - negnegSignal[lookback:-1])
    turnoverNegPos = np.abs(negposSignal[lookback + 1:] - negposSignal[lookback:-1]) 
    turnoverNegRet = np.abs(negRetSignal[lookback + 1:] - negRetSignal[lookback:-1])
    turnoverPosRet = np.abs(posRetSignal[lookback + 1:] - posRetSignal[lookback:-1])    
    
    turnoverGammaZscore    = np.abs(gammaZscoreSignal[lookback + 1:] - gammaZscoreSignal[lookback:-1])
    turnoverReturnZscore   = np.abs(returnZscoreSignal[lookback + 1:] - returnZscoreSignal[lookback:-1])
    turnoverCombinedZscore = np.abs(combinedZscoreSignal[lookback + 1:] - combinedZscoreSignal[lookback:-1])
    
    
    
    c = 0.0005 #one-way cost
    negnegXsReturnsTC = negnegXsReturns - turnoverNegNeg*c
    negRetXsReturnsTC = negRetXsReturns - turnoverNegRet*c
    negposXsReturnsShortTC = -negposXsReturns - turnoverNegPos*c
    posRetXsReturnsShortTC = -posRetXsReturns - turnoverPosRet*c
    
    gammaZscoreXsReturnsTC    = gammaZscoreXsReturns - turnoverGammaZscore * c
    returnZscoreXsReturnsTC   = returnZscoreXsReturns - turnoverReturnZscore * c
    combinedZscoreXsReturnsTC = combinedZscoreXsReturns - turnoverCombinedZscore * c
    
    
    
    #Cumulative Excess Returns
    #Signle Signal Strategies
    negNetGammaCum     = np.cumprod(1 + negGammaXsReturns)
    posNetGammaCum     = np.cumprod(1 + posGammaXsReturns)
    negRetCum          = np.cumprod(1 + negRetXsReturns)
    posRetCum          = np.cumprod(1 + posRetXsReturns)
    posRetCumShort     = np.cumprod(1 - posRetXsReturns)
    buyAndHold         = np.cumprod(1 + xsret[lookback+1:])
 
    #Combined Signal Long Only Strategies
    negnegCum = np.cumprod(1 + negnegXsReturns)
    negposCum = np.cumprod(1 + negposXsReturns)
    negposCumShort = np.cumprod(1 - negposXsReturns)  
    posnegCum = np.cumprod(1 + posnegXsReturns)
    posposCum = np.cumprod(1 + posposXsReturns)
    
    #Zscore
    gammaZscoreCum    = np.cumprod(1 + gammaZscoreXsReturns)
    returnZscoreCum   = np.cumprod(1 + returnZscoreXsReturns)
    combinedZscoreCum = np.cumprod(1 + combinedZscoreXsReturns)
    combinedZscoreShortCum = np.cumprod(1 + combinedZscoreShortXsReturns)
    returnZscoreShortCum   = np.cumprod(1 + returnZscoreShortXsReturns)
    
    gammaZscoreCumTC    = np.cumprod(1 + gammaZscoreXsReturnsTC)
    returnZscoreCumTC   = np.cumprod(1 + returnZscoreXsReturnsTC)
    combinedZscoreCumTC = np.cumprod(1 + combinedZscoreXsReturnsTC)
    
    
    #Transaction Costs
    negnegCumTC = np.cumprod(1 + negnegXsReturnsTC)
    negRetCumTC = np.cumprod(1 + negRetXsReturnsTC)
    negposCumShortTC = np.cumprod(1 + negposXsReturnsShortTC)
    posRetCumShortTC = np.cumprod(1 + posRetXsReturnsShortTC)
    
    
    #Long Short Strategies
    negGammaReversalCum = np.cumprod(1 + negGammaReversalXsReturns)
    posGammaReversalCum = np.cumprod(1 + posGammaReversalXsReturns)
    returnReversalsCum  = np.cumprod(1 + returnReversalsXsReturns)
    
    
    #Average Returns
    StrategyXsReturns = np.concatenate((negnegXsReturns.reshape(-1,1), negRetXsReturns.reshape(-1,1), -negposXsReturns.reshape(-1,1), \
                                  -posRetXsReturns.reshape(-1,1), xsret[lookback + 1:].reshape(-1,1)), axis = 1)

    AverageDailyXsReturns = np.prod(1 + StrategyXsReturns, axis = 0)**(1 / len(StrategyXsReturns)) - 1
    AverageAnnualizedXsReturns = (1 + AverageDailyXsReturns)**252 - 1
    
    
    ######################
    #Plot Equity Lines
    
    # #Single Signal Strategies
    # plt.figure()
    # plt.plot(dates4fig[1:], negNetGammaCum, color = "red",       alpha = 0.8, label = "Long After Negative Net Gamma")
    # plt.plot(dates4fig[1:], posNetGammaCum, color = prefColor,   alpha = 0.8, label = "Long After Positive Net Gamma")
    # plt.plot(dates4fig[1:], negRetCum,      color = "black",     alpha = 0.8, label = "Long After Negative Return")
    # plt.plot(dates4fig[1:], posRetCum,      color = "lightblue", alpha = 0.8, label = "Long After Positive Return")
    # plt.plot(dates4fig[1:], buyAndHold,     color = "lightgrey", alpha = 0.8, label = "Buy-and-Hold")
    # plt.title("Single Signal Long-Only Timing Strategies, " + ticker)
    # plt.ylabel("Cumulative Excess Returns")
    # plt.legend()
    
    # #Combined Signal Strategies
    # plt.figure()
    # plt.plot(dates4fig[1:], negnegCum,  color = prefColor,   alpha = 0.8, label = "Neg-Neg")
    # plt.plot(dates4fig[1:], negposCum,  color = "red",       alpha = 0.8, label = "Neg-Pos")
    # plt.plot(dates4fig[1:], posnegCum,  color = "lightblue", alpha = 0.8, label = "Pos-Neg")
    # plt.plot(dates4fig[1:], posposCum,  color = "black",     alpha = 0.8, label = "Pos-Pos")
    # plt.plot(dates4fig[1:], buyAndHold, color = "lightgrey", alpha = 0.8, label = "Buy-and-Hold")
    # plt.title("Long-Only Strategies Day After \n Given Return and Gamma Combination,  " + ticker)
    # plt.ylabel("Cumulative Excess Returns")
    # plt.legend()


    # #Long Short Strategies
    # plt.figure()
    # plt.plot(dates4fig[1:], negGammaReversalCum,  color = prefColor,   alpha = 0.8, label = "Neg Gamma Reversals")
    # plt.plot(dates4fig[1:], posGammaReversalCum,  color = "red",       alpha = 0.8, label = "Pos Gamma Reversals")
    # plt.plot(dates4fig[1:], returnReversalsCum,   color = "black",     alpha = 0.8, label = "Uncond. Return Reversals")
    # plt.title("Return Reversal Strategies," + ticker)
    # plt.ylabel("Cumulative Excess Returns")
    # plt.legend()
    dates4fig = dates4fig[lookback + 1:]
    #Long Short Strategies
    #Daily Reversal Strategies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
    fig.suptitle('Daily Reversal Strategies, ' + ticker)
    ax1.plot(dates4fig, negnegCum,  color = prefColor,   alpha = 0.8, label = "Long Reversal Gamma-Timed")
    ax1.plot(dates4fig, negRetCum,      color = "red",     alpha = 0.8, label = "Long Reversal")
    ax1.plot(dates4fig, buyAndHold, color = "silver",   alpha = 1, label = "Buy-and-hold")
    ax1.legend()
    
    ax2.plot(dates4fig, negposCumShort,  color = prefColor, alpha = 0.8, label = "Short Reversal Gamma-Timed")
    ax2.plot(dates4fig, posRetCumShort,  color = "red",     alpha = 0.8, label = "Short Reversal")
    ax2.plot(dates4fig, buyAndHold, color = "silver",   alpha = 1, label = "Buy-and-hold")
    ax2.legend()
    
    #Zscore Reversal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
    fig.suptitle('Z-Score Reversal Strategies, ' + ticker)
    ax1.plot(dates4fig, combinedZscoreCum,  color = prefColor, alpha = 0.8, label = "Long Reversal Gamma-Timed")
    ax1.plot(dates4fig, returnZscoreCum,  color = "red",     alpha = 0.8, label = "Long Reversal")
    ax1.legend()

    ax2.plot(dates4fig, combinedZscoreShortCum,  color = prefColor, alpha = 0.8, label = "Short Reversal Gamma-Timed")
    ax2.plot(dates4fig, returnZscoreShortCum,  color = "red",     alpha = 0.8, label = "Short Reversal")
    ax2.legend(loc = 'upper left')
    
    
    #Daily and Zscore long w/ TC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3))
    fig.suptitle('Long Reversal Strategies w/ TC, c = ' + str(int(c*10000)) + "bps" )
    ax1.plot(dates4fig, negnegCum, linestyle = '-',  color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed")
    ax1.plot(dates4fig, negRetCum, linestyle = '-',  color = "red",     alpha = 0.8, label = "Reversal")   
    ax1.plot(dates4fig, negnegCumTC, linestyle = '--',  color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed w/ TC")
    ax1.plot(dates4fig, negRetCumTC, linestyle = '--',  color = "red",     alpha = 0.8, label = "Reversal w/ TC")
    ax1.legend(loc = 'upper left')
    ax1.set_xlabel('Long Reversal')
    
    ax2.plot(dates4fig, combinedZscoreCum, linestyle = '-', color = prefColor, alpha = 0.8, label = "Z-score Gamma-Timed")
    ax2.plot(dates4fig, returnZscoreCum,  linestyle = '-', color = "red",     alpha = 0.8, label = "Z-score Reversal")
    ax2.plot(dates4fig, combinedZscoreCumTC, linestyle = '--', color = prefColor, alpha = 0.8, label = "Z-score Gamma-Timed w/ TC")
    ax2.plot(dates4fig, returnZscoreCumTC,  linestyle = '--', color = "red",     alpha = 0.8, label = "Z-score Reversal w/ TC")
    ax2.set_xlabel('Z-Score Long Reversal')
    ax1.legend()
    
    
    
    
    plt.figure()
    plt.plot(dates4fig, negnegCum,  color = prefColor,   alpha = 0.8, label = "Negative Return Reversal Gamma-Timed")
    plt.plot(dates4fig, negRetCum,      color = "red",     alpha = 0.8, label = "Negative Return Reversal")
    plt.plot(dates4fig, buyAndHold, color = "silver",   alpha = 1, label = "Buy-and-hold")
    plt.title("Return Reversal Strategies, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend()        
                
    plt.figure()
    plt.plot(dates4fig, negnegCum,  color = prefColor,   alpha = 0.8, label = "Negative Return Reversal Gamma-Timed")
    plt.plot(dates4fig, negRetCum,      color = "red",     alpha = 0.8, label = "Negative Return Reversal")
    plt.plot(dates4fig, negnegCumTC, linestyle = '--', color = prefColor,   alpha = 0.8, label = "Negative Return Reversal Gamma-Timed w/ TC") 
    plt.plot(dates4fig, negRetCumTC, linestyle = '--',  color = "red",   alpha = 0.8, label = "Negative Return Reversal w/ TC")
    #plt.plot(dates4fig[1:], buyAndHold, color = "silver",   alpha = 1, label = "Buy-and-hold")
    plt.title("Return Reversal Strategies, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend()        
    
    
    plt.figure()
    plt.plot(dates4fig, negposCumShort,  color = prefColor, alpha = 0.8, label = "Pos Return Reversal Gamma-Timed")
    plt.plot(dates4fig, posRetCumShort,  color = "red",     alpha = 0.8, label = "Pos Return Reversal")
    plt.plot(dates4fig, buyAndHold, color = "silver",   alpha = 1, label = "Buy-and-hold")
    plt.title("Return Reversal Strategies, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend(loc = 'upper left')        
                
    plt.figure()
    plt.plot(dates4fig, negposCumShort,  color = prefColor, alpha = 0.8, label = "Pos Return Reversal Gamma-Timed")
    plt.plot(dates4fig, posRetCumShort,  color = "red",     alpha = 0.8, label = "Pos Return Reversal")
    plt.plot(dates4fig, negposCumShortTC, linestyle = '--', color = prefColor, alpha = 0.8, label = "Pos Return Reversal Gamma-Timed w/ TC")
    plt.plot(dates4fig, posRetCumShortTC, linestyle = '--', color = "red",     alpha = 0.8, label = "Pos Return Reversal w/ TC")
    plt.title("Return Reversal Strategies, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend(loc = 'upper left')        
                
    plt.figure()
    plt.plot(dates4fig, gammaZscoreCum,  color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed")
    plt.plot(dates4fig, returnZscoreCum,  color = "red",     alpha = 0.8, label = "Reversal")
    plt.plot(dates4fig, combinedZscoreCum, color = "black", alpha = 0.8, label = "Combined Reversal")
    plt.plot(dates4fig, gammaZscoreCumTC,  linestyle = '--',  color = prefColor, alpha = 0.8, label = "Reversal Gamma-Timed w/ TC")
    plt.plot(dates4fig, returnZscoreCumTC,  linestyle = '--',  color = "red",     alpha = 0.8, label = "Reversal w/ TC")
    plt.plot(dates4fig, combinedZscoreCumTC, linestyle = '--', color = "black", alpha = 0.8, label = "Combined Reversal w/ TC")
    plt.title("Return Reversal Z-score Strategy, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend(loc = 'upper left')  
               
    
 
    
                
    #Investigate number of signals 
    nNegNegSignals = np.sum(negnegSignal)
    nNegPosSignals = np.sum(negposSignal)
    nNegRetSignals = np.sum(negRetSignal)
    nPosRetSignals = np.sum(posRetSignal)  
    nNegGammaSignals = np.sum(negGammaSignal)
    
    
    nDays = len(negnegSignal)    
    
    
    print(nNegNegSignals)
    print(nNegRetSignals)
    print(nNegGammaSignals)
    print(nNegPosSignals)
    print(nPosRetSignals)
    
    print(np.mean(turnoverNegNeg))
    print(np.mean(turnoverNegPos))
    print(np.mean(turnoverNegRet))
    print(np.mean(turnoverPosRet))
    
    
    

    rf = rf[lookback + 1:]
    #Compute performance
    negnegPerformance   = bt.ComputePerformance(negnegXsReturns,   rf, 0, 252)
    negnegPerformanceTC = bt.ComputePerformance(negnegXsReturnsTC, rf, 0, 252)
    negRetPerformance   = bt.ComputePerformance(negRetXsReturns,   rf, 0, 252)
    negRetPerformanceTC = bt.ComputePerformance(negRetXsReturnsTC, rf, 0, 252)
    buyAndHoldPerformance = bt.ComputePerformance(xsret[lookback + 1:], rf, 0, 252)
    
    negposShortPerformance   = bt.ComputePerformance(-negposXsReturns,       rf, 0, 252)
    negposShortPerformanceTC = bt.ComputePerformance(negposXsReturnsShortTC, rf, 0, 252)
    negRetShortPerformance   = bt.ComputePerformance(-posRetXsReturns,       rf, 0, 252)
    negRetShortPerformanceTC = bt.ComputePerformance(posRetXsReturnsShortTC, rf, 0, 252)
    
    zscoreRetPerformance       = bt.ComputePerformance(returnZscoreXsReturns, rf, 0, 252)
    zscoreCombPerformance      = bt.ComputePerformance(combinedZscoreXsReturns, rf, 0, 252)
    zscoreShortRetPerformance  = bt.ComputePerformance(returnZscoreShortXsReturns, rf, 0, 252)
    zscoreShortCombPerformance = bt.ComputePerformance(combinedZscoreShortXsReturns, rf, 0, 252)
    
    zscoreRetPerformanceTC      = bt.ComputePerformance(returnZscoreXsReturnsTC, rf, 0, 252)
    zscoreCombPerformanceTC      = bt.ComputePerformance(combinedZscoreXsReturnsTC, rf, 0, 252)
    
    
    
    #Construct Latex Table
    def constructPerformanceDf(performanceList, colNames, DaysIn, to_latex = True):
        legend      = performanceList[0][0]
        nStrategies = len(performanceList)
        performanceMat = np.zeros((len(legend), nStrategies))
        
        for i in np.arange(0, nStrategies):
            performanceMat[:, i]  = performanceList[i][1].reshape(-1,)
                
        performanceMat = np.vstack(performanceMat).astype(np.float)
        performanceMat = np.round(performanceMat, decimals = 4) 
        performanceMat  = np.concatenate((legend.reshape(-1, 1), performanceMat), axis = 1)
        if len(DaysIn) > 0:
            performanceMat = np.concatenate((performanceMat, DaysIn.reshape(1, -1)), axis = 0)
        performanceDf   = pd.DataFrame.from_records(performanceMat, columns = colNames)
    
        if to_latex == True:
            print(performanceDf.to_latex(index=False))
            
        return performanceDf
    
    
    colNames = np.array(["Daily Reversal", "Long GT", "Long", "Short GT", "Short", "Buy-and-hold"])
    perfList = [negnegPerformance, negRetPerformance, negposShortPerformance, negRetShortPerformance, buyAndHoldPerformance]
    DaysIn   = np.array([ "Days In", np.sum(negnegSignal[lookback:]), np.sum(negRetSignal[lookback:]), np.sum(negposSignal[lookback:]),  np.sum(posRetSignal[lookback:]), len(xsret[lookback:])])
    test     = constructPerformanceDf(perfList, colNames = colNames, DaysIn = DaysIn, to_latex = True)
    

    colNames = np.array(["w/ TC", "Daily GT", "Daily", "Z-Score GT", "Z-Score"])
    perfList = [negnegPerformanceTC, negRetPerformanceTC, zscoreCombPerformanceTC, zscoreRetPerformanceTC]
    test     = constructPerformanceDf(perfList, colNames = colNames, DaysIn = [], to_latex = True)


    colNames = np.array(["Z-score", "Long GT", "Long", "Short GT", "Short"])
    perfList = [zscoreCombPerformance, zscoreRetPerformance, zscoreShortCombPerformance, zscoreShortRetPerformance,]
    DaysIn   = np.array([ "Days In", np.sum(combinedZscoreSignal), np.sum(returnZscoreSignal), np.sum(combinedZscoreShortSignal),  np.sum(returnZscoreShortSignal)])
    test     = constructPerformanceDf(perfList, colNames = colNames, DaysIn = DaysIn, to_latex = True)

    
    
    