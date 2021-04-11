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

loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
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
    return RfDaily, dates4fig


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
   
    #Compute returns
    if ticker  != "VIX":
        #Use Total Return Index for index, and normal price for ETF
        if eqIndexFlag == True:
            price = data["TR Index"].to_numpy()   
        else:
            price    = data[ticker].to_numpy()
           
        rf, dates4fig = computeRfDaily(data)    
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
     
            
    #Signal Generation
    negNetGamma    = (netGamma_scaled < 0)*1    
    posNetGamma    = (netGamma_scaled > 0)*1
    posReturns     = (xsret > 0)*1
    negReturns     = (xsret < 0)*1
    
    #Combined Signals
    negnegSignal  = negNetGamma * negReturns
    negposSignal  = negNetGamma * posReturns
    posnegSignal  = posNetGamma * negReturns    
    posposSignal  = posNetGamma * posReturns
    
    #Single Signal Strategy Returns
    negGammaXsReturns = negNetGamma[0:-1] * xsret[1:]
    posGammaXsReturns = posNetGamma[0:-1] * xsret[1:]
    negRetXsReturns   = negReturns[0:-1] * xsret[1:]
    posRetXsReturns   = posReturns[0:-1] * xsret[1:]
    
    #Combined Signal Returns
    negnegXsReturns = negnegSignal[0:-1] * xsret[1:]
    negposXsReturns = negposSignal[0:-1] * xsret[1:]
    posnegXsReturns = posnegSignal[0:-1] * xsret[1:]
    posposXsReturns = posposSignal[0:-1] * xsret[1:]
    
    #Long Short Strategies
    negGammaReversalXsReturns = negnegXsReturns - negposXsReturns #long after negative, short after positive return (and negative Gamma)
    posGammaReversalXsReturns = posnegXsReturns - posposXsReturns #long after negative, short after positive return (and positive Gamma)
    returnReversalsXsReturns  = negRetXsReturns - posRetXsReturns #Long after negative, short after positive return (unconditional on gamma) 
    
    #Cumulative Excess Returns
    #Signle Signal Strategies
    negNetGammaCum     = np.cumprod(1 + negGammaXsReturns)
    posNetGammaCum     = np.cumprod(1 + posGammaXsReturns)
    negRetCum          = np.cumprod(1 + negRetXsReturns)
    posRetCum          = np.cumprod(1 + posRetXsReturns)
    buyAndHold         = np.cumprod(1 + xsret[1:])
    
    #Combined Signal Long Only Strategies
    negnegCum = np.cumprod(1 + negnegXsReturns)
    negposCum = np.cumprod(1 + negposXsReturns)
    posnegCum = np.cumprod(1 + posnegXsReturns)
    posposCum = np.cumprod(1 + posposXsReturns)
    
    #Long Short Strategies
    negGammaReversalCum = np.cumprod(1 + negGammaReversalXsReturns)
    posGammaReversalCum = np.cumprod(1 + posGammaReversalXsReturns)
    returnReversalsCum  = np.cumprod(1 + returnReversalsXsReturns)
    
    ######################
    #Plot Equity Lines
    
    #Single Signal Strategies
    plt.figure()
    plt.plot(dates4fig[1:], negNetGammaCum, color = "red",       alpha = 0.8, label = "Long After Negative Net Gamma")
    plt.plot(dates4fig[1:], posNetGammaCum, color = prefColor,   alpha = 0.8, label = "Long After Positive Net Gamma")
    plt.plot(dates4fig[1:], negRetCum,      color = "black",     alpha = 0.8, label = "Long After Negative Return")
    plt.plot(dates4fig[1:], posRetCum,      color = "lightblue", alpha = 0.8, label = "Long After Positive Return")
    plt.plot(dates4fig[1:], buyAndHold,     color = "lightgrey", alpha = 0.8, label = "Buy-and-Hold")
    plt.title("Single Signal Long-Only Timing Strategies, " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend()
    
    #Combined Signal Strategies
    plt.figure()
    plt.plot(dates4fig[1:], negnegCum,  color = prefColor,   alpha = 0.8, label = "Neg-Neg")
    plt.plot(dates4fig[1:], negposCum,  color = "red",       alpha = 0.8, label = "Neg-Pos")
    plt.plot(dates4fig[1:], posnegCum,  color = "lightblue", alpha = 0.8, label = "Pos-Neg")
    plt.plot(dates4fig[1:], posposCum,  color = "black",     alpha = 0.8, label = "Pos-Pos")
    plt.plot(dates4fig[1:], buyAndHold, color = "lightgrey", alpha = 0.8, label = "Buy-and-Hold")
    plt.title("Long-Only Strategies Day After \n Given Return and Gamma Combination,  " + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend()


    #Long Short Strategies
    plt.figure()
    plt.plot(dates4fig[1:], negGammaReversalCum,  color = prefColor,   alpha = 0.8, label = "Neg Gamma Reversals")
    plt.plot(dates4fig[1:], posGammaReversalCum,  color = "red",       alpha = 0.8, label = "Pos Gamma Reversals")
    plt.plot(dates4fig[1:], returnReversalsCum,   color = "black",     alpha = 0.8, label = "Uncond. Return Reversals")
    plt.title("Return Reversal Strategies," + ticker)
    plt.ylabel("Cumulative Excess Returns")
    plt.legend()

            
                
                
                
                
                
        





    
    
    
        
        
    
    
    
    