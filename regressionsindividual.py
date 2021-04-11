# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:31:00 2021

@author: ekblo
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:10:36 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import Backtest as bt
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import sys
#Regressions

### SET IMPORT PARAMS ####################################################################
UnderlyingAssetName   = ["SPX Index"]
UnderlyingTicker      = ["SPX"]
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData = []
for ticker in UnderlyingTicker:
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    AggregateData.append(data)


#Compute returns and adjusted gamma
nAssets = len(UnderlyingTicker)
TotalReturns = []
XsReturns    = []
for i in np.arange(0, nAssets):
    data   = AggregateData[i]
    ticker = UnderlyingTicker[i]
    rf     = data["LIBOR"].to_numpy()
    
    def computeRfDaily(data):
        dates            = data["Dates"].to_numpy()
        dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
        daycount         = bt.dayCount(dates4fig)
        
        Rf               = data["LIBOR"].to_numpy() / 100
        RfDaily          = np.zeros((np.size(Rf, 0), ))
        RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
        return RfDaily
    
    if ticker  != "VIX":
       price    = data[ticker].to_numpy()
       rf       = computeRfDaily(data)
       
       #Compute returns 
       ret   = price[1:] / price[0:-1] - 1 
       ret   = np.concatenate((np.zeros((1, )), ret), axis = 0)
       xsret = ret - rf
       
       #Store
       TotalReturns.append(ret)
       XsReturns.append(ret)
    
    else:
       frontPrices    = data["frontPrices"].to_numpy()
       rf = computeRfDaily(data)
       
       frontXsReturns = frontPrices[1:] / frontPrices[0:-1] - 1
       frontXsReturns =  np.concatenate((np.zeros((1, )), frontXsReturns), axis = 0)
       frontTotalReturns = frontXsReturns + rf
       
       #Store
       TotalReturns.append(frontTotalReturns)
       XsReturns.append(frontXsReturns)
    
     
def plotResiduals(residuals, lag = None, ticker = None, color = '#0504aa', histtype = 'stepfilled'):
    skewness   = skew(residuals)
    xsKurtosis = kurtosis(residuals) - 3
    mu         = np.mean(residuals)
    sigma      = np.std(residuals)

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x = residuals, bins='auto', color=color, histtype = histtype)
    fig.suptitle('Regression Residuals Histogram, lag = ' + str(lag) + ", Asset = " + ticker)
    

    textstr = '\n'.join((
                r'$\mu=%.2f$' % (mu, ),
                r'$\sigma=%.2f$' % (sigma, ),
                '$\mathrm{skewness}=%.2f$' % (skewness, ),
                '$\mathrm{xs kurtosis}=%.2f$' % (xsKurtosis, )
                ))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor=prefColor, alpha=0.2)

    # place a text box in upper left in axes coords
    ax.text(0.60, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.show() 
    
    
# #Set up Regressions
lagList = [1, 2, 5, 10]
#lagList = [1]
nRegs   = len(lagList)
hist    = True
scatter = True
RegResultList = []

for j in np.arange(0, len(UnderlyingTicker)):
    ticker   = UnderlyingTicker[j]
    data     = AggregateData[j]
    xsret    = XsReturns[j]
    netGamma = data["netGamma"].to_numpy()
    netGamma_norm = (netGamma - np.mean(netGamma)) / np.std(netGamma)
    
    MAdollarVolume = data["MADollarVolume"].to_numpy() 
    adjNetGamma    = netGamma / MAdollarVolume 
    adjNetGamma_norm = (adjNetGamma - np.mean(adjNetGamma)) / np.std(adjNetGamma)
    
    regResults = []
    
    for i in np.arange(0, nRegs):
        lag    = lagList[i]
        y      = np.abs(xsret[lag:])
        nObs   = np.size(y)
        
        X_raw  = adjNetGamma_norm[0:-lag]
        X_sqrd = adjNetGamma_norm[0:-lag]**2
        X      = np.concatenate((X_raw.reshape(nObs, 1), X_sqrd.reshape(nObs, 1)), axis = 1)
        X      = sm.add_constant(X)
    
        reg       = sm.OLS(y, X).fit()
        coefs     = np.round(reg.params, decimals = 5)
        tvals     = np.round(reg.tvalues, decimals = 5)
        pvals     = np.round(reg.pvalues, decimals = 5)
        r_squared = np.round(reg.rsquared, decimals = 5)
        
        legend  = np.array(["statistic", "coefficient", "t stats", "p-values", "r_squared"])
        alpha   = np.array(["alpha", coefs[0], tvals[0], pvals[0], r_squared])
        b1      = np.array(["b1", coefs[1], tvals[1], pvals[1], " "])
        b2      = np.array(["b2", coefs[2], tvals[2], pvals[2], " "])
        
        lag_df = pd.DataFrame()
        
        if i == 0:
            lag_df["Statistic(Lag = " + str(lag) + ")"] = legend
            lag_df["alpha (Lag = " + str(lag) + ")"]  = alpha
            lag_df["B1 (Lag = " + str(lag) + ")"] = b1
            lag_df["B2 (Lag = " + str(lag) + ")"] = b2 
        else:
            lag_df["alpha (Lag = " + str(lag) + ")"] = alpha
            lag_df["B1 (Lag = " + str(lag) + ")"] = b1 
            lag_df["B2 (Lag = " + str(lag) + ")"] = b2 
        
        regResults.append(lag_df)
        
        if scatter == True:
            x        = np.linspace(np.min(X_raw), np.max(X_raw), np.size(X_raw))
            reg_line = coefs[0] + coefs[1] * x + coefs[2]*x**2
            
            fig, ax = plt.subplots()
            
            ax.scatter(X_raw, y, color = '#0504aa', s = 0.5)
            ax.plot(x, reg_line, color = "red", alpha = 0.5)
            fig.suptitle("Absolute Returns vs Net Gamma Exposure (normalized)")
            ax.set_xlabel("Net Gamma Exposure (Normalized)")
            ax.set_ylabel("Daily Absolute Returns")
            
            textstr = '\n'.join((
                'Asset = ' + ticker,
                'Lag = ' + str(lag) + ' days',
                r'$\alpha=%.5f$' % (coefs[0], ),
                r'$\beta_1=%.5f$' % (coefs[1], ),
                r'$\beta_2=%.5f$' % (coefs[2], ),
                ))

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor=prefColor, alpha=0.2)

            # place a text box in upper left in axes coords
            ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            plt.show() 
            
            
        residuals  = reg.resid
        if hist == True: #Plot netGamma histogram
            plot = plotResiduals(residuals, lag = lag, ticker = ticker)
    
    #Concatenate
    AssetDf = pd.concat(regResults, axis = 1)
    print(AssetDf)
    
    RegResultList.append(AssetDf)
    
    
# X   = netGamma[0:-lag]
# X   = sm.add_constant(X)
# y   = np.abs(Returns[lag:])

# regression = sm.OLS(y, X).fit()
# print(regression.summary())

