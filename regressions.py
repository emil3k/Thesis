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
UnderlyingAssetName   = ["SPX Index", "SPY US Equity", "VIX Index"]
UnderlyingTicker      = ["SPX", "SPY", "VIX"]
loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData = []
for ticker in UnderlyingTicker:
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    AggregateData.append(data)


#Compute returns
nAssets = len(UnderlyingTicker)
TotalReturns = []
XsReturns    = []
for i in np.arange(0, nAssets):
    data   = AggregateData[i]
    ticker = UnderlyingTicker[i]
    
    if ticker  != "VIX":
       price  = data[ticker].to_numpy()
       rf     = data["Rf Daily"].to_numpy() #grab Rf    
       
       #Compute returns 
       ret   = price[1:] / price[0:-1] - 1 
       ret   = np.concatenate((np.zeros((1, )), ret), axis = 0)
       xsret = ret - rf
       
       #Store
       TotalReturns.append(ret)
       XsReturns.append(ret)
    
    else:
       frontPrices    = data["frontPrices"].to_numpy()
       rf             = data["Rf Daily"].to_numpy() #grab Rf    
       
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


for j in np.arange(0, 1):
    ticker   = UnderlyingTicker[j]
    data     = AggregateData[j]
    xsret    = XsReturns[j]
    netGamma = data["netGamma"].to_numpy()
    
    regResults = []
    
    for i in np.arange(0, nRegs):
        lag = lagList[i]
        y   = np.abs(xsret[lag:])
        X   = netGamma[0:-lag]
        X   = sm.add_constant(X)
    
        reg       = sm.OLS(y, X).fit()
        coefs     = np.round(reg.params, decimals = 3)
        tvals     = np.round(reg.tvalues, decimals = 3)
        pvals     = np.round(reg.pvalues, decimals = 3)
        r_squared = np.round(reg.rsquared, decimals = 3)
        
        legend  = np.array(["coefficient", "t stats", "p-values", "r_squared"])
        alpha   = np.array([coefs[0], tvals[0], pvals[0], r_squared])
        beta    = np.array([coefs[1], tvals[1], pvals[1], " "])
        
        reg_df = pd.DataFrame()
        reg_df["Statistic"] = legend
        reg_df["alpha"] = alpha
        reg_df["beta"] = beta
        
        regResults.append(reg_df)
        
        print(reg_df)
        
        residuals  = reg.resid
        if hist == True: #Plot netGamma histogram
            plot = plotResiduals(residuals, lag = lag, ticker = ticker)
    
    
    
# X   = netGamma[0:-lag]
# X   = sm.add_constant(X)
# y   = np.abs(Returns[lag:])

# regression = sm.OLS(y, X).fit()
# print(regression.summary())

