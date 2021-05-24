# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:16:37 2021

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
UnderlyingAssetName   = ["SPX Index", "SPY US Equity", "NDX Index", "QQQ US Equity", "RUT Index", "IWM US Equity"]
UnderlyingTicker      = ["SPX", "SPY", "NDX", "QQQ", "RUT", "IWM"]
volIndexTicker        = ["VIX Index", "VIX Index", "VXN Index", "VXN Index", "RVX Index", "RVX Index"]
IndexTicker = ["SPX", "NDX", "RUT"]
ETFTicker   = ["SPY", "QQQ", "IWM"]


IsEquityIndex        = [True, False, True, False, True, False, False]
#UnderlyingAssetName   = ["SPX US Equity"]
#UnderlyingTicker      = ["SPX"]
#IsEquityIndex         = [True]

loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData = []
netGammaDf         = pd.DataFrame()
netGammaScaledDf   = pd.DataFrame()
netGammaAdjustedDf = pd.DataFrame()
totalReturnsDf     = pd.DataFrame()
xsReturnsDf        = pd.DataFrame()
volIndexDf         = pd.DataFrame()

Multiplier = [1, 10, 1, 40, 1, 10]
MarketCapTicker = ["SPX", "SPX", "NDX", "NDX", "RUT", "RUT"]

 
def computeRfDaily(data):
    dates            = data["Dates"].to_numpy()
    dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
    daycount         = bt.dayCount(dates4fig)
    
    Rf               = data["LIBOR"].to_numpy() / 100
    RfDaily          = np.zeros((np.size(Rf, 0), ))
    RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
    return RfDaily


for i in np.arange(0, len(UnderlyingTicker)):
    ticker       = UnderlyingTicker[i]
    mktcapticker = MarketCapTicker[i]
    M            = Multiplier[i]
    index_flag   = IsEquityIndex[i]
    
    data          = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    marketcapdata = pd.read_csv(loadloc + mktcapticker + "AggregateData.csv")
    
    startDate = 20060101
    endDate   = 20191231
    
    #Trim to dates
    data          = bt.trimToDates(data, data["Dates"], startDate, endDate)
    marketcapdata = bt.trimToDates(marketcapdata, marketcapdata["Dates"], startDate, endDate)
    
    #Grab needed data
    netGamma  = data["netGamma"].to_numpy()
    marketCap = marketcapdata["Market Cap"].to_numpy()
    
    
    if index_flag == True:
        prices = data["TR Index"].to_numpy()
    else:
        prices = data[ticker].to_numpy()
   
    rfdaily = computeRfDaily(data)
    
    #Compute returns
    totalReturns = bt.computeReturns(prices)
    xsReturns    = totalReturns - rfdaily
    
    #Store net gamma, net gamma scaled, and net gamma adjusted
    netGammaDf[ticker]         = netGamma
    netGammaScaledDf[ticker]   = netGamma / marketCap
    netGammaAdjustedDf[ticker] = (netGamma / marketCap) / M**2
       
    #Store returns
    totalReturnsDf[ticker] = totalReturns
    xsReturnsDf[ticker] = xsReturns
 
    AggregateData.append(data)

gammaChanges = netGammaDf.diff()
gammaChangesCorr = gammaChanges.iloc[1:, :].corr()


#Correlation Matrices
corrMatrixGamma = netGammaAdjustedDf.corr()
corrMatrixReturns = xsReturnsDf.corr()

plt.figure()
sn.heatmap(corrMatrixGamma, annot = True)
plt.title("Market Maker Net Gamma Correlation")

plt.figure()
sn.heatmap(corrMatrixReturns, annot = True)
plt.title("Return Correlation")

plt.figure()
sn.heatmap(gammaChangesCorr, annot = True)
plt.title("Changes in Net Gamma Correlation")

totalGamma = netGammaAdjustedDf.to_numpy()
totalGamma = np.sum(totalGamma, axis = 1)

# #Set up Regressions
#lagList = [1, 2, 5, 10]
lag     = 1
hist    = False
scatter = False

for j in np.arange(0, len(UnderlyingTicker)):
    ticker    = UnderlyingTicker[j]
    volticker = volIndexTicker[j]
    name      = UnderlyingAssetName[j]
    data      = AggregateData[j]
    xsret     = xsReturnsDf.iloc[:, j].to_numpy()
       
    
    #Compute Independent Variable Time Series
    #Grab necessary data
    volIndex       = data[volticker].to_numpy()
    
    
    
    #Concatenate Independent variables to X matrix
    X = np.concatenate((totalGamma.reshape(-1,1), volIndex.reshape(-1,1)), axis = 1)
    
    
    
    ####################################################
    ######### Fit regressions and store result #########
    
    #######################################
    ######### Standard Regression #########    
    y      = np.abs(xsret[lag:])
    nObs   = np.size(y)
   
    X      = X[0:-lag, :]       #Lag matrix accordingly 
    X      = sm.add_constant(X) #add constant

    reg       = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds = {'maxlags':0})
    coefs     = np.round(reg.params*100, decimals = 4) #Multiply coefs by 100 to get in bps format
    tvals     = np.round(reg.tvalues, decimals = 4)
    pvals     = np.round(reg.pvalues, decimals = 4)
    r_squared = np.round(reg.rsquared, decimals = 4)
        
        
       
    ### Result Print
    legend = np.array(['$\Gamma^{MM}_{t - ' + str(lag) + '}$', " ", '$IV_{t-1}$', " ", 'Intercept', " ", '$R^2$' ])
    
    sign_test = []
    for pval in pvals:
        if pval < 0.01:
            sign_test.append("***")
        elif pval < 0.05:
            sign_test.append("**")
        elif pval < 0.1:
            sign_test.append("*")
        else:
            sign_test.append("")
                
    results = np.array([ str(coefs[1]) + sign_test[1], "(" + str(tvals[1]) + ")", \
                         str(coefs[2]) + sign_test[2], "(" + str(tvals[2]) + ")", \
                         str(coefs[0]) + sign_test[0], "(" + str(tvals[0]) + ")", r_squared])        
    
    resultsDf = pd.DataFrame()
    if j == 0:
        resultsDf["Lag = " + str(lag) + " day"] = legend
        resultsDf[ticker] = results
        allresDf = resultsDf
    else:
        resultsDf[ticker] = results
     
    if j > 0:
        allresDf = pd.concat((allresDf, resultsDf), axis = 1)

print(allresDf.to_latex(index=False, escape = False)) 


