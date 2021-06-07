# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:13:19 2021

@author: ekblo
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:54:25 2021

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
import seaborn as sn
import sys
from sklearn.linear_model import LassoCV
#Regressions

### SET IMPORT PARAMS ####################################################################
UnderlyingAssetName   = ["SPX Index", "SPY US Equity", "NDX Index", "QQQ US Equity", "RUT Index", "IWM US Equity"]
UnderlyingTicker      = ["SPX", "SPY", "NDX", "QQQ", "RUT", "IWM"]
volIndexTicker        = ["VIX Index", "VIX Index", "VXN Index", "VXN Index", "RVX Index", "RVX Index"]


IndexTickers = ["SPX", "NDX", "RUT"]
ETFTickers   = ["SPY", "QQQ", "IWM"]
volIndexTickers = ["VIX Index", "VXN Index", "RVX Index"]
ETFMultipliers = [10, 40, 10]

IsEquityIndex        = [True, False, True, False, True, False, False]
#UnderlyingAssetName   = ["SPX US Equity"]
#UnderlyingTicker      = ["SPX"]
#IsEquityIndex         = [True]

loadloc               = "../Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################


def computeRfDaily(data):
        dates            = data["Dates"].to_numpy()
        dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
        daycount         = bt.dayCount(dates4fig)
        
        Rf               = data["LIBOR"].to_numpy() / 100
        RfDaily          = np.zeros((np.size(Rf, 0), ))
        RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
        return RfDaily
    
#Load Data
AggregateData = []
for ticker in UnderlyingTicker:
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    AggregateData.append(data)


AggregateGammaList       = []
AggregateGammaListScaled = []
IndexGammaList           = []
ETFGammaList             = []
XsReturnsList            = []
VolIndexList             = []

for i in np.arange(0, len(IndexTickers)):
    #Store index and ETF data separately for simplicity
    indexticker = IndexTickers[i]
    etfticker   = ETFTickers[i]
    volindexticker = volIndexTickers[i]
    etfmultiplier = ETFMultipliers[i]
    
    IndexData   = pd.read_csv(loadloc + indexticker + "AggregateData.csv")
    ETFData     = pd.read_csv(loadloc + etfticker + "AggregateData.csv")
    
    #Store Vol Index
    volIndex = ETFData[volindexticker].to_numpy()
    volIndex = volIndex[np.isfinite(volIndex)]
    VolIndexList.append(volIndex)
    VolIndexList.append(volIndex)
    
    #Compute aggregata net gamma 
    IndexGamma  = IndexData["netGamma"].to_numpy()
    ETFGamma    = ETFData["netGamma"].to_numpy()
    ETFGamma    = ETFGamma / etfmultiplier**2
    MarketCap   = IndexData["Market Cap"].to_numpy()
    
    if len(volIndex) < len(ETFGamma): #trim to vol index
        trim = len(volIndex)
    else:
        trim = len(ETFGamma)
    
    AggregateGamma = IndexGamma[-trim:] + ETFGamma[-trim:]
    IndexGamma     = IndexGamma[-trim:]
    MarketCap      = MarketCap[-trim:]
    AggregateGammaScaled = AggregateGamma / MarketCap
    
    AggregateGammaList.append(AggregateGamma)
    AggregateGammaList.append(AggregateGamma)
    AggregateGammaListScaled.append(AggregateGammaScaled)
    AggregateGammaListScaled.append(AggregateGammaScaled)
    #IndexGammaList.append(IndexGamma)
    #ETFGammaList.append(ETFGamma)
    
    #Compute Excess Returns
    IndexPrices = IndexData["TR Index"].to_numpy()
    ETFPrices   = ETFData[etfticker].to_numpy()
    RfDaily     = computeRfDaily(ETFData) 
    RfDaily     = RfDaily[-trim:]
    RfDaily[0]  = 0
    
    IndexReturns    = bt.computeReturns(IndexPrices)
    IndexReturns    = IndexReturns[-trim:] #trim
    IndexReturns[0] = 0
    
    ETFReturns    = bt.computeReturns(ETFPrices)
    ETFReturns    = ETFReturns[-trim:]
    ETFReturns[0] = 0
    
    XsReturnsList.append(IndexReturns - RfDaily)
    XsReturnsList.append(ETFReturns - RfDaily)
    

    
#Run regressions
lag = 1
for i in np.arange(0, len(XsReturnsList)):
    ticker = UnderlyingTicker[i]
    aggGammaScaled  = AggregateGammaListScaled[i] #grab gamma
    xsret           = XsReturnsList[i]            #grab returns
    volIndex              = VolIndexList[i]             #grab vol
    
    #standard regression
    X = np.concatenate((aggGammaScaled.reshape(-1,1), volIndex.reshape(-1,1)), axis = 1)
    
    ###Control Regression###
    #Lagged values
    volIndex1      = volIndex[2:].reshape(-1,1)
    volIndex2      = volIndex[1:-1].reshape(-1,1)
    volIndex3      = volIndex[0:-2].reshape(-1,1)
   
    AbsXsReturns  = np.abs(xsret)
    AbsXsRet1     = AbsXsReturns[2:].reshape(-1,1)
    AbsXsRet2     = AbsXsReturns[1:-1].reshape(-1,1)
    AbsXsRet3     = AbsXsReturns[0:-2].reshape(-1,1)
   
    #Contatenate laged values to control matrix
    X_control = np.concatenate((volIndex2, volIndex3, AbsXsRet1, AbsXsRet2, AbsXsRet3), axis = 1)
    X_control = np.concatenate((X[2:], X_control), axis = 1)
    y_control = np.abs(xsret[2:])
    

    #Run regressions                   
    y      = np.abs(xsret[lag:])
    nObs   = len(y)
   
    X      = X[0:-lag, :]       #Lag matrix accordingly 
    X      = sm.add_constant(X) #add constant

    reg       = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds = {'maxlags':0})
    coefs     = np.round(reg.params*100, decimals = 4) #Multiply coefs by 100 to get in bps format
    tvals     = np.round(reg.tvalues, decimals = 4)
    pvals     = np.round(reg.pvalues, decimals = 4)
    r_squared = np.round(reg.rsquared, decimals = 4)
        
       
    ### Result Print
    legend = np.array(['$\Gamma^{MM}_{t - ' + str(lag) + ', Agg}$', " ", '$IV_{t-1}$', " ", 'Intercept', " ", '$R^2$' ])
    
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
    if i == 0:
        resultsDf["Lag = " + str(lag) + " day"] = legend
        resultsDf[ticker] = results
        allresDf = resultsDf
    else:
        resultsDf[ticker] = results
     
    if i > 0:
        allresDf = pd.concat((allresDf, resultsDf), axis = 1)

    
    #######################################
    ######### Control Regression ##########
    
    y_control = y_control[lag:]
    X_control = X_control[0:-lag, :]       #Lag matrix accordingly 
    X_control = sm.add_constant(X_control) #add constant

    reg_control       = sm.OLS(y_control, X_control).fit(cov_type = "HAC", cov_kwds = {'maxlags':0})
    #netGamma_coef     = np.roun(reg_control.params[1]*100, decimals = 3) 
    coefs_control     = np.round(reg_control.params*100, decimals = 4)   #Multiply net gamma coef by 100 to get bps format
    tvals_control     = np.round(reg_control.tvalues, decimals = 4)
    pvals_control     = np.round(reg_control.pvalues, decimals = 4)
    r_squared_control = np.round(reg_control.rsquared, decimals = 4)
    r_squared_control_adj = np.round(reg_control.rsquared_adj, decimals = 4)
    
    ### Alternative Result Print
    legend_control = np.array(['$\Gamma^{MM}_{t-' + str(lag) + ', agg}$', " ", '$IV_{t-1}$', " ", '$IV_{t-2}$', " ", '$IV_{t-3}$', " ", \
                       '$|R_{t-1}|$', " ", '$|R_{t-2}|$', " ", '$|R_{t-3}|$', " ",  'Intercept', " ", '$R^2$', '$R^2_{adj}$' ])
        
    sign_test_control = []
    for pval in pvals_control:
        if pval < 0.01:
            sign_test_control.append("***")
        elif pval < 0.05:
            sign_test_control.append("**")
        elif pval < 0.1:
            sign_test_control.append("*")
        else:
            sign_test_control.append("")

    results_control = np.array([ str(coefs_control[1]) + sign_test_control[1], "(" + str(tvals_control[1]) + ")", \
                                 str(coefs_control[2]) + sign_test_control[2], "(" + str(tvals_control[2]) + ")", \
                                 str(coefs_control[3]) + sign_test_control[3], "(" + str(tvals_control[3]) + ")", \
                                 str(coefs_control[4]) + sign_test_control[4], "(" + str(tvals_control[4]) + ")", \
                                 str(coefs_control[5]) + sign_test_control[5], "(" + str(tvals_control[5]) + ")", \
                                 str(coefs_control[6]) + sign_test_control[6], "(" + str(tvals_control[6]) + ")", \
                                 str(coefs_control[7]) + sign_test_control[7], "(" + str(tvals_control[7]) + ")", \
                                 str(coefs_control[0]) + sign_test_control[0], "(" + str(tvals_control[0]) + ")", r_squared_control, r_squared_control_adj])        
        
        
    results_controlDf = pd.DataFrame()
    if i == 0:
        results_controlDf["Lag = " + str(lag) + " day"] = legend_control
        results_controlDf[ticker] = results_control
        allres_controlDf = results_controlDf
    else:
        results_controlDf[ticker] = results_control
   
   
    if i > 0 :
        allres_controlDf = pd.concat((allres_controlDf, results_controlDf), axis = 1)
        
    


print(allresDf.to_latex(index=False, escape = False)) 
print(allres_controlDf.to_latex(index=False, escape = False)) 







