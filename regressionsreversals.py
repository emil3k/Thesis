# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:39:46 2021

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
UnderlyingAssetName   = ["SPX Index", "SPY US Equity", "NDX Index", "QQQ US Equity", "RUT Index", "IWM US Equity", "VIX Index"]
UnderlyingTicker      = ["SPX", "SPY", "NDX", "QQQ", "RUT", "IWM", "VIX"]

IndexTicker = ["SPX", "NDX", "RUT"]
ETFTicker   = ["SPY", "QQQ", "IWM"]


IsEquityIndex        = [True, False, True, False, True, False, False]
#UnderlyingAssetName   = ["SPX US Equity"]
#UnderlyingTicker      = ["SPX"]
#IsEquityIndex         = [True]

loadloc               = "../Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################

#Load Data
AggregateData = []
for ticker in UnderlyingTicker:    
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    
    #trim data
    startDate = 20060101
    endDate   = 20200101
    data  = bt.trimToDates(data, data["Dates"], startDate, endDate)
    AggregateData.append(data)
    
def computeRfDaily(data):
       dates            = data["Dates"].to_numpy()
       dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
       daycount         = bt.dayCount(dates4fig)
       
       Rf               = data["LIBOR"].to_numpy() / 100
       RfDaily          = np.zeros((np.size(Rf, 0), ))
       RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
       return RfDaily

#Compute returns and adjusted gamma
nAssets = len(UnderlyingTicker)
TotalReturns = []
XsReturns    = []
for i in np.arange(0, nAssets):
    data   = AggregateData[i]
    ticker = UnderlyingTicker[i]
    eqIndexFlag  = IsEquityIndex[i]
    rf       = computeRfDaily(data)    
    
    if ticker  != "VIX":
        #Use Total Return Index for index, and normal price for ETF
        if eqIndexFlag == True:
            price = data["TR Index"].to_numpy()   
        else:
            price    = data[ticker].to_numpy()
           
        #Compute returns 
        ret   = price[1:] / price[0:-1] - 1 
        ret   = np.concatenate((np.zeros((1, )), ret), axis = 0)
        xsret = ret - rf
        
        #Store
        TotalReturns.append(ret)
        XsReturns.append(xsret)

    #Use futures for VIX Returns
    else:
       frontPrices    = data["frontPrices"].to_numpy()
             
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
#lagList = [1, 2, 5, 10]
lag     = 1
hist    = False
scatter = False
ETFMultiplierList = np.array([10,40,10])
for j in np.arange(0, len(UnderlyingTicker)):
    ticker   = UnderlyingTicker[j]
    name     = UnderlyingAssetName[j]
    data     = AggregateData[j]
    xsret    = XsReturns[j]
    isETF    = np.in1d(ETFTicker, ticker)
    
    if np.sum(isETF) == 1: #if ticker is an ETF
       ETFMultiplier  = ETFMultiplierList[isETF]   
       indexData      = AggregateData[j-1] #grab corresponding Index data
       marketCap      = indexData["Market Cap"].to_numpy()
       marketCap      = marketCap[-len(data):]
       marketCap      = (ETFMultiplier**2)*marketCap
    else:
       marketCap      = data["Market Cap"].to_numpy()
    
    
    #Compute Independent Variable Time Series
    #Grab necessary data
    netGamma       = data["netGamma"].to_numpy() 
    IVOL           = data["IVOL"].to_numpy()*100
    
    
    netGamma_scaled = netGamma / marketCap
     
  
    #Separate data by month
    dates = data["Dates"].to_numpy() #grab dates
    firstDayList, lastDayList = bt.GetFirstAndLastDayInPeriod(dates, nDigits = 2) #get first and last day indices
     
    #Grab and save returns and gamma by month
    returnsByMonth = []
    gammaByMonth = []
    for i in np.arange(1, len(firstDayList)):
        start = firstDayList[i]
        stop  = lastDayList[i]
        
        returnsByMonth.append(xsret[start:stop + 1])
        gammaByMonth.append(netGamma_scaled[start:stop + 1])
   
    
    #run monthly reversal regressions and store monthly average gamma
    nMonths         = len(returnsByMonth)
    reversalCoefs   = np.zeros((nMonths,))
    avgMonthlyGammaVec = np.zeros((nMonths,))
    
    for i in np.arange(0, nMonths):
        dailyXsReturns = returnsByMonth[i]
        dailyGamma     = gammaByMonth[i]
              
        avgMonthlyGamma = np.mean(dailyGamma) 
        avgMonthlyGammaVec[i] = avgMonthlyGamma
  
        #Set up regression
        y = np.abs(dailyXsReturns[lag:])
        X = np.abs(dailyXsReturns[0:-lag])    
        X = sm.add_constant(X)
        
        reg = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds = {'maxlags':0})
        reversalCoefs[i] =  reg.params[1] #Store coefs
        
    
    #run regression of reversal coefficients on monthly average gamma
    y_rev = reversalCoefs
    X_rev = avgMonthlyGammaVec
    X_rev = sm.add_constant(X_rev)    
    
    reg_rev = sm.OLS(y_rev, X_rev).fit(cov_type = "HAC", cov_kwds = {'maxlags':0})
    
    
    coefs     = np.round(reg_rev.params, decimals = 4) #Multiply coefs by 100 to get in bps format
    tvals     = np.round(reg_rev.tvalues, decimals = 4)
    pvals     = np.round(reg_rev.pvalues, decimals = 4)
    r_squared = np.round(reg_rev.rsquared, decimals = 4)
        
         
    
    ### Result Print
    legend = np.array(['$\bar{\Gamma}^{MM}_{t}$', " ", 'Intercept', " ", '$R^2$' ])
    
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
                         str(coefs[0]) + sign_test[0], "(" + str(tvals[0]) + ")", r_squared])        
    
    print(coefs)
    resultsDf = pd.DataFrame()
    if j == 0:
        resultsDf["Lag = " + str(lag) + " day"] = legend
        resultsDf[ticker] = results
        allresDf = resultsDf
    else:
        resultsDf[ticker] = results
     
    if j > 0:
        allresDf = pd.concat((allresDf, resultsDf), axis = 1)

   
   
    
    ########################################

   
  
   
    if scatter == True:
        #x        = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), np.size(X[:, 1]))
        #reg_line = coefs[0] + coefs[1] * x #+ coefs[2]*x**2
        
        fig, ax = plt.subplots()
        
        ax.scatter(X[:, 1], y, color = '#0504aa', s = 0.5)
        #ax.plot(x, reg_line, color = "red", alpha = 0.5)
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
    #AssetDf = pd.concat(regResults, axis = 1)
    #print(AssetDf)
    
    #RegResultList.append(AssetDf)

#Print To Latex
print(allresDf.to_latex(index=False, escape = False)) 

plt.plot(avgMonthlyGammaVec)

