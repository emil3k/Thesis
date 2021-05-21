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
ETFMultipliers = [10, 40, 10]

IsEquityIndex        = [True, False, True, False, True, False, False]
#UnderlyingAssetName   = ["SPX US Equity"]
#UnderlyingTicker      = ["SPX"]
#IsEquityIndex         = [True]

loadloc               = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
prefColor             = '#0504aa'
##########################################################################################


#Load Data

#Load Data
AggregateData = []
for ticker in UnderlyingTicker:
    data  = pd.read_csv(loadloc + ticker + "AggregateData.csv")
    AggregateData.append(data)


AggregateGammaList = []
IndexGammaList = []
ETFGammaList = []
for i in np.arange(0, len(IndexTickers)):
   
    #Store index and ETF data separately for simplicity
    indexticker = IndexTickers[i]
    etfticker   = ETFTickers[i]
    etfmultiplier = ETFMultipliers[i]
    
    IndexData   = pd.read_csv(loadloc + indexticker + "AggregateData.csv")
    ETFData     = pd.read_csv(loadloc + etfticker + "AggregateData.csv")
   
    #Compute aggregata net gamma 
    IndexGamma  = IndexData["netGamma"].to_numpy()
    ETFGamma    = ETFData["netGamma"].to_numpy()
    ETFGamma = ETFGamma / etfmultiplier**2
    
    AggregateGamma = IndexGamma[-len(ETFGamma):] + ETFGamma
    IndexGamma     = IndexGamma[-len(ETFGamma):]
    
    AggregateGammaList.append(AggregateGamma)
    AggregateGammaList.append(AggregateGamma)
    
    IndexGammaList.append(IndexGamma)
    ETFGammaList.append(ETFGamma)


#Compute returns and adjusted gamma
nAssets = len(UnderlyingTicker)
TotalReturns = []
XsReturns    = []
for i in np.arange(0, nAssets):
    data   = AggregateData[i]    
    ticker = UnderlyingTicker[i]
    eqIndexFlag  = IsEquityIndex[i]
    rf           = data["LIBOR"].to_numpy()
    
    def computeRfDaily(data):
        dates            = data["Dates"].to_numpy()
        dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
        daycount         = bt.dayCount(dates4fig)
        
        Rf               = data["LIBOR"].to_numpy() / 100
        RfDaily          = np.zeros((np.size(Rf, 0), ))
        RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
        return RfDaily
    
    if ticker  != "VIX":
        #Use Total Return Index for index, and normal price for ETF
        if eqIndexFlag == True:
            price = data["TR Index"].to_numpy()   
        else:
            price    = data[ticker].to_numpy()
           
        rf       = computeRfDaily(data)    
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
#lagList = [1, 2, 5, 10]
lag     = 1
hist    = False
scatter = False
ETFMultiplierList = np.array([10,40,10])
for j in np.arange(0, len(UnderlyingTicker)):
    ticker    = UnderlyingTicker[j]
    volticker = volIndexTicker[j]
    name      = UnderlyingAssetName[j]
    data      = AggregateData[j]
    aggGamma  = AggregateGammaList[j]
    xsret     = XsReturns[j]
   
    isETF    = np.in1d(ETFTickers, ticker)
    
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
    netGamma       = aggGamma #set netGamma equal to aggGamma to be able to reuse code
    MAdollarVolume = data["MADollarVolume"].to_numpy() 
    ILLIQ          = data["ILLIQ"].to_numpy()
    IVOL           = data["IVOL"].to_numpy()
    volIndex       = data[volticker].to_numpy()
    
    #trim data to align with shortened net Gamma
    if len(aggGamma) != len(xsret):
        xsret     = xsret[-len(aggGamma):]
        marketCap = marketCap[-len(aggGamma):]
        IVOL      = IVOL[-len(aggGamma):]
        ILLIQ     = ILLIQ[-len(aggGamma):]
        MAdollarVolume = MAdollarVolume[-len(aggGamma):]
        volIndex  = volIndex[-len(aggGamma):]
  
    #plt.plot(ILLIQ)
    #plt.plot(np.ones((len(ILLIQ),))*ILLIQMedian, "--r")
    #plt.plot(np.ones((len(ILLIQ),))*ILLIQMean, "--b")
    
    #Net Gamma Measures
    netGamma_norm   = (netGamma - np.mean(netGamma)) / np.std(netGamma)
    
    #Use barbon measure for VIX
    
    netGamma_scaled = netGamma / marketCap
    netGamma_barbon = netGamma / MAdollarVolume
    
       
   
    #Standardize meaesure
    #netGamma_scaled = (netGamma_scaled - np.mean(netGamma_scaled)) / np.std(netGamma_scaled)
    #netGamma_scaled = netGamma_scaled / np.std(netGamma_scaled)
    
    #Concatenate Independent variables to X matrix
    #X = np.concatenate((netGamma_scaled.reshape(-1,1), IVOL.reshape(-1,1)), axis = 1)
    
    ###Control Regression###
    #Lagged values
    # IVOL1          = IVOL[2:].reshape(-1,1)
    # IVOL2          = IVOL[1:-1].reshape(-1,1)
    # IVOL3          = IVOL[0:-2].reshape(-1,1)
   
    # AbsXsReturns  = np.abs(xsret)
    # AbsXsRet1  = AbsXsReturns[2:].reshape(-1,1)
    # AbsXsRet2  = np.abs(xsret[1:-1]).reshape(-1,1)
    # AbsXsRet3  = np.abs(xsret[0:-2]).reshape(-1,1)
   
    # #Contatenate laged values to control matrix
    # X_control = np.concatenate((IVOL2, IVOL3, AbsXsRet1, AbsXsRet2, AbsXsRet3), axis = 1)
    # X_control = np.concatenate((X[2:], X_control), axis = 1)
    # y_control = np.abs(xsret[2:])
    
    
    #Trim to match length of vol index
    volIndex        = volIndex[np.isfinite(volIndex)]
    netGamma_scaled = netGamma_scaled[-len(volIndex):]
    xsret           = xsret[-len(volIndex):]
    ILLIQ           = ILLIQ[-len(volIndex):]
    
    ILLIQMedian    = np.median(ILLIQ)    
    ILLIQMean      = np.mean(ILLIQ)  
    
    #Concatenate Independent variables to X matrix
    X = np.concatenate((netGamma_scaled.reshape(-1,1), volIndex.reshape(-1,1)), axis = 1)
    
    ##Control Regression###
    #Lagged values
    volIndex1      = volIndex[2:].reshape(-1,1)
    volIndex2      = volIndex[1:-1].reshape(-1,1)
    volIndex3      = volIndex[0:-2].reshape(-1,1)
   
    AbsXsReturns  = np.abs(xsret)
    AsXsRet1   = AbsXsReturns[2:].reshape(-1,1)
    AbsXsRet2  = np.abs(xsret[1:-1]).reshape(-1,1)
    AbsXsRet3  = np.abs(xsret[0:-2]).reshape(-1,1)
   
    #Contatenate laged values to control matrix
    X_control = np.concatenate((volIndex2, volIndex3, AsXsRet1, AbsXsRet2, AbsXsRet3), axis = 1)
    X_control = np.concatenate((X[2:], X_control), axis = 1)
    y_control = np.abs(xsret[2:])
    
    

    
    ######## Illiquidity regression ###############
    negNetGammaDummy   = (netGamma_scaled < 0)
    IlliquidDummy      = (ILLIQ > ILLIQMedian)
    InteractionDummy   = IlliquidDummy * negNetGammaDummy
    
    X_liq = np.concatenate((negNetGammaDummy.reshape(-1,1), IlliquidDummy.reshape(-1,1), InteractionDummy.reshape(-1,1)), axis = 1)
    
    ######## Reversal Regression ####################
    #Dummies
    negNetGammaDummy = (netGamma_scaled < 0)
    posNetGammaDummy = (netGamma_scaled > 0)
    negReturnDummy   = (xsret < 0)
    posReturnDummy   = (xsret > 0)
    
    #Interactions
    negneg = negNetGammaDummy * negReturnDummy
    negpos = negNetGammaDummy * posReturnDummy
    posneg = posNetGammaDummy * negReturnDummy
    pospos = posNetGammaDummy * posReturnDummy
    
    X_rev = np.concatenate((negneg.reshape(-1,1), negpos.reshape(-1,1), posneg.reshape(-1,1), pospos.reshape(-1,1)), axis = 1)
    
    
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
    if j == 0:
        resultsDf["Lag = " + str(lag) + " day"] = legend
        resultsDf[ticker] = results
        allresDf = resultsDf
    else:
        resultsDf[ticker] = results
     
    if j > 0:
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
        
    ### Alternative Result Print
    legend_control = np.array(['$\Gamma^{MM}_{t-' + str(lag) + ', Agg}$', " ", '$IV_{t-1}$', " ", '$IV_{t-2}$', " ", '$IV_{t-3}$', " ", \
                       '$|R_{t-1}|$', " ", '$|R_{t-2}|$', " ", '$|R_{t-3}|$', " ",  'Intercept', " ", '$R^2$' ])
        
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
                                 str(coefs_control[0]) + sign_test_control[0], "(" + str(tvals_control[0]) + ")", r_squared_control])        
        
        
    results_controlDf = pd.DataFrame()
    if j == 0:
        results_controlDf["Lag = " + str(lag) + " day"] = legend_control
        results_controlDf[ticker] = results_control
        allres_controlDf = results_controlDf
    else:
        results_controlDf[ticker] = results_control
   
        
    if j > 0 :
        allres_controlDf = pd.concat((allres_controlDf, results_controlDf), axis = 1)
        
    
    
    
    ########################################
    ######### Liquidity Regression #########
    
    y_liq  = np.abs(xsret[lag:])
    X_liq  = sm.add_constant(X_liq*1)
    X_liq  = X_liq[0:-lag]
    
    reg_liq       = sm.OLS(y_liq, X_liq).fit(cov_type = "HAC", cov_kwds= {'maxlags':1})
    coefs_liq     = np.round(reg_liq.params*100, decimals = 3)   #Multiply net gamma coef by 100 to get bps format
    tvals_liq     = np.round(reg_liq.tvalues, decimals = 3)
    pvals_liq     = np.round(reg_liq.pvalues, decimals = 3)
    r_squared_liq = np.round(reg_liq.rsquared, decimals = 3)
    
    ### Alternative Result Print
    legend_liq = np.array(['Negative $\Gamma^{MM}$', " ", 'Low Liquidity', " ", 'Negative $netGamma \times $ Low Liquidity', " ",  'Intercept', " ", '$R^2$' ])
        
    sign_test_liq = []
    for pval in pvals_liq:
        if pval < 0.01:
            sign_test_liq.append("***")
        elif pval < 0.05:
            sign_test_liq.append("**")
        elif pval < 0.1:
            sign_test_liq.append("*")
        else:
            sign_test_liq.append("")

    results_liq = np.array([ str(coefs_liq[1]) + sign_test_liq[1], "(" + str(tvals_liq[1]) + ")", \
                             str(coefs_liq[2]) + sign_test_liq[2], "(" + str(tvals_liq[2]) + ")", \
                             str(coefs_liq[3]) + sign_test_liq[3], "(" + str(tvals_liq[3]) + ")", \
                             str(coefs_liq[0]) + sign_test_liq[0], "(" + str(tvals_liq[0]) + ")", r_squared_liq])        
    #Store in DataFrame      
    results_liqDf = pd.DataFrame()
    if j == 0:
        results_liqDf["Lag = " + str(lag) + " day"] = legend_liq
        results_liqDf[ticker] = results_liq
        allres_liqDf = results_liqDf
    elif j < len(UnderlyingTicker)-1:
        results_liqDf[ticker] = results_liq
           
    if j > 0 and j < len(UnderlyingTicker)-1:
        allres_liqDf = pd.concat((allres_liqDf, results_liqDf), axis = 1)
   
    ########################################
    ######### Reversal Regression #########
   
    y_rev  = xsret[lag:]
    X_rev  = sm.add_constant(X_rev*1)
    X_rev  = X_rev[0:-lag]
    
    reg_rev       = sm.OLS(y_rev, X_rev).fit()
    coefs_rev     = np.round(reg_rev.params*100, decimals = 3)   #Multiply net gamma coef by 100 to get bps format
    tvals_rev     = np.round(reg_rev.tvalues, decimals = 3)
    pvals_rev     = np.round(reg_rev.pvalues, decimals = 3)
    r_squared_rev = np.round(reg_rev.rsquared, decimals = 3)
    
    ### Alternative Result Print
    legend_rev = np.array(['Neg-Neg', " ", 'Neg-Pos', " ", 'Pos-Neg', " ",'Pos-Pos', " ", 'Intercept', " ", '$R^2$' ])
        
    sign_test_rev = []
    for pval in pvals_rev:
        if pval < 0.01:
            sign_test_rev.append("***")
        elif pval < 0.05:
            sign_test_rev.append("**")
        elif pval < 0.1:
            sign_test_rev.append("*")
        else:
            sign_test_rev.append("")

    results_rev = np.array([ str(coefs_rev[1]) + sign_test_rev[1], "(" + str(tvals_rev[1]) + ")", \
                             str(coefs_rev[2]) + sign_test_rev[2], "(" + str(tvals_rev[2]) + ")", \
                             str(coefs_rev[3]) + sign_test_rev[3], "(" + str(tvals_rev[3]) + ")", \
                             str(coefs_rev[4]) + sign_test_rev[4], "(" + str(tvals_rev[4]) + ")", \
                                 str(coefs_rev[0]) + sign_test_rev[0], "(" + str(tvals_rev[0]) + ")", r_squared])        
    #Store in DataFrame      
    results_revDf = pd.DataFrame()
    if j == 0:
        results_revDf["Lag = " + str(lag) + " day"] = legend_rev
        results_revDf[ticker] = results_rev
        allres_revDf = results_revDf
    elif j < len(UnderlyingTicker)-1:
        results_revDf[ticker] = results_rev
           
    if j > 0 and j < len(UnderlyingTicker)-1:
        allres_revDf = pd.concat((allres_revDf, results_revDf), axis = 1)
    
   
    
   
    
    
    
    
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
print(allres_controlDf.to_latex(index=False, escape = False))     
print(allres_liqDf.to_latex(index=False, escape = False))      
print(allres_revDf.to_latex(index=False, escape = False))   

# X   = netGamma[0:-lag]
# X   = sm.add_constant(X)
# y   = np.abs(Returns[lag:])

# regression = sm.OLS(y, X).fit()
# print(regression.summary())

