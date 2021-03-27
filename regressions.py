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
       XsReturns.append(xsret)
    
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
lagList = [1]
nRegs   = len(lagList)
hist    = False
scatter = False
RegResultList = []

for j in np.arange(0, len(UnderlyingTicker)):
    ticker   = UnderlyingTicker[j]
    name     = UnderlyingAssetName[j]
    data     = AggregateData[j]
    xsret    = XsReturns[j]
    
    #Compute Independent Variable Time Series
    #Grab necessary data
    netGamma       = data["netGamma"].to_numpy()
    marketCap      = data["Market Cap"].to_numpy()
    MAdollarVolume = data["MADollarVolume"].to_numpy() 
    
    
    #Net Gamma Measures
    netGamma_norm   = (netGamma - np.mean(netGamma)) / np.std(netGamma)
    netGamma_scaled = netGamma / marketCap
    netGamma_barbon = netGamma / MAdollarVolume
    IVOL            = data["IVOL"].to_numpy()
   
    #Concatenate Independent variables to X matrix
    X = np.concatenate((netGamma_scaled.reshape(-1,1), IVOL.reshape(-1,1)), axis = 1)
    
    ###Control Regression###
    #Lagged values
    IVOL1          = IVOL[2:].reshape(-1,1)
    IVOL2          = IVOL[1:-1].reshape(-1,1)
    IVOL3          = IVOL[0:-2].reshape(-1,1)
   
    AbsXsReturns  = np.abs(xsret)
    AbsXsRet1  = AbsXsReturns[2:].reshape(-1,1)
    AbsXsRet2  = np.abs(xsret[1:-1]).reshape(-1,1)
    AbsXsRet3  = np.abs(xsret[0:-2]).reshape(-1,1)
   
    #Contatenate laged values to control matrix
    X_control = np.concatenate((IVOL2, IVOL3, AbsXsRet1, AbsXsRet2, AbsXsRet3), axis = 1)
    X_control = np.concatenate((X[2:], X_control), axis = 1)
    
    
    #Feature correlation
    IndependentVarDf = pd.DataFrame.from_records(X_control, columns = [r'$netGamma_t%.5f$', r'$IVOL_t%.5f$', r'$IVOL_{t-1}%.5f$', r'$IVOL_{t-2}%.5f$', r'$|R_t|%.5f$', r'$|R_{t-1}|%.5f$', r'$|R_{t-2}|%.5f$' ])
    corrMatrix       = IndependentVarDf.corr()
    sn.heatmap(corrMatrix, annot = True)

    
   
    regResults = []
    
    for i in np.arange(0, nRegs):
        lag    = lagList[i]
        y      = np.abs(xsret[lag:])
        nObs   = np.size(y)
   
       
        X      = X[0:-lag, :]       #Lag matrix accordingly 
        X      = sm.add_constant(X) #add constant
    
        reg       = sm.OLS(y, X).fit()
        coefs     = np.round(reg.params, decimals = 3)
        tvals     = np.round(reg.tvalues, decimals = 3)
        pvals     = np.round(reg.pvalues, decimals = 3)
        r_squared = np.round(reg.rsquared, decimals = 3)
            
        legend  = np.array(['\textbf{' +name + '}', "coefficient", "t stats", "p-values"])
        alpha   = np.array([" ", coefs[0], tvals[0], pvals[0]])
        b1      = np.array([" ", coefs[1], tvals[1], pvals[1]])
        b2      = np.array([" ", coefs[2], tvals[2], pvals[2]])
        r       = np.array([" ", r_squared, " ", " ", ])
        lag_df = pd.DataFrame()
        
        
        #LASSO FOR Parameter selection
        LassoFeature = LassoCV(n_alphas = 100, fit_intercept = False, normalize = False, cv = 10, \
                              max_iter = 10000, tol = 1e-3)
        LassoFeature.fit(X, y)
        LassoCoefs = LassoFeature.coef_
        isImportantFeatures = (LassoCoefs != 0) 
        
        
        if i == 0: #First Data Frame
            lag_df["Lag = " + str(lag) + " day(s)"]  = legend
            lag_df[r'$\alpha$']                 = alpha
            lag_df['$netGamma_t$']             = b1
            lag_df['$IVOL_t$']                 = b2
            lag_df['$R^2$']                    = r
        else:
            lag_df["alpha (Lag = " + str(lag) + ")"]    = alpha
            lag_df["netGamma (Lag = " + str(lag) + ")"] = b1 
            lag_df["IVOL (Lag = " + str(lag) + ")"]     = b2 
        
        regResults.append(lag_df)
        print(lag_df.to_latex(index=False, escape = False)) #print to latex

        
        legend = np.array(['$netGamma_t$', " ", '$IVOL_t$', " ", 'Intercept', " ", '$R^2$' ])
        
        sign_test = []
        for coef in coefs:
            if coef < 0.01:
                sign_test.append("***")
            elif coef < 0.05:
                sign_test.append("**")
            elif coef < 0.1:
                sign_test.append("*")
            else:
                sign_test.append("")
        
            
        results = np.array([coefs[1], ])



        sys.exit()     
        
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
    AssetDf = pd.concat(regResults, axis = 1)
    print(AssetDf)
    
    RegResultList.append(AssetDf)
    
   
    
   
    
# X   = netGamma[0:-lag]
# X   = sm.add_constant(X)
# y   = np.abs(Returns[lag:])

# regression = sm.OLS(y, X).fit()
# print(regression.summary())

