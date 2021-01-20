# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:32:35 2021

@author: ekblo
"""
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import Backtest as bt
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
np.seterr(divide='ignore', invalid='ignore')

#Get Option Chain From Yahoo Finance
def getOptionChain(symbol):
    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
    
    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)
    
    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    #Compute BS Delta and Gamma  
    tbill = yf.Ticker("^IRX") #Get risk free rate (US 13-week T-bill)
    rf    = tbill.info["previousClose"] / 100
    
    cp_flag   = 1*options["CALL"].to_numpy()
    K         = options["strike"].to_numpy()
    S         = tk.info["previousClose"]
    divYield  = tk.info["yield"]
    IV        = options["impliedVolatility"].to_numpy()
    T         = options["dte"].to_numpy()
    
    d1 = (np.log(S/K) + (rf + IV**2/2)*T) / (IV*np.sqrt(T))
    
    delta_call = np.exp(-divYield*T) * norm.cdf(d1)
    delta_put  = np.exp(-divYield*T) * delta_call - 1
    
    gamma      = 1 / (S * IV * np.sqrt(T)) * (np.exp(-divYield*T) / np.sqrt(2*np.pi)) * np.exp(-d1**2 / 2)
    delta = cp_flag * delta_call + (1 - cp_flag) * delta_put
    
    #Add to dataframe
    options["delta"] = delta
    options["gamma"] = gamma
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    spotDummy = np.ones((len(options), 1))*S
    options["spot"] = spotDummy
                        
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

#Compute MM net Gamma Exposure
def computeNetGammaExposure(optionChain):
    #grab needed columns
    openInterest = optionChain["openInterest"].to_numpy()
    gamma        = optionChain["gamma"].to_numpy()
    keepIndex    = np.isfinite(openInterest) * np.isfinite(gamma) # columns to keep
    
    #Trim data
    optionChainTr = optionChain.loc[keepIndex, :] 
    
    gammaTr        = optionChainTr["gamma"].to_numpy()
    openInterestTr = optionChainTr["openInterest"].to_numpy()
    cp_flag        = optionChainTr["CALL"].to_numpy() 
    
    #Compute net gamma
    call_gamma = gammaTr[cp_flag] * openInterestTr[cp_flag] * 100
    put_gamma  = gammaTr[(cp_flag == 0)] * openInterestTr[(cp_flag == 0)] * 100
    netGamma   = np.sum(call_gamma) - np.sum(put_gamma) 
    
    return netGamma

yahoo_ticker = "SPY"
options      = getOptionChain(yahoo_ticker)
netGamma     = computeNetGammaExposure(options)
today        = datetime.date.today()

print("Market Maker Net Gamma Before Open", today, ":",  netGamma)


#Plot IV surface
expDates       = options["expirationDate"]
expDatesNum    = bt.yyyymmdd(expDates).reshape(len(expDates,))
uniqueExpDates = np.unique(expDatesNum)
nExpirations   = np.size(uniqueExpDates)
OTMFlag        = options["inTheMoney"].to_numpy() == 0
cp_flag        = options["CALL"].to_numpy()
ImpliedVol     = options["impliedVolatility"].to_numpy()
spot           = options["spot"].to_numpy()
strike         = options["strike"].to_numpy()

#Vsurface = []

matName = np.array(["Front Month", "First Back Month", "Second Back Month"])
colors  = np.array(["blue", "red", "black"])
for i in np.arange(0, 3):
    exp        = uniqueExpDates[i]
    isRightExp = (exp == expDatesNum)
    
    rel_OTM    = OTMFlag[isRightExp]
    rel_IV     = ImpliedVol[isRightExp]
    rel_spot   = spot[isRightExp]
    rel_strike = strike[isRightExp]
    moneyness  = rel_strike / rel_spot    
    
    #Keep Only OTM Values
    moneyness_OTM = moneyness[rel_OTM]
    IV_OTM         = rel_IV[rel_OTM]
    nOTMOptions   = np.size(moneyness_OTM)
    
    IVmat   = np.concatenate((IV_OTM.reshape(nOTMOptions, 1), moneyness_OTM.reshape(nOTMOptions, 1)), axis = 1)
    IVdf    = pd.DataFrame.from_records(IVmat, columns = ["ImpliedVol", "Moneyness"])
    sortDf  = IVdf.sort_values("Moneyness")
    sortMat = sortDf.to_numpy()
    
    plt.plot(sortMat[:, 1], sortMat[:, 0], color = colors[i], label = matName[i])   
    plt.title("Implied Volatility - No Smoothing")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("IV (%)")
    plt.legend()
    









