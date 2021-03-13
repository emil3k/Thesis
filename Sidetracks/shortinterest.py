# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:10:20 2021

@author: ekblo
"""
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

## Sort by Put Smirk and Call Smirk Steepness

#load tickers of all stocks in Russell 3000
RussellTickers = pd.read_excel(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\ThesisRepository\Sidetracks\RussellTickers.xlsx")
isListed = RussellTickers["Company"].to_numpy() != "Delisted"
tickers  = RussellTickers["Ticker"].to_numpy()
tickers  = tickers[isListed]
#Get Option Chain From Yahoo Finance

def getOptionChain(symbol, frontOnly = False):
    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
    
    # Get options for each expiration
    options = pd.DataFrame()
    
    if frontOnly == True:
        e = exps[0]
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)
    else:    
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
    
    if type(divYield) != type(None):
        delta_call = np.exp(-divYield*T) * norm.cdf(d1)
        delta_put  = np.exp(-divYield*T) * delta_call - 1
        
        gamma      = 1 / (S * IV * np.sqrt(T)) * (np.exp(-divYield*T) / np.sqrt(2*np.pi)) * np.exp(-d1**2 / 2)
        delta = cp_flag * delta_call + (1 - cp_flag) * delta_put
    else:
        delta_call = norm.cdf(d1)
        delta_put  = delta_call - 1
        
        gamma      = 1 / (S * IV * np.sqrt(T)) * (1 / np.sqrt(2*np.pi)) * np.exp(-d1**2 / 2)
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

def getOptionChainFast(symbol, frontOnly = False):
    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options
    
    # Get options for each expiration
    options = pd.DataFrame()
    
    if frontOnly == True:
        e = exps[0]
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)
    else:    
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
    
    S = tk.info["previousClose"]
   
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    spotDummy = np.ones((len(options), 1))*S
    options["spot"] = spotDummy
                        
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options



#function for ATM options
def getATMandOTMOption(chain, callflag = True, OTMdist = 0.1):
    strikes = chain["strike"].to_numpy()
    spot    = chain["spot"].to_numpy()
    moneyness = spot/strikes
    isATM = (np.abs(1 - moneyness) == np.min(np.abs(1 - moneyness)))
    if callflag == True:
        dist  = 1 - OTMdist
        isOTM = (np.abs(dist - moneyness) == np.min(np.abs(dist - moneyness)))
    else:
        dist = 1 + OTMdist
        isOTM = (np.abs(dist - moneyness) == np.min(np.abs(dist - moneyness)))
    
    ATMOption = chain.loc[isATM, :]
    OTMOption = chain.loc[isOTM, :]
    
    return ATMOption, OTMOption


#function for ATM options
def getATMandOTMOptionArr(chain, callflag = True, OTMdist = 0.1):
    strikes = chain[:, 1]
    spot    = chain[:, -1]
    moneyness = spot/strikes
    isATM = (np.abs(1 - moneyness) == np.min(np.abs(1 - moneyness)))
    if callflag == True:
        dist  = 1 - OTMdist
        isOTM = (np.abs(dist - moneyness) == np.min(np.abs(dist - moneyness)))
    else:
        dist = 1 + OTMdist
        isOTM = (np.abs(dist - moneyness) == np.min(np.abs(dist - moneyness)))
    
    ATMOption = chain[isATM, :]
    OTMOption = chain[isOTM, :]
    
    return ATMOption, OTMOption


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



#chain_test = getOptionChainFast(tickers[0])



#summaryCols = np.array(["ticker", "Short pct of float", "put Skew", "call Skew", "net Gamma"])
summaryCols = np.array(["ticker", "Short pct of float", "put Skew", "call Skew"])
data = []
#Compute smirk
nStocks = np.size(tickers)
for i in np.arange(0, nStocks):
    ticker = tickers[i]    
    
    if (i % 100) == 0:
        print(i)
        
    tk       = yf.Ticker(ticker) #grab yahoo finance ticker
    try:
        shortPct = tk.info["shortPercentOfFloat"] #grab short interest
    except: 
        continue
    
    if shortPct == None or shortPct < 0.1:
        continue
    
    
    try:
        chain    = getOptionChainFast(ticker, frontOnly = True) #grab option chain
    except:
        continue
    
    cols     = chain.columns
    chainArr = chain.to_numpy()
    #try: 
    #    netGamma = computeNetGammaExposure(chain) #compute MM net gamma exposure
    #except:
    #    netGamma = np.nan
        
    #Grab front options Df
    #frontIdx   = chain["dte"].to_numpy() == np.min(chain["dte"].to_numpy()) #identify front options
    #frontChain = chain.loc[frontIdx, :] #grab front options
    #frontCalls =  frontChain.loc[frontChain["CALL"], :] #grab calls
    #frontPuts  =  frontChain.loc[frontChain["CALL"] == 0, :] #grab puts
    
    
    #rontIdx   = chainArr[:,9]  == np.min(chainArr[:, 9]) #identify front options
    #frontChain = chainArr[frontIdx, :] #grab front options
    
    frontCalls =  chainArr[chainArr[:, 10] == 1, :] #grab calls
    frontPuts  =  chainArr[chainArr[:, 10] == 0, :] #grab puts
    
    
    
    try:
        ATMCall, OTMCall = getATMandOTMOptionArr(frontCalls, OTMdist = 0.1) #grab 1 ATM and 1 OTM call
        ATMPut, OTMPut   = getATMandOTMOptionArr(frontPuts, callflag = False, OTMdist = 0.1) #grab 1 ATM and 1 OTM put
    except:
        continue 
    
    #putSkew  = OTMPut["impliedVolatility"].to_numpy() - OTMCall["impliedVolatility"].to_numpy() #compute put skew (from Xing 2010)
    #callSkew = OTMCall["impliedVolatility"].to_numpy() - ATMPut["impliedVolatility"].to_numpy() #compute call skew
    
    putSkew  = OTMPut[:, 6] - OTMCall[:, 6] #compute put skew (from Xing 2010)
    callSkew = OTMCall[:, 6] - ATMPut[:, 6] #compute call skew
    
    #Store results
    #summary  = np.array([ticker, shortPct, np.round(float(putSkew), decimals = 4), np.round(float(callSkew), decimals = 4), np.round(netGamma, decimals = 4)])
    summary  = np.array([ticker, shortPct, np.round(float(putSkew), decimals = 4), np.round(float(callSkew), decimals = 4)])
    
    data.append(summary)

dataDf = pd.DataFrame.from_records(data, columns = summaryCols)


dataSorted = dataDf.sort_values(["Short pct of float", "call Skew"], ascending = False)




saveloc = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/"
dataDf.to_csv(path_or_buf = saveloc + "shortinterestdata.csv" , index = False)




sys.exit()
today = pd.datetime.today()
td    = bt.yyyymmdd(np.array([today]))








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
    









