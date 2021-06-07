# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:48:36 2020

@author: ekblo
"""
#Function gives first and last day in period for some vector of dates
#Use for compounding returns
#Input vector of dates as np.array of format yyyymmdd
#nDigits controls number of periods to skip. 2 gives from daily to monthly
#4 gives from daily to yearly

def GetFirstAndLastDayInPeriod(dates, nDigits):
    import numpy as np
    
    #From daily to monthly or daily to yearly?
    if nDigits == 2:
        denum = 100
    elif nDigits == 4:
        denum = 10000
    
    nDays = np.size(dates)        
    isFirstDay = np.zeros((nDays, 1))
    isLastDay  = np.zeros((nDays, 1))
    datesTrimmed = np.round(dates / denum)
    
    for i in np.arange(0, nDays - 1):
        if datesTrimmed[i] < datesTrimmed[i + 1]:
           isLastDay[i] = 1
           isFirstDay[i + 1] = 1
           
    isFirstDay[0] = 1
    isLastDay[-1] = 1
    
    FirstDayList = np.nonzero(isFirstDay > 0)
    FirstDayList = FirstDayList[0]
    LastDayList  = np.nonzero(isLastDay > 0)
    LastDayList  = LastDayList[0]

    return FirstDayList, LastDayList


#Compute performance
#Computes most important statistics for backtest of trading strategy
#Input XsReturns of given period, and risk free rate (factors are optional)
#AnnualizationFactor: 255 for daily, 12 for monthly, 1 for annual
#Avg return and standard deviation are in same period as input
#Sharpe ratio, alpha is annualized
#Returns, Std and alpha reported as pct points.
def ComputePerformance(XsReturns, Rf, Factors, AnnualizationFactor):
    import numpy as np
    from scipy.stats import skew, kurtosis
    
    nPeriods        = np.size(Rf)
    if np.size(np.shape(XsReturns)) > 1:
        nAssets         = np.size(XsReturns, 1)
        XsReturns       = XsReturns.reshape((nPeriods, nAssets))
    else:
        XsReturns = XsReturns.reshape(nPeriods, 1)
        
    Rf              = Rf.reshape((nPeriods, 1))
    
    #Returns
    TotalReturns    = XsReturns + Rf
    AvgTotalReturn  = np.prod(1 + TotalReturns, 0)**(1 / nPeriods) - 1
    AvgRf           = np.prod(1 + Rf)**(1/nPeriods) - 1
    AvgXsReturn     = AvgTotalReturn - AvgRf
    
    StdXsReturn     = np.std(XsReturns, 0)
    StdTotalReturn  = np.std(TotalReturns, 0)
    SharpeGeo       = np.sqrt(AnnualizationFactor) * (AvgXsReturn / StdTotalReturn)
    
    SkewXsReturn    = skew(XsReturns, 0)
    KurtXsReturn    = kurtosis(XsReturns, 0)
    minXsReturn     = np.min(XsReturns, 0)
    maxXsReturn     = np.max(XsReturns, 0)

    if np.size(Factors) > 1:
        y = XsReturns
        X = np.concatenate([np.ones([nPeriods, 1]), Factors], axis =  1)
        coeffs = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        alpha  = AnnualizationFactor * coeffs[0]
        mkt_beta = coeffs[1]
        
        legend = np.array(["Avg Total Return", "Avg Excess Return", "Std Excess Return", "Sharpe Geometric", \
                       "Skewness", "Kurtosis", "Min", "Max", "alpha", "Market Beta"])
        
        performance = np.array([AvgTotalReturn*100, AvgXsReturn*100, StdXsReturn*100, SharpeGeo, SkewXsReturn, \
                                KurtXsReturn, minXsReturn*100, maxXsReturn*100, alpha*100, mkt_beta  ] )
        
    else:
        legend = np.array(["Avg Total Return", "Avg Excess Return", "Std Excess Return", "Sharpe Geometric", \
                       "Skewness", "Kurtosis", "Min", "Max"])
        performance = np.array([AvgTotalReturn*100, AvgXsReturn*100, StdXsReturn*100, SharpeGeo, SkewXsReturn, \
                                KurtXsReturn, minXsReturn*100, maxXsReturn*100])
        
    return legend, performance

#Compute performance
#Computes most important statistics for backtest of trading strategy
#Input XsReturns of given period, and risk free rate (factors are optional)
#AnnualizationFactor: 255 for daily, 12 for monthly, 1 for annual
#Avg return and standard deviation are in same period as input
#Sharpe ratio, alpha is annualized
#Returns, Std and alpha reported as pct points.
def FactorPerformance(XsReturns, Rf, Factors, AnnualizationFactor):
    import numpy as np
    from scipy.stats import skew, kurtosis
    import Backtest as bt
    
    nPeriods        = np.size(Rf)
    nAssets         = np.size(XsReturns, 1)
    XsReturns       = XsReturns.reshape((nPeriods, nAssets))
    Rf              = Rf.reshape((nPeriods, 1))
    
    #Returns
    TotalReturns    = XsReturns + Rf
    AvgTotalReturn  = np.prod(1 + TotalReturns, 0)**(1 / nPeriods) - 1
    AvgRf           = np.prod(1 + Rf)**(1/nPeriods) - 1
    AvgXsReturn     = AvgTotalReturn - AvgRf
    
    StdXsReturn     = np.std(XsReturns, 0)
    StdTotalReturn  = np.std(TotalReturns, 0)
    SharpeGeo       = np.sqrt(AnnualizationFactor) * (AvgXsReturn / StdTotalReturn)
    
    SkewXsReturn    = skew(XsReturns, 0)
    KurtXsReturn    = kurtosis(XsReturns, 0)
    minXsReturn     = np.min(XsReturns, 0)
    maxXsReturn     = np.max(XsReturns, 0)

    if np.size(Factors) > 1:
        alphaMat = np.zeros(nAssets,)
        betaMat  = np.zeros(nAssets,)
        PSRMat   = np.zeros(nAssets,)
        for j in np.arange(0, nAssets):
            y = XsReturns[:, j].reshape(nPeriods, 1)
            Factor = Factors[:, j].reshape(nPeriods, 1)
            X = np.concatenate((np.ones((nPeriods, 1)), Factor ), axis =  1)
            coeffs = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
            alphaMat[j]  = AnnualizationFactor * coeffs[0]
            betaMat[j] = coeffs[1]
            
            [PSR, SR, SRb] = bt.PSR(TotalReturns[:, j], Factors[:, j] + Rf.reshape(nPeriods,), Rf.reshape(nPeriods,) , 12)
            PSRMat[j] = PSR
            
            legend = np.array(["Avg Total Return", "Avg Excess Return", "Std Excess Return", "Sharpe Geometric", \
                           "Skewness", "Kurtosis", "Min", "Max", "alpha", "Factor Beta", "PSR"])
            
            performance = np.array([AvgTotalReturn*100, AvgXsReturn*100, StdXsReturn*100, SharpeGeo, SkewXsReturn, \
                                    KurtXsReturn, minXsReturn*100, maxXsReturn*100, alphaMat*100, betaMat, PSRMat  ] )
            
    else:
        legend = np.array(["Avg Total Return", "Avg Excess Return", "Std Excess Return", "Sharpe Geometric", \
                       "Skewness", "Kurtosis", "Min", "Max"])
        performance = np.array([AvgTotalReturn*100, AvgXsReturn*100, StdXsReturn*100, SharpeGeo, SkewXsReturn, \
                                KurtXsReturn, minXsReturn*100, maxXsReturn*100])
        
    return legend, performance



##Compute days between dates
#Input dates in datetime64 format
#Function returns vector of number of days between dates

def dayCount(dates):
    import numpy as np

    nObs     = np.size(dates)
    dayCount = np.zeros((nObs, ))

    for i in np.arange(0, nObs - 1):
        td = dates[i+1] - dates[i]
        td = td.days
        dayCount[i + 1] = td

    return dayCount

#Compute returns
#Prices given

def computeReturns(Prices, dates = None):
    import numpy as np
    nDays = len(Prices)
    
    if np.ndim(Prices) > 1:
        nAssets = np.size(Prices, 1)
        returns = np.zeros((nDays, nAssets))
        returns[1:] = Prices[1:] / Prices[0:-1] - 1
    else:
        returns = np.zeros((nDays,))
        returns[1:] = Prices[1:] / Prices[0:-1] - 1

    if dates == None:
        return returns
    else:
        returns = np.concatenate((dates.reshape(-1,1), returns.reshape(-1,1)), axis = 1)
        return returns
 



##yyyymmdd function
#transforms dates from pd.datetime (datetime64) to integer of format yyyymmdd
#Needed for GetFirstAndLastDayInPeriod
#Input is dates of dateteime64 format

def yyyymmdd(dates):
    import numpy as np
    
    nObs      = np.size(dates)
    datesnice = np.zeros((nObs, 1)) #Preallocate for yyyymmdd format

    for i in np.arange(0, nObs):
        date  = dates[i]     #grab date
        year  = date.year    #grab year
        month = date.month   #grab month
        day   = date.day     #grab day
        
        #Check if "0" needs to be added and convert to string
        if month < 10:
            monthStr = '0' + str(month)
        else:
            monthStr = str(month)
           
        if day < 10:
            dayStr = '0' + str(day)
        else:
            dayStr = str(day) 
        
        yearStr = str(year)
        
        datesStr = yearStr + monthStr + dayStr #concatenate string
        datesnice[i]     = int(datesStr)       #store as integer
        
    return datesnice


## Compute turnover
#Computes turnover by comparing the previous and the new target weights
#of the portfolio, accounting for the returns on the assets.
#The funciton also computes the portfolio return excluding transaction costs Rp
#Input is the matrix of strategy weights, the asset returns and risk free rate
#Function returns vector of turnover and vector of strategy returns

def ComputeTurnover(StrategyWeights, assetReturns, Rf):
        import numpy as np
        
        nPeriods    = np.size(Rf)
        nAssets     = np.size(assetReturns, 1)
        turnoverVec = np.zeros([nPeriods, 1])
        RpVec       = np.zeros([nPeriods, 1])
    
        #Transaction costs
        for i in np.arange(1, nPeriods):
            prevWeights   = StrategyWeights[i - 1, :].reshape(1, nAssets)
            newWeights    = StrategyWeights[i, :].reshape(1, nAssets)
            periodReturns = assetReturns[i, :].reshape(1, nAssets)
            periodRf      = Rf[i]
            
            Rp = np.sum(prevWeights * periodReturns) + (1 - np.sum(prevWeights)) * periodRf
            valuePerAsset  = prevWeights * (1 + periodReturns)
            currentWeights = (valuePerAsset / (1 + Rp)).reshape(1, nAssets)
            turnover       = np.sum(np.abs(newWeights - currentWeights))
            
            turnoverVec[i] = turnover
            RpVec[i] = Rp
            
        turnoverVec[0] = 1
        
        return turnoverVec, RpVec
           
#Adjust Futures Prices for rollovers
#Input Front-month Prices, Back-month Prices, Front-month tickers
#Adjusts for rollovers and returns return series

def RolloverFutures(FrontMonthPrices, BackMonthPrices, FrontMonthTickers, returnRoll = False):
    
    isRollover = 1 - (FrontMonthTickers[0:-1,:] == FrontMonthTickers[1:,:])
    
    futuresReturns = (FrontMonthPrices[1:, :] / FrontMonthPrices[0:-1,:] * (1 - isRollover) \
                + FrontMonthPrices[1:, :] / BackMonthPrices[0:-1, :] * isRollover) - 1


    if returnRoll == True:
        return futuresReturns, isRollover
    else:
        return futuresReturns



## Aggregate futures excess returns taking into account daily mark to market and interest earned on account
#Input rolled futures excess returns, Rf, dates (in yyyymmdd format) and nDigits(2:daily to monthly, 4:daily to yearly)
#Returns compounded total returns, excess returns and risk free rate

def aggregateFutXsReturns(originalReturns, Rf, dates, nDigits):
    import numpy as np
    import Backtest as bt
    
    [firstDayList, lastDayList] = bt.GetFirstAndLastDayInPeriod(dates, nDigits)
    nPeriods = np.size(firstDayList)
    nAssets  = np.size(originalReturns, 1)
    cumRf    = np.zeros([nPeriods, 1])
    aggregatedTotalReturns = np.zeros([nPeriods, nAssets])
    
    for i in np.arange(0, nPeriods):
        first = firstDayList[i]
        last  = lastDayList[i]
        nDays = last - first + 1
        
        #Compute cumulative Rf
        cumRf[i] = np.prod(1 + Rf[first:(last + 1)]) - 1
        
        #Compute normalized futures prices
        futPrices = np.cumprod(1 + originalReturns[first:(last + 1)], 0)        
        
        #Compute daily MTM gain/loss
        MTM = np.zeros([nDays, nAssets])
        MTM[0, :] = futPrices[0, :] - 1
        MTM[1:, :] = futPrices[1:, :] - futPrices[0:-1, :]
        
        #Compute cash balance earning interest and add the interest accrual
        cash = np.ones([1, nAssets])
        for j in np.arange(first, (last + 1)):
            cash = cash*(1 + Rf[j]) + MTM[j - first, :]
      
        aggregatedTotalReturns[i, :] = cash - 1
    
    cumTR = aggregatedTotalReturns
    cumXsR = aggregatedTotalReturns - cumRf
    
    return cumTR, cumXsR, cumRf


#Compounds returns over given period (indicated by nDigits)
#nDigits = 2: daily -> monthly, monthly -> yearly   
#nDigits = 4: daily -> yearly

#Input: XsReturns, Rf, and dates of format yyyymmdd (use Backtest.yyyymmdd())
#Output: compounded TotalReturns, XsReturns, Rf
    
def CompoundReturns(XsReturns, Rf, dates, nDigits):
    import numpy as np
    import Backtest as bt
    
    [FirstDayList, LastDayList] = bt.GetFirstAndLastDayInPeriod(dates, nDigits) #Grab first and last index
    nPeriods = np.size(FirstDayList) #Set first and last date   
    nAssets = np.size(XsReturns, 1) #Find number of assets
    TotalReturns = XsReturns + Rf
    
    cumTr  = np.zeros((nPeriods, nAssets)) #preallocate
    cumRf  = np.zeros((nPeriods, 1))       #preallocate
    
    #Compound returns
    for i in np.arange(0, nPeriods):
        first = FirstDayList[i]
        last  = LastDayList[i]
    
        returns  = TotalReturns[first:(last + 1), :]
        riskfree = Rf[first:(last + 1),:] 
        
        cumTr[i, :] = np.prod(1 + returns, 0) - 1
        cumRf[i] = np.prod(1 + riskfree) - 1
    
    cumXs = cumTr - cumRf
    
    return cumTr, cumXs, cumRf

    
## Computes the probabilistic Sharpe Ration following Bailey, De Prado (2012)
#Input Strategy Total Returns, Total Returns for Benchmark, Rf and Annualization Factor
#Annualization Factor is used to annualize sharpe ratio (12 for monthly returns, 255 for daily)    

# The fucntion Computes a statistical test that the Sharpe Ratio of the strategy is larger than the benchmark
# If PSR > 0.95, the Sharpe of the strategy is statistically significantly larger than that of the
# Benchmark at the 5% significance level.

#Function returns PSR, as well as Sharpe for both Strategy and Benchmark.

def PSR(StrategyTotalReturns, BenchmarkTotalReturns, Rf, AnnualizationFactor):
    import numpy as np
    from scipy.stats import skew, kurtosis, norm
    
    nObs         = np.size(StrategyTotalReturns)
    
    #Compute Moments
    #Mean
    MeanStrategy   = np.prod(1 + StrategyTotalReturns)**(1 / nObs) - 1
    MeanBenchmark  = np.prod(1 + BenchmarkTotalReturns)**(1 / nObs) - 1
    MeanRf         = np.prod(1 + Rf)**(1 / nObs) - 1
    
    #Standard Deviation
    StdStrategy  = np.std(StrategyTotalReturns)
    StdBenchmark = np.std(BenchmarkTotalReturns) 

    #Skewness and kurtosis
    SkewStrategy = skew(StrategyTotalReturns)
    KurtStrategy = kurtosis(StrategyTotalReturns)
    
    #Sharpe Ratios
    SharpeStrategy  = np.sqrt(AnnualizationFactor)*(MeanStrategy - MeanRf) / StdStrategy
    SharpeBenchmark = np.sqrt(AnnualizationFactor)*(MeanBenchmark - MeanRf) / StdBenchmark
    
    #Probabilistic Sharpe Ratio
    num   = (SharpeStrategy - SharpeBenchmark)*np.sqrt(nObs - 1)     
    denom = np.sqrt(1 - SkewStrategy*SharpeStrategy + ((KurtStrategy - 1)/4)*(SharpeStrategy**2))
    X     = num / denom
    PSR = norm.cdf(X)

    return PSR, SharpeStrategy, SharpeBenchmark
    

    
#Compute Geometric Sharpe Ratio
#Annualization Factor: 12 for monthly, 255 for daily    
def GeometricSharpe(TotalReturns, Rf, AnnualizationFactor):
    import numpy as np

    nObs         = np.size(TotalReturns)
    
    #Mean
    MeanRet   = np.prod(1 + TotalReturns)**(1 / nObs) - 1
    MeanRf    = np.prod(1 + Rf)**(1 / nObs) - 1
    StdRet    = np.std(TotalReturns)
    
    #Sharpe Ratios
    SharpeRatio = np.sqrt(AnnualizationFactor)*(MeanRet - MeanRf) / StdRet
    
    return SharpeRatio


#Split to training and validation set for model selection
def TrainTestSplit(X, y, testAmount, seed):
    import numpy as np
    np.random.seed(seed = seed) #Set seed
    nObs      = np.size(X, 0)
    nFeatures = np.size(X, 1)
    
    Unif      = np.random.uniform(size = (nObs, 1)) #generate uniform random numbers
    TestSelect = (Unif < testAmount).reshape(nObs,)        #select approximately 15% of training data as validation
    TrainKeep = (TestSelect == 0) #Data to keep in training set
    
    #Grab Validation Set 
    Xtest        = X[TestSelect, :]
    ytest        = y[TestSelect]
    
    #Remove validation variables from training set
    Xtrain        = X[TrainKeep, :]
    ytrain        = y[TrainKeep]
    
    return Xtrain, ytrain, Xtest, ytest



#Clean Option Data from Option Metrics
#Function takes in: 
# - Raw Option Data from Option Metrics
# - Spot prices of underlying security
# - Dates of underlying security

#The function synchronizes the data to the first day both series are available
#It then cleans the data set and returns two datasets
#1) The cleaned option data containing all options in the raw set
#2) The traded option data set, which only includes standard options

def CleanOptionData(OptionData, UnderlyingDates, UnderlyingPrices):
    import numpy as np
    import pandas as pd
    import Backtest as bt
    
    #Clean Option Data
    ColsToKeep = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", \
                           "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta", \
                               "contract_size", "forward_price"])
    
    ColsForTrade = np.array(["am_settlement", "ss_flag", "expiry_indicator", "index_flag", "exercise_style", "am_set_flag"])
        
    #Grab and store needed option data
    OptionDates      = OptionData["date"].to_numpy()           #Grab option dates
    UniqueDates      = np.unique(OptionDates)                  #Grab unique option dates
    ExpirationDates  = OptionData["exdate"].to_numpy()         #Grab expiration dates
    OptionData["cp_flag"] = (OptionData["cp_flag"] == "C") * 1 #Transform flag to numeric
    OptionData["best_bid"] = OptionData["best_bid"] / OptionData["contract_size"]
    OptionData["best_offer"] = OptionData["best_offer"] / OptionData["contract_size"]
   
   
    OptionDataTr     = OptionData[ColsToKeep].to_numpy()       #Extract columns that should be kept as is
    OptionDataTr[:, 3] = OptionDataTr[:, 3] / 1000             #Adjust strike price by dividing by 1000
    nDays = np.size(UniqueDates)
    
    #Sync Option Data and Underlying Data
    #Select start and end date of sample
    #Return error if option sample is longer than underlying sample
    if (UniqueDates[0] < UnderlyingDates[0]):
        raise ValueError("Option Sample Exceeds Underlying Sample")
    else:
        StartDate = UniqueDates[0]
    
    if (UniqueDates[-1] > UnderlyingDates[-1]):
        raise ValueError("Option Sample Exceeds Underlying Sample")
    else:
        EndDate = UniqueDates[-1]
    
    #Trim underlying to match option sample
    StartInd = np.nonzero(UnderlyingDates == StartDate)
    StartInd = StartInd[0]
    
    EndInd  = np.nonzero(UnderlyingDates == EndDate)
    EndInd  = EndInd[0]
    
    #Check if Start and End Dates match
    if (len(StartInd) == 0) or (len(EndInd) == 0):
        raise ValueError("StartDate or EndDate does not match Underlying dates")
    
    #Transform to integers after check
    StartInd = int(StartInd)
    EndInd   = int(EndInd)
    
    #Return Trimmed Values of Underlying
    UnderlyingDates   = UnderlyingDates[StartInd:EndInd + 1]
    UnderlyingPrices  = UnderlyingPrices[StartInd:EndInd + 1]
    
    #Check Dates
    DateDiff = np.abs(UnderlyingDates - UniqueDates.reshape(nDays, 1)) 
    if np.sum(DateDiff) > 0.5:
        raise ValueError("Dates of underlying and option differ")
    
    
    #Construct Trade Indicator (for options to trade)
    #These options are standard index option with AM settlement third friday of each month
    am_settlement = OptionData["am_settlement"].to_numpy()
    ss_flag       = OptionData["ss_flag"].to_numpy()
    exp_indicator = OptionData["expiry_indicator"].to_numpy()
    index_flag    = OptionData["index_flag"].to_numpy()
    ex_style      = OptionData["exercise_style"].to_numpy()
    am_set_flag   = OptionData["am_set_flag"].to_numpy()
    
    #Construct Booleans
    am_settlement = (am_settlement == 1)
    ss_flag       = (ss_flag == 0)
    
    weekly_exp  = (exp_indicator == "w")     #weekly expiration
    daily_exp   = (exp_indicator == "d")     #daily expiration
    non_normal_exp = weekly_exp + daily_exp  #combine for all non-normal exp
    exp_flag    = (non_normal_exp == 0)      #normal exp is whnen non-normal is false
    
    index_flag  = (index_flag == 1)          #index option
    eur_flag    = (ex_style == "E")          #European option
    am_set_flag = (am_set_flag == 1)         #AM settlement
    
    #Combine flags to create options to trade indicator
    OptionsToTrade = am_settlement * ss_flag * exp_flag * index_flag * eur_flag * am_set_flag 
    
    
    #Add columns to option data
    nObs = np.size(OptionDates)
    
    #Mid price
    bid   = OptionData["best_bid"].to_numpy()
    offer = OptionData["best_offer"].to_numpy()
    mid_price = (bid + offer) / 2
    
    #Spot Price and OTM Flag
    #Grab data necessary
    OptionStrikes = OptionDataTr[:, 3]
    CallFlag      = OptionDataTr[:, 2]
    ForwardPrice  = OptionDataTr[:, 14]
    
    #Initialize
    OTM_forward_flag = np.zeros((1, 1))
    OTM_flag         = np.zeros((1, 1))
    UnderlyingVec    = np.zeros((1, 1))
    
    for i in np.arange(0, nDays):
        CurrentDate       = UnderlyingDates[i] #Grab current date
        CurrentUnderlying = UnderlyingPrices[i] #Grab underlying price    
    
        isRightDate   = (CurrentDate == OptionDates)  #right date boolean
        Strikes       = OptionStrikes[isRightDate]    #Grab strikes for right date
        Flag          = CallFlag[isRightDate]         #Grab call_flag for right date
        Forward       = ForwardPrice[isRightDate]     #Grab forward price for right date
        
        nStrikes         = np.size(Strikes)  
        Underlying_dummy = CurrentUnderlying * np.ones((nStrikes, 1)) #vector of underlying
        
        #Spot OTM
        CallOTM  = (Flag == 1) * (CurrentUnderlying < Strikes) #OTM dummy for calls
        PutOTM   = (Flag == 0) * (CurrentUnderlying > Strikes) #OTM dummy for puts
        OTM_dummy = CallOTM + PutOTM                           #combine to make OTM dummy for both calls and puts
       
        #Forward OTM
        CallOTM_forward = (Flag == 1)*(Forward < Strikes)      #OTM dummy for calls
        PutOTM_forward  = (Flag == 0)*(Forward > Strikes)      #OTM dummy for puts
        OTM_forward_dummy = CallOTM_forward + PutOTM_forward   #Combine
        
        #Stack flags and underlying for each date on top of eachother
        OTM_flag         = np.concatenate((OTM_flag, OTM_dummy.reshape(nStrikes, 1)), axis = 0)
        OTM_forward_flag = np.concatenate((OTM_forward_flag, OTM_forward_dummy.reshape(nStrikes, 1)), axis = 0)
        UnderlyingVec    = np.concatenate((UnderlyingVec, Underlying_dummy), axis = 0)    
    
    #Delete initialization value
    OTM_flag         = OTM_flag[1:]
    OTM_forward_flag = OTM_forward_flag[1:]
    UnderlyingVec    = UnderlyingVec[1:]
    
    #Add to option data
    OptionDataTr         = np.concatenate((OptionDataTr, mid_price.reshape(nObs, 1), eur_flag.reshape(nObs, 1), OTM_forward_flag, OTM_flag, UnderlyingVec ), axis = 1)   
    AmericanOptionDataTr = OptionDataTr[~eur_flag, :]        #Store American Option Data separately
    OptionDataTr         = OptionDataTr[eur_flag, :]         #Keep only European Options
    OptionDataToTrade    = OptionDataTr[OptionsToTrade, :]   #Options To Trade
    
    #Add ATM Flag for options to trade
    nTradedOptions    = np.size(OptionDataToTrade, 0)
    isFirstExpiration = np.zeros((nTradedOptions, 1))
    isLastExpiration  = np.zeros((nTradedOptions, 1))
    Expirations       = OptionDataToTrade[:, 1]
    
    for i in np.arange(0, nTradedOptions - 1):
        if (Expirations[i] != Expirations[i + 1]):
            isFirstExpiration[i + 1] = 1
            isLastExpiration[i] = 1
        
    isFirstExpiration[0] = 1
    isLastExpiration[-1] = 1
    
    FirstExpList = np.nonzero(isFirstExpiration)
    LastExpList  = np.nonzero(isLastExpiration)
    FirstExpList = FirstExpList[0]
    LastExpList  = LastExpList[0]
    
    nExpirations     = np.size(FirstExpList)
    ATMF_flag = np.zeros((1, 1))
    ATM_flag         = np.zeros((1, 1))   
    
    for i in np.arange(0, nExpirations):
        start = FirstExpList[i]
        stop  = LastExpList[i]
        
        #Grab needed batches from option data to trade
        Flag    = OptionDataToTrade[start:stop + 1, 2]   #Grab Call Flag
        Strikes = OptionDataToTrade[start:stop + 1, 3]   #Grab strikes
        Forward = OptionDataToTrade[start:stop + 1, 14]  #Grab forward
        Spot    = OptionDataToTrade[start:stop + 1, 19]  #Grab Spot
        nStrikes = np.size(Strikes)
        
        diff_forward  = np.abs(Forward - Strikes)
        diff_spot     = np.abs(Spot - Strikes)
        
        #Split by call and put
        #Call
        diff_forward_call = diff_forward[(Flag == 1)]
        diff_spot_call    = diff_spot[(Flag == 1)]
        ATMF_call         = (diff_forward_call == np.min(diff_forward_call))
        ATM_call          = (diff_spot_call == np.min(diff_spot_call))
        
        #Put
        diff_forward_put  = diff_forward[(Flag == 0)]
        diff_spot_put     = diff_spot[(Flag == 0)]
        ATMF_put          = (diff_forward_put == np.min(diff_forward_put))
        ATM_put           = (diff_spot_put == np.min(diff_spot_put))
        
        ATMF_dummy = np.concatenate((ATMF_call, ATMF_put), axis = 0).reshape(nStrikes, 1)
        ATM_dummy  = np.concatenate((ATM_call, ATM_put), axis = 0).reshape(nStrikes, 1)
        
        ATMF_flag = np.concatenate((ATMF_flag, ATMF_dummy), axis = 0)
        ATM_flag  = np.concatenate((ATM_flag, ATM_dummy), axis = 0)
        
    
    ATMF_flag = ATMF_flag[1:]
    ATM_flag  = ATM_flag[1:]    
        
    colsFull = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", \
                           "open_interest", "impl_volatility", "delta", "gamma",  "vega", "theta", "contract_size", \
                               "forward_price", "mid_price", "european_flag", "OTM_forward_flag", "OTM_flag", "spot_price"])
    
    colsToTrade = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", \
                           "open_interest", "impl_volatility", "delta", "gamma",  "vega", "theta", "contract_size", \
                               "forward_price", "mid_price", "european_flag", "OTM_forward_flag", "OTM_flag", "spot_price", "ATMF_flag", "ATM_flag"])
    
    OptionDataToTrade  = np.concatenate((OptionDataToTrade, ATMF_flag, ATM_flag), axis = 1)    
    UnderlyingData     = np.concatenate((UnderlyingDates.reshape(nDays, 1), UnderlyingPrices.reshape(nDays, 1)), axis = 1)
    
    OptionDataClean         = pd.DataFrame.from_records(OptionDataTr, columns = colsFull)
    AmericanOptionDataClean = pd.DataFrame.from_records(AmericanOptionDataTr, columns = colsFull)
    OptionDataToTrade       = pd.DataFrame.from_records(OptionDataToTrade, columns = colsToTrade)    
    UnderlyingData          = pd.DataFrame.from_records(UnderlyingData, columns = ["Dates", "Price"])
    
    return OptionDataClean, OptionDataToTrade, AmericanOptionDataClean, UnderlyingData

## Trim to dates
#Function trims option dataset to set start and end dates
#Dates should be of format yyyymmdd
#Input data can be both np.ndarray or pd.dataframe

def trimToDates(Data, dates, startDate, endDate):
    import numpy as np
    import pandas as pd
    
    if type(dates) == pd.core.series.Series:
        dates       = dates.to_numpy()
        
    startInd = np.nonzero(dates >= startDate)[0]
    if len(startInd) > 0: #if desired start date is before data starts
        startInd    = startInd[0]    
    else:
        startInd = 0
    
    endInd = np.nonzero(dates >= endDate)[0]
    if len(endInd) > 0:
        endInd      = endInd[0]    
    else:
        endInd = np.size(dates) + 1 
    
    if type(Data) == pd.core.frame.DataFrame:
        DataTr  = Data.iloc[startInd:endInd, :]
    elif type(Data) == np.ndarray:
        DataTr = Data[startInd:endInd, :]
    else:
        raise ValueError("Data is not of right format. Should be pd.df of np.ndarray")
    
    return DataTr

#Import all packages needed for backtesting

#Synchronize data
#Arrays should be numpy array with first column as date column 
#Date needs to be float with format yyyymmdd
#If fill previous is false, the "new" observations will have value zero

def SyncData(array1, array2, fillPrevious = False, removeNonOverlapping = False):
    import numpy as np
    import pandas as pd
    #Construct empty SyncMat
    
    nRows   = max(len(array1), len(array2))
    nCols_1 = np.size(array1, 1)
    nCols_2 = np.size(array2, 1)
    nCols   = nCols_1 + nCols_2 - 1
    syncMat = np.zeros((nRows, nCols)) #matrix where synced data wil be stored

    #Assume dates are in first column
    date_bool  = np.in1d(array1[:, 0], array2[:, 0]) #Dates where option data is recorded

    #Check if dates are mismatched both ways
    date_bool2  = np.in1d(array2[:, 0],array1[:, 0]) #Dates where option data is recorded
    if np.sum(date_bool2 == 0) != 0:
        raise ValueError("Dates are mismatched both ways")


    syncMat[:, 0:nCols_1] = array1
    syncMat[date_bool, nCols_1:] = array2[:, 1:]
    
    if removeNonOverlapping == True and fillPrevious == True:
        raise ValueError("Both removeNoneOverlapping and fillPrevious cannot be True")

    if removeNonOverlapping == True:
        keep    = date_bool #keep only overlapping observations
        syncMat = syncMat[keep, :]

    if fillPrevious == True:
        date_shift = np.concatenate((date_bool[1:], np.ones((1,))), axis = 0) == 1 
        syncMat[(date_bool == 0), nCols_1:] = syncMat[(date_shift == 0), nCols_1:]  

    return syncMat



