# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:23:47 2021

@author: ekblo
"""

#Option Data Exploration
import numpy as np
import pandas as pd
import Backtest as bt
import time
import sys

OptionData = pd.read_csv(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\OptionData\SPXOptionData2.csv")
SpotData   = pd.read_excel(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\SpotData\SPXSPYData.xlsx", "Prices")

def trimToStartDate(OptionData, startDate):
    optionDates = OptionData["date"].to_numpy()
    startInd    = np.nonzero(optionDates > startDate)[0]
    startInd    = startInd[0]
    OptionData  = OptionData.iloc[startInd:, :]
    return OptionData

#startDate   = 20180101
#OptionData  = trimToStartDate(OptionData, startDate)
print(OptionData.head())

datesSeries = SpotData["Dates"]
datesTime   = pd.to_datetime(datesSeries, format = '%d.%m.%Y')
dayCount    = bt.dayCount(datesTime) #get ndays between dates

UnderlyingDates  = bt.yyyymmdd(datesTime) #get desired date format
UnderlyingPrices = SpotData["SPX Index"].to_numpy()

#[OptionDataClean, OptionDataToTrade, AmericanOptionDataClean, UnderlyingData] = bt.CleanOptionData(OptionData, UnderlyingDates, UnderlyingPrices)

tic = time.time()
#Clean Option Data
ColsToKeep = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume",\
                       "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta",\
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
#Store and return as "All" for return and sync with gamma exposure later
UnderlyingDatesAll  = UnderlyingDates[StartInd:EndInd + 1]
UnderlyingPricesAll = UnderlyingPrices[StartInd:EndInd + 1]


#Check if all Option Dates are in Underlying Sample
if np.sum(np.in1d(UniqueDates, UnderlyingDates)) != np.size(UniqueDates):
    raise ValueError("Option Dates are unaccounted for in Underlying dates")

#Trim Data from underlying to match that of the option data
keepIndex = (np.in1d(UnderlyingDatesAll, UniqueDates) == 1) #Dates where option data is recorded
UnderlyingDates  = UnderlyingDatesAll[keepIndex] #Keep underlying dates and prices where matching options
UnderlyingPrices = UnderlyingPricesAll[keepIndex]


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

## Attach Spot price to option data
## Obtain OTM and ATM Flags
#Grab data necessary
OptionStrikes = OptionDataTr[:, 3]
CallFlag      = OptionDataTr[:, 2]
ForwardPrice  = OptionDataTr[:, 14]

#Initialize
UnderlyingVec    = np.zeros((1, 1))
for i in np.arange(0, nDays):
    CurrentDate       = UnderlyingDates[i]            #Grab current date
    CurrentUnderlying = UnderlyingPrices[i]           #Grab underlying price    
    isRightDate       = (CurrentDate == OptionDates)  #right date boolean
    Strikes           = OptionStrikes[isRightDate]    #Grab strikes for right date
  
    nStrikes          = np.size(Strikes)  
    Underlying_dummy  = CurrentUnderlying * np.ones((nStrikes, 1)) #vector of underlying
   
    UnderlyingVec     = np.concatenate((UnderlyingVec, Underlying_dummy), axis = 0)    


#Delete initialization value
UnderlyingVec    = UnderlyingVec[1:]

#Define OTM and ATM from moneyness
def computeMoneynessFlag(Strike, Spot, CallFlag, level):
    nObs = np.size(Strike)
    upper = 1 + level
    lower = 1 - level
    
    Moneyness = Spot / Strike
    CallOTM    = CallFlag * (Moneyness < lower)
    PutOTM     = (1 - CallFlag) * (Moneyness > upper)
    OTM_flag   = CallOTM + PutOTM
    ATM_flag   = (Moneyness > lower)*(Moneyness < upper)*1    
    
    return OTM_flag.reshape(nObs, 1), ATM_flag.reshape(nObs, 1)    
    
[OTM_flag, ATM_flag]   = computeMoneynessFlag(OptionStrikes, UnderlyingVec.reshape(nObs,), CallFlag, 0.05)
[OTMF_flag, ATMF_flag] = computeMoneynessFlag(OptionStrikes, ForwardPrice, CallFlag, 0.05)

OptionDataTr         = np.concatenate((OptionDataTr, mid_price.reshape(nObs, 1), eur_flag.reshape(nObs, 1), OTMF_flag, OTM_flag, UnderlyingVec, ATMF_flag, ATM_flag), axis = 1)  
AmericanOptionDataTr = OptionDataTr[~eur_flag, :]        #Store American Option Data separately
OptionDataTr         = OptionDataTr[eur_flag, :]         #Keep only European Options
OptionDataToTrade    = OptionDataTr[OptionsToTrade, :]   #Options To Trade



#colsFull = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", \
#                       "open_interest", "impl_volatility", "delta", "gamma",  "vega", "theta", "contract_size", \
#                           "forward_price", "mid_price", "european_flag", "OTM_forward_flag", "OTM_flag", "spot_price"])

cols  = np.array(["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", \
                       "open_interest", "impl_volatility", "delta", "gamma",  "vega", "theta", "contract_size", \
                           "forward_price", "mid_price", "european_flag", "OTM_forward_flag", "OTM_flag", "spot_price", "ATMF_flag", "ATM_flag"])


UnderlyingData     = np.concatenate((UnderlyingDatesAll.reshape(np.size(UnderlyingDatesAll), 1), UnderlyingPricesAll.reshape(np.size(UnderlyingPricesAll), 1)), axis = 1)

OptionDataClean         = pd.DataFrame.from_records(OptionDataTr, columns = cols)
AmericanOptionDataClean = pd.DataFrame.from_records(AmericanOptionDataTr, columns = cols)
OptionDataToTrade       = pd.DataFrame.from_records(OptionDataToTrade, columns = cols)    
UnderlyingData          = pd.DataFrame.from_records(UnderlyingData, columns = ["Dates", "Price"])

toc = time.time()

print (toc-tic)
