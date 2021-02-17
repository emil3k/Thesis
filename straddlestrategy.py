# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:44:19 2021

@author: ekblo
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:09:23 2021

@author: ekblo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Backtest as bt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import sys

#Trading Strategy Development
### SET IMPORT PARAMS ####################################################################
UnderlyingBBGTicker   = "SPX Index"
UnderlyingTicker      = "SPX"
loadlocAgg            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/AggregateData/"
loadlocOpt            = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/CleanData/"
loadlocSpot           = "C:/Users/ekblo/Documents/MScQF/Masters Thesis/Data/SpotData/SpotData.xlsx"
prefColor             = '#0504aa'
##########################################################################################

## Load Data
#Aggregate Data
AggregateData   = pd.read_csv(loadlocAgg + UnderlyingTicker + "AggregateData.csv")
AggregateDates  = AggregateData["Dates"].to_numpy()
startDate = AggregateDates[0]
endDate   = AggregateDates[-1]

#Underlying
Underlying      = AggregateData[UnderlyingTicker].to_numpy()
UnderlyingDates = AggregateData["Dates"].to_numpy()

#Compute Daily Rf
def computeRfDaily(data):
        dates            = data["Dates"].to_numpy()
        dates4fig        = pd.to_datetime(dates, format = '%Y%m%d')
        daycount         = bt.dayCount(dates4fig)
        
        Rf               = data["LIBOR"].to_numpy() / 100
        RfDaily          = np.zeros((np.size(Rf, 0), ))
        RfDaily[1:]      = Rf[0:-1] * daycount[1:]/360 
        return RfDaily
RfDaily = computeRfDaily(AggregateData)


#Option Data
#Trim Options data for match aggregate data
OptionsData    = pd.read_csv(loadlocOpt + UnderlyingTicker + "OptionDataToTrade.csv")
OptionDataTr   = bt.trimToDates(OptionsData, OptionsData["date"], startDate, endDate + 1)
OptionDates    = OptionDataTr["date"].to_numpy()
UniqueDates    = np.unique(OptionDates)

#Signal
netGamma = AggregateData["netGamma"].to_numpy()

#Trading strategy 1)
#Open Front Month Position in Straddle whenever gamma is negative
#Hold position until roll or gamma turns negative
#Roll 1-week before expiration


OptionCols      = OptionDataTr.columns
OptionDataArr   = OptionDataTr.to_numpy()


def getATMOptions(options):
    strikes    = options[:, 3]      #grab strikes
    spot       = options[:, 19][0]  #grab spot
    cpflag     = options[:, 2]      #grab call/put flag
    
    callstrikes = strikes[cpflag == 1]
    calloptions = options[cpflag == 1, :]
    callToGetIdx   = (np.abs(callstrikes - spot) == np.min(np.abs(callstrikes - spot)))
    
    putstrikes = strikes[cpflag == 0]
    putoptions = options[cpflag == 0, :]
    putToGetIdx   = (np.abs(putstrikes - spot) == np.min(np.abs(putstrikes - spot)))
    
    ATMCall = calloptions[callToGetIdx, :]
    ATMPut  = putoptions[putToGetIdx, :]
    
    #ATMOptions = options[toGetIdx, :] #grab the options closest
    #ATMCalls   = ATMOptions[(cpflag[toGetIdx] == 1), :] #separate calls
    #ATMPuts    = ATMOptions[(cpflag[toGetIdx] == 0), :] #separate puts
    
    return ATMCall, ATMPut

#match option across trading days
def matchOption(option, optionsToMatch, matchClosest = False):
    #grab unique features of option
    exdate = option[0,1] 
    cpflag = option[0,2]
    strike = option[0,3]
    
    #construct booleans
    isRightExp    = (optionsToMatch[:, 1] == exdate)
    isRightType   = (optionsToMatch[:, 2] == cpflag)
    isRightStrike = (optionsToMatch[:, 3] == strike)
    
    if matchClosest == True and np.sum(isRightStrike) == 0:
        matchStrikes  = optionsToMatch[:, 3]
        strikeDiff    = np.abs(matchStrikes - strike)
        closestStrike = matchStrikes[(strikeDiff == np.min(strikeDiff))]
        isRightStrike = (optionsToMatch[:, 3] == closestStrike) 
                                     
        
    #combine to one boolean and extract
    matchIndex  = isRightExp * isRightType * isRightStrike
    matchOption = optionsToMatch[matchIndex, :]
    
    return matchOption


#Trading Strategy
#Set params for loop
nDays      = np.size(UniqueDates)
rollDay    = 7 #days before expiration
currentPos = 0 #Initialize no position
lag        = 0

#Preallocate
isRollover  = np.zeros((nDays, )) #
disapearingOption = []

disapearingOptionFront_un = []
disapearingOptionBack_un = []



callReturns = np.zeros((nDays, ))
putReturns  = np.zeros((nDays, ))
uncondCallReturns = np.zeros((nDays, ))
uncondPutReturns = np.zeros((nDays, ))

#Preallocate for traded options track record
currentDayCallsTraded = np.zeros((nDays, np.size(OptionDataArr, 1)))
nextDayCallsTraded    = np.zeros((nDays, np.size(OptionDataArr, 1)))

currentDayPutsTraded  = np.zeros((nDays, np.size(OptionDataArr, 1)))
nextDayPutsTraded     = np.zeros((nDays, np.size(OptionDataArr, 1)))


for i in np.arange(0, nDays - lag - 1):
    #Get data from right day        
    day     = UniqueDates[i + lag] #grab day
    nextDay = UniqueDates[i + lag + 1]
    isRightDay      = (OptionDates == day) #right day boolean
    rightDayOptions = OptionDataArr[isRightDay, :] #grab options for trading day
    
    #Grab next day options
    isNextDay      = (OptionDates == nextDay)
    nextDayOptions = OptionDataArr[isNextDay, :]
    
    expirations    = rightDayOptions[:, 1] 
    frontExpiry    = np.unique(expirations)[0] #grab front month expiration
    backExpiry     = np.unique(expirations)[1] #grab back month expiration
    daysToMatFront = frontExpiry - day #days to maturity of front contract
    
    #Front, back and ATM booleans
    isFrontExp = (expirations == frontExpiry)
    isBackExp  = (expirations == backExpiry)
    isATM      = (rightDayOptions[:, -1] == 1)
    
    
    
    #Unconditional strategy
    if daysToMatFront > rollDay: #trade front    
        frontToTrade_un = rightDayOptions[isFrontExp * isATM, :] #Front ATM Options 
        if len(frontToTrade_un) > 0:
            currentDayCall_un, currentDayPut_un = getATMOptions(frontToTrade_un)
            
            #Grab next day options
            nextDayCall_un = matchOption(currentDayCall_un, nextDayOptions, matchClosest = True)
            nextDayPut_un  = matchOption(currentDayPut_un, nextDayOptions, matchClosest = True)
        
            if len(nextDayCall_un) > 0:
                uncondCallReturns[i + lag +1] = nextDayCall_un[0, 15] / currentDayCall_un[0, 15] - 1
            else:
                disapearingOptionFront_un.append([nextDay, "next"])
        
            if len(nextDayPut_un) > 0:
                uncondPutReturns[i + lag +1] = nextDayPut_un[0, 15] / currentDayPut_un[0, 15] - 1
            else:
                disapearingOptionFront_un.append([nextDay, "next"])    
                
        else:
            disapearingOptionFront_un.append([day, "current"])
           
    else: #trade back
        backToTrade_un   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
        if len(backToTrade_un) > 0:
            currentDayCall_un, currentDayPut_un = getATMOptions(backToTrade_un)
            #Grab next day options
            nextDayCall_un = matchOption(currentDayCall_un, nextDayOptions, matchClosest = True)
            nextDayPut_un  = matchOption(currentDayPut_un, nextDayOptions, matchClosest = True)
        
            if len(nextDayCall_un) > 0:
                uncondCallReturns[i + lag +1] = nextDayCall_un[0, 15] / currentDayCall_un[0, 15] - 1
            else:
                disapearingOptionBack_un.append([nextDay, "next"])
    
            if len(nextDayPut_un) > 0:
                uncondPutReturns[i + lag +1] = nextDayPut_un[0, 15] / currentDayPut_un[0, 15] - 1
            else:
                disapearingOptionBack_un.append([nextDay, "next"])    
        else:
            disapearingOptionBack_un.append([day, "current"]) 
            
        
    # Gamma Timed Strategy
    gammaSignal = (netGamma[i] < 0)
    if gammaSignal == True and currentPos == 0: #New Position to be opened
      
        if daysToMatFront > rollDay: #trade front month 
           frontToTrade = rightDayOptions[isFrontExp * isATM, :] #Front ATM Options 
           if len(frontToTrade) > 0:
               currentDayCall, currentDayPut = getATMOptions(frontToTrade)
               
               #Grab next day options
               nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
               nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
        
               #Call leg returns
               if len(nextDayCall) > 0:    
                   callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                   
                   #Store information on calls traded
                   currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                   nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
               else:
                   disapearingOption.append([nextDay, i + lag + 1, "front", "next", "call"])
               
               #Put leg returns
               if len(nextDayPut) > 0:
                   putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                   
                   #Store information on puts traded
                   currentDayPutsTraded[i + lag, :]     = currentDayPut.reshape(22,)
                   nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
               else:
                   disapearingOption.append([nextDay, i + lag + 1, "front", "next", "put"])
               
               currentPos = 1 #add current position fla
               
           else:
              disapearingOption.append([day, i + lag, "front", "current", "no ATM options"])
           
            
    
        else: #trade back month
           isRollover[i + lag] = 1 #roll boolean   
           backToTrade   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
           
           if len(backToTrade) > 0:
               currentDayCall, currentDayPut = getATMOptions(backToTrade)
           else:
               disapearingOption.append([day, i + lag, "back", "current", "no ATM Options"])
               currentPos = 0
               continue
           
           #Grab next day options
           nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
           nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
    
           #Call leg returns
           if len(nextDayCall) > 0:    
               callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
               
               #Store information on calls traded
               currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
               nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
           else:
               disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
           
           #Put leg returns
           if len(nextDayPut) > 0:
               putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
               
               #Store information on puts traded
               currentDayPutsTraded[i + lag , :]    = currentDayPut.reshape(22,)
               nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
           else:
               disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
           
           currentPos = 1 #add current position flag
           
                   
 
    if gammaSignal == True and currentPos == 1: #Keep the same position
        
        #option shifts one day to become most recent 
        if len(nextDayCall) > 0 and len(nextDayPut) > 0:
            currentDayCall = nextDayCall
            currentDayPut  = nextDayPut  
        else:
            currentPos = 0
            disapearingOption.append([day, i + lag, "existing", "shift", "existing position gone"])
            continue #jump to next iteration 
            
    
        #Grab next day options
        nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
        nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
       
        #Call leg returns
        if len(nextDayCall) > 0:    
            callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
            
            #Store information on calls traded
            currentDayCallsTraded[i + lag, :]  = currentDayCall.reshape(22,)
            nextDayCallsTraded[i + lag, :]     = nextDayCall.reshape(22,)
        else:
            disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
        
        #Put leg returns
        if len(nextDayPut) > 0:
            putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
            
            #Store information on puts traded
            currentDayPutsTraded[i + lag + 1, :] = currentDayPut.reshape(22,)
            nextDayPutsTraded[i + lag + 1, :] = nextDayPut.reshape(22,)
        else:
            disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
        
        currentPos = 1 #add current position flag
       
    elif gammaSignal == False:
        currentPos = 0 #No position if gamma signal is not there
    
    

    
#Straddle returns        
straddleReturns = callReturns + putReturns
uncondStraddleReturns = uncondCallReturns + uncondPutReturns

#VIX Returns
#Returns of underlying 
#### TO BE REPLACED BY TOTAL RETURN INDEX RETURNS
underlyingTotalReturns   = Underlying[1:] / Underlying[0:-1] - 1
underlyingTotalReturns   = np.concatenate((np.zeros((1,1)), underlyingTotalReturns.reshape(-1, 1)), axis = 0)
underlyingXsReturns      = underlyingTotalReturns[:, 0] - RfDaily #excess returns

#Scale Returns
scale = 0.01
callReturnsScaled     = callReturns*scale
putReturnsScaled      = putReturns*scale
straddleReturnsScaled = straddleReturns*scale

uncondCallReturnsScaled = uncondCallReturns*scale
uncondPutReturnsScaled  = uncondPutReturns*scale
uncondStraddleReturnsScaled = uncondStraddleReturns*scale


#Overlay
overlayReturns = (1 - scale)*underlyingXsReturns + straddleReturnsScaled


#Cumulative Returns
cumCallReturns     = np.cumprod(1 + callReturnsScaled)
cumPutReturns      = np.cumprod(1 + putReturnsScaled)
cumStraddleReturns = np.cumprod(1 + straddleReturnsScaled)
cumOverlayReturns  = np.cumprod(1 + overlayReturns)
cumUnderlyingReturns     = np.cumprod(1 + underlyingXsReturns)
cumUncondCallReturns     = np.cumprod(1 + uncondCallReturnsScaled)
cumUncondPutReturns      = np.cumprod(1 + uncondPutReturnsScaled)
cumUncondStraddleReturns = np.cumprod(1 + uncondStraddleReturnsScaled)


dates4fig = pd.to_datetime(AggregateDates, format = '%Y%m%d')


#Plot Equity Lines
plt.figure()
plt.plot(dates4fig, cumCallReturns, color = "blue", label = "Calls Only Gamma Timed")
plt.plot(dates4fig, cumPutReturns, color = "red", label = "Puts Only Gamma Timed")
plt.plot(dates4fig, cumStraddleReturns, c = "black", label = "Straddle Gamma Timed")
plt.legend()
plt.title("NAV for Gamma Timed Option Strategies")
plt.ylabel("Cumulative Excess Returns")

plt.figure()
plt.plot(dates4fig, cumUncondCallReturns, color = "blue", label = "Calls Only")
plt.plot(dates4fig, cumUncondPutReturns, color = "red", label = "Puts Only")
plt.plot(dates4fig, cumUncondStraddleReturns, c = "black", label = "Straddles")
plt.legend()
plt.title("NAV for Unconditional Option Strategies")
plt.ylabel("Cumulative Excess Returns")

plt.figure()
plt.plot(dates4fig, cumOverlayReturns, color = "blue", label = "Overlay Strategy")
plt.plot(dates4fig, cumUnderlyingReturns, color = "black", label = "Passive Strategy")
plt.legend()
plt.title("Overlay Strategy vs Passive")
plt.ylabel("Cumulative Excess Returns")

#Investigate errors

checkDate = disapearingOptionFront_un[0]
checkDateOptions = OptionDataArr[(OptionDates == checkDate[0]), :]


























