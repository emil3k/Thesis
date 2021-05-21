# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:16:18 2021

@author: ekblo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:00:15 2021

@author: ekblo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:36:12 2021

@author: ekblo
"""
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
lag        = 1
lookback = 500

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

daysToMatFrontTrackerLong = np.zeros((nDays,))
daysToMatFrontTrackerShort = np.zeros((nDays,))
turnoverLong = np.zeros((nDays,))
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
    
    
    #Get days to expiry
    expiryVsToday = np.array([day, frontExpiry])
    expiryVsTodayDateTime = pd.to_datetime(expiryVsToday, format = '%Y%m%d')
    daycount        = bt.dayCount(expiryVsTodayDateTime)
    daysToMatFront = daycount[-1]
    
    #Front, back and ATM booleans
    isFrontExp = (expirations == frontExpiry)
    isBackExp  = (expirations == backExpiry)
    isATM      = (rightDayOptions[:, -1] == 1)
    
    daysToMatFrontTrackerLong[i] = daysToMatFront
    
    
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
            
        
    # Gamma Timed Long Strategy
    
    #Only trade if complete lookback period
    if i < lookback:
        continue
    
    lookbackGamma    = netGamma[i - lookback:i]
    negLookbackGamma = lookbackGamma[lookbackGamma < 0]
    posLookbackGamma = lookbackGamma[lookbackGamma > 0]
    negThreshold = np.median(negLookbackGamma)
    posThreshold = np.median(posLookbackGamma)
    
    gammaSignal = (netGamma[i] < negThreshold)
    if gammaSignal == True: #Trading Signal Observed
        if currentPos == 0: #Position is not open, open it
                   
            #Check days to maturity for front month
            if daysToMatFront > rollDay: #If higher than 7 (rollDay), trade front month
               
               frontToTrade = rightDayOptions[isFrontExp * isATM, :] #Front ATM Options  
           
               #Check if the options exist
               if len(frontToTrade) > 0: #If options exist, find ATM options to trade
                   currentDayCall, currentDayPut = getATMOptions(frontToTrade) #Grab ATM options
                    
                   #Grab next day options
                   nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                   nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
         
                   #Check if Call Options disappear the next day
                   if len(nextDayCall) > 0: #If they exist, compute returns     
                       callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                       #Store information on calls traded
                       #currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                       #nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                       currentPos = 1 #add current position flag
                       frontPos   = 1 #Signal trade is in front
                       backPos    = 0 #Signal trade is not in back month
                       turnoverLong[i] = 1
                   
                   else: #If not, note error
                       disapearingOption.append([nextDay, i + lag + 1, "front", "next", "call"])
                       currentPos = 0
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                   
                   #Check if Put Options disappear the next day
                   if len(nextDayPut) > 0: #If they exist, compute returns
                       putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                            
                       #Store information on puts traded
                       #currentDayPutsTraded[i + lag, :]     = currentDayPut.reshape(22,)
                       #nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                        
                       currentPos = 1 #add current position flag
                       frontPos   = 1 #Signal trade is in front
                       backPos    = 0 #Signal trade is not in back month
                   else: #if not, store error
                       disapearingOption.append([nextDay, i + lag + 1, "front", "next", "put"])
                       currentPos = 0
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                   
               #Options does not exist for given day, do not open any position 
               #Note error
               else:
                  disapearingOption.append([day, i + lag, "front", "current", "no ATM options"])
                  #remove all positions if no front available to trade
                  currentPos = 0
                  frontPos   = 0
                  backPos    = 0
            
            #Position to be opened in Backmonth
            #If not higher than 7 (rollday), initiate position in back-month
            else: 
               isRollover[i + lag] = 1 #roll boolean   
               backToTrade   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
               
               #Check if back month options exist
               if len(backToTrade) > 0:
                   currentDayCall, currentDayPut = getATMOptions(backToTrade)
               
                   #Grab next day options
                   nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                   nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
            
                   #Check if call options disappear
                   if len(nextDayCall) > 0: #If they exist, compute returns 
                       callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                       
                       #Store information on calls traded
                       #currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                       #nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                       currentPos = 1 #add current position flag
                       backPos  = 1 #Signal trade is in back month
                       frontPos = 0 #Signal trade is not in front
                       turnoverLong[i] = 1
                   else: #If not, add error
                       disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                    
                   #Check if Put options disappear
                   if len(nextDayPut) > 0: #If they exist, compute returns
                       putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                       
                       #Store information on puts traded
                       #currentDayPutsTraded[i + lag , :]    = currentDayPut.reshape(22,)
                       #nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                       currentPos = 1 #add current position flag
                       backPos  = 1 #Signal trade is in back month
                       frontPos = 0 #Signal trade is not in front
                   else:
                       disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
               #Options does not exist for given day, do not open any position 
               #Note error
               else:
                  disapearingOption.append([day, i + lag, "back", "current", "no ATM options"])
                  #remove all positions if no front available to trade
                  currentPos = 0
                  frontPos   = 0
                  backPos    = 0
                   
                   
        #Current position already exists
        else: #Keep the same position
            
            #Check if current position is front position
            if frontPos == 1: 
                
                #If front position, check if needs rolling
                if daysToMatFront < rollDay: #Front position need to be rolled
                    backToTrade   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
               
                    if len(backToTrade) > 0:
                        currentDayCall, currentDayPut = getATMOptions(backToTrade)
                    else:
                        disapearingOption.append([day, i + lag, "back", "current", "no ATM Options"])
                        currentPos = 0
                        frontPos   = 0
                        backPos    = 0 
                        continue
                    
                    #Grab next day options
                    nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                    nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
             
                    #Call leg returns
                    if len(nextDayCall) > 0:    
                        callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                        #Store information on calls traded
                        #currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                        #nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                        currentPos = 1 #add current position flag
                        backPos    = 1 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front
                        turnoverLong[i]   = 2 #rollover
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                        currentPos = 0 #add current position flag
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    #Put leg returns
                    if len(nextDayPut) > 0:
                        putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                        
                        #Store information on puts traded
                        #currentDayPutsTraded[i + lag , :]    = currentDayPut.reshape(22,)
                        #nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                        currentPos = 1 #add current position flag
                        backPos    = 1 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                        currentPos = 0 #add current position flag
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    
                else: #No need to roll
                    #option shifts one day to become most recent 
                    #Check of options exist
                    if len(nextDayCall) > 0 and len(nextDayPut) > 0:
                        currentDayCall = nextDayCall
                        currentDayPut  = nextDayPut  
                    
                    else: #if not set position to zero and jump to next iteration
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                        disapearingOption.append([day, i + lag, "existing", "shift", "existing position gone"])
                        continue #jump to next iteration 
                        
                
                    #Grab next day options
                    nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                    nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
                   
                    #Call leg returns
                    if len(nextDayCall) > 0:    
                        callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                        #Store information on calls traded
                        #currentDayCallsTraded[i + lag, :]  = currentDayCall.reshape(22,)
                        #nextDayCallsTraded[i + lag + 1, :]     = nextDayCall.reshape(22,)
                        currentPos = 1 #add current position flag
                        frontPos   = 1
                        backPos    = 0
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    
                    #Put leg returns
                    if len(nextDayPut) > 0:
                        putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                        
                        #Store information on puts traded
                        #currentDayPutsTraded[i + lag, :] = currentDayPut.reshape(22,)
                        #nextDayPutsTraded[i + lag + 1, :] = nextDayPut.reshape(22,)
                        currentPos = 1 #add current position flag
                        frontPos   = 1
                        backPos    = 0
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front      
                
                
            #Current position is backmonth position     
            else:          
                #option shifts one day to become most recent 
                #Check of options exist
                if len(nextDayCall) > 0 and len(nextDayPut) > 0:
                    currentDayCall = nextDayCall
                    currentDayPut  = nextDayPut  
                
                else: #if not set position to zero and jump to next iteration
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front    
                    disapearingOption.append([day, i + lag, "existing", "shift", "existing position gone"])
                    continue #jump to next iteration 
             
                #Grab next day options
                nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
               
                #Call leg returns
                if len(nextDayCall) > 0:    
                    callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                    
                    #Store information on calls traded
                    #currentDayCallsTraded[i + lag, :]  = currentDayCall.reshape(22,)
                    #nextDayCallsTraded[i + lag + 1, :]     = nextDayCall.reshape(22,)
                    currentPos = 1 #add current position flag
                    frontPos   = int((currentDayCall[:, 1] == frontExpiry)*1) #Set front flag if expiration is equal to front exp 
                    backPos    = int((currentDayCall[:, 1] == backExpiry)*1)  #Set back flag if expiration is equal to back exp
                else:
                    disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front    
                
                #Put leg returns
                if len(nextDayPut) > 0:
                    putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                    
                    #Store information on puts traded
                    #currentDayPutsTraded[i + lag, :] = currentDayPut.reshape(22,)
                    #nextDayPutsTraded[i + lag + 1, :] = nextDayPut.reshape(22,)
                    currentPos = 1 #add current position flag
                    frontPos   = int((currentDayCall[:, 1] == frontExpiry)*1) #Set front flag if expiration is equal to front exp 
                    backPos    = int((currentDayCall[:, 1] == backExpiry)*1)  #Set back flag if expiration is equal to back exp
                else:
                    disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front   
              
 

    #No gamma signal       
    else:
        if currentPos == 1:
            turnoverLong[i] = -1
   
        currentPos = 0 #No position if gamma signal is not there
        backPos    = 0 #Signal trade is in back month
        frontPos   = 0 #Signal trade is not in front    
    
#Straddle returns        
straddleReturnsLong = callReturns[lookback:] + putReturns[lookback:]
uncondStraddleReturnsLong = uncondCallReturns[lookback:] + uncondPutReturns[lookback:]


#Trading Strategy
#Set params for loop
nDays      = np.size(UniqueDates)
rollDay    = 7 #days before expiration
currentPos = 0 #Initialize no position
lag        = 1

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
#currentDayCallsTraded = np.zeros((nDays, np.size(OptionDataArr, 1)))
#nextDayCallsTraded    = np.zeros((nDays, np.size(OptionDataArr, 1)))

#currentDayPutsTraded  = np.zeros((nDays, np.size(OptionDataArr, 1)))
#nextDayPutsTraded     = np.zeros((nDays, np.size(OptionDataArr, 1)))




turnoverShort = np.zeros((nDays, ))

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
    
    #Get days to expiry
    expiryVsToday = np.array([day, frontExpiry])
    expiryVsTodayDateTime = pd.to_datetime(expiryVsToday, format = '%Y%m%d')
    daycount        = bt.dayCount(expiryVsTodayDateTime)
    daysToMatFront = daycount[-1]
    
    
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
            
    # Gamma Timed Short Strategy
    gammaSignal = (netGamma[i] > posThreshold)
    if gammaSignal == True: #Trading Signal Observed
        if currentPos == 0: #Position is not open, open it
                   
            #Check days to maturity for front month
            if daysToMatFront > rollDay: #If higher than 7 (rollDay), trade front month
               
               frontToTrade = rightDayOptions[isFrontExp * isATM, :] #Front ATM Options  
           
               #Check if the options exist
               if len(frontToTrade) > 0: #If options exist, find ATM options to trade
                   currentDayCall, currentDayPut = getATMOptions(frontToTrade) #Grab ATM options
                    
                   #Grab next day options
                   nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                   nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
         
                   #Check if Call Options disappear the next day
                   if len(nextDayCall) > 0: #If they exist, compute returns     
                       callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                       #Store information on calls traded
                       # currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                       # nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                       currentPos = 1 #add current position flag
                       frontPos   = 1 #Signal trade is in front
                       backPos    = 0 #Signal trade is not in back month
                       turnoverShort[i] = 1
                   else: #If not, note error
                       disapearingOption.append([nextDay, i + lag + 1, "front", "next", "call"])
                       currentPos = 0
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                    #Check if Put Options disappear the next day
                   if len(nextDayPut) > 0: #If they exist, compute returns
                       putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                            
                       #Store information on puts traded
                       # currentDayPutsTraded[i + lag, :]     = currentDayPut.reshape(22,)
                       # nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                        
                       currentPos = 1 #add current position flag
                       frontPos   = 1 #Signal trade is in front
                       backPos    = 0 #Signal trade is not in back month
                       turnoverShort[i] = 1
                   else: #if not, store error
                       disapearingOption.append([nextDay, i + lag + 1, "front", "next", "put"])
                       currentPos = 0
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                   
               #Options does not exist for given day, do not open any position 
               #Note error
               else:
                  disapearingOption.append([day, i + lag, "front", "current", "no ATM options"])
                  #remove all positions if no front available to trade
                  currentPos = 0
                  frontPos   = 0
                  backPos    = 0
            
            #Position to be opened in Backmonth
            #If not higher than 7 (rollday), initiate position in back-month
            else: 
               isRollover[i + lag] = 1 #roll boolean   
               backToTrade   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
               
               #Check if back month options exist
               if len(backToTrade) > 0:
                   currentDayCall, currentDayPut = getATMOptions(backToTrade)
               
                   #Grab next day options
                   nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                   nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
            
                   #Check if call options disappear
                   if len(nextDayCall) > 0: #If they exist, compute returns 
                       callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                       
                       #Store information on calls traded
                       # currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                       # nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                       currentPos = 1 #add current position flag
                       backPos  = 1 #Signal trade is in back month
                       frontPos = 0 #Signal trade is not in front
                       turnoverShort[i] = 1
                   else: #If not, add error
                       disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
                    
                   #Check if Put options disappear
                   if len(nextDayPut) > 0: #If they exist, compute returns
                       putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                       
                       #Store information on puts traded
                       # currentDayPutsTraded[i + lag , :]    = currentDayPut.reshape(22,)
                       # nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                       currentPos = 1 #add current position flag
                       backPos  = 1 #Signal trade is in back month
                       frontPos = 0 #Signal trade is not in front
                   else:
                       disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                       currentPos = 0 #add current position flag
                       backPos    = 0 #Signal trade is in back month
                       frontPos   = 0 #Signal trade is not in front    
               #Options does not exist for given day, do not open any position 
               #Note error
               else:
                  disapearingOption.append([day, i + lag, "back", "current", "no ATM options"])
                  #remove all positions if no front available to trade
                  currentPos = 0
                  frontPos   = 0
                  backPos    = 0
                   
                   
        #Current position already exists
        else: #Keep the same position
            
            #Check if current position is front position
            if frontPos == 1: 
                
                #If front position, check if needs rolling
                if daysToMatFront < rollDay: #Front position need to be rolled
                    backToTrade   = rightDayOptions[isBackExp * isATM, :]  #Back ATM Options
               
                    if len(backToTrade) > 0:
                        currentDayCall, currentDayPut = getATMOptions(backToTrade)
                    else:
                        disapearingOption.append([day, i + lag, "back", "current", "no ATM Options"])
                        currentPos = 0
                        frontPos   = 0
                        backPos    = 0 
                        continue
                    
                    #Grab next day options
                    nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                    nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
             
                    #Call leg returns
                    if len(nextDayCall) > 0:    
                        callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                        #Store information on calls traded
                        # currentDayCallsTraded[i + lag, :]   = currentDayCall.reshape(22,)
                        # nextDayCallsTraded[i + lag + 1, :]  = nextDayCall.reshape(22,)
                        currentPos = 1 #add current position flag
                        backPos    = 1 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front
                        turnoverShort[i] = 2
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                        currentPos = 0 #add current position flag
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    #Put leg returns
                    if len(nextDayPut) > 0:
                        putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                        
                        #Store information on puts traded
                        # currentDayPutsTraded[i + lag , :]    = currentDayPut.reshape(22,)
                        # nextDayPutsTraded[i + lag + 1, :]    = nextDayPut.reshape(22,)
                        currentPos = 1 #add current position flag
                        backPos    = 1 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "back", "next", "call"])
                        currentPos = 0 #add current position flag
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    
                else: #No need to roll
                    #option shifts one day to become most recent 
                    #Check of options exist
                    if len(nextDayCall) > 0 and len(nextDayPut) > 0:
                        currentDayCall = nextDayCall
                        currentDayPut  = nextDayPut  
                    
                    else: #if not set position to zero and jump to next iteration
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                        disapearingOption.append([day, i + lag, "existing", "shift", "existing position gone"])
                        continue #jump to next iteration 
                        
                
                    #Grab next day options
                    nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                    nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
                   
                    #Call leg returns
                    if len(nextDayCall) > 0:    
                        callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                        
                        #Store information on calls traded
                        # currentDayCallsTraded[i + lag, :]  = currentDayCall.reshape(22,)
                        # nextDayCallsTraded[i + lag + 1, :]     = nextDayCall.reshape(22,)
                        currentPos = 1 #add current position flag
                        frontPos   = 1
                        backPos    = 0
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front    
                    
                    #Put leg returns
                    if len(nextDayPut) > 0:
                        putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                        
                        #Store information on puts traded
                        # currentDayPutsTraded[i + lag, :] = currentDayPut.reshape(22,)
                        # nextDayPutsTraded[i + lag + 1, :] = nextDayPut.reshape(22,)
                        currentPos = 1 #add current position flag
                        frontPos   = 1
                        backPos    = 0
                    else:
                        disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                        currentPos = 0
                        backPos    = 0 #Signal trade is in back month
                        frontPos   = 0 #Signal trade is not in front      
                
                
            #Current position is backmonth position     
            else:          
                #option shifts one day to become most recent 
                #Check of options exist
                if len(nextDayCall) > 0 and len(nextDayPut) > 0:
                    currentDayCall = nextDayCall
                    currentDayPut  = nextDayPut  
                
                else: #if not set position to zero and jump to next iteration
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front    
                    disapearingOption.append([day, i + lag, "existing", "shift", "existing position gone"])
                    continue #jump to next iteration 
             
                #Grab next day options
                nextDayCall = matchOption(currentDayCall, nextDayOptions, matchClosest = True)
                nextDayPut  = matchOption(currentDayPut, nextDayOptions, matchClosest = True)
               
                #Call leg returns
                if len(nextDayCall) > 0:    
                    callReturns[i + lag + 1] = nextDayCall[0, 15] / currentDayCall[0, 15] - 1
                    
                    #Store information on calls traded
                    # currentDayCallsTraded[i + lag, :]  = currentDayCall.reshape(22,)
                    # nextDayCallsTraded[i + lag + 1, :]     = nextDayCall.reshape(22,)
                    currentPos = 1 #add current position flag
                    frontPos   = int((currentDayCall[:, 1] == frontExpiry)*1) #Set front flag if expiration is equal to front exp 
                    backPos    = int((currentDayCall[:, 1] == backExpiry)*1)  #Set back flag if expiration is equal to back exp
                else:
                    disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front    
                
                #Put leg returns
                if len(nextDayPut) > 0:
                    putReturns[i + lag + 1]  = nextDayPut[0, 15] / currentDayPut[0, 15] - 1
                    
                    #Store information on puts traded
                    # currentDayPutsTraded[i + lag, :] = currentDayPut.reshape(22,)
                    # nextDayPutsTraded[i + lag + 1, :] = nextDayPut.reshape(22,)
                    currentPos = 1 #add current position flag
                    frontPos   = int((currentDayCall[:, 1] == frontExpiry)*1) #Set front flag if expiration is equal to front exp 
                    backPos    = int((currentDayCall[:, 1] == backExpiry)*1)  #Set back flag if expiration is equal to back exp
                else:
                    disapearingOption.append([nextDay, i + lag + 1, "existing", "next", "call"])
                    currentPos = 0
                    backPos    = 0 #Signal trade is in back month
                    frontPos   = 0 #Signal trade is not in front   
              
 

    #No gamma signal       
    else:
        if currentPos == 1:
            turnoverShort[i] = -1
        
        currentPos = 0 #No position if gamma signal is not there
        backPos    = 0 #Signal trade is in back month
        frontPos   = 0 #Signal trade is not in front    
    
    

    
#Straddle returns        
straddleReturnsShort       = (-1)*(callReturns[lookback:] + putReturns[lookback:])
uncondStraddleReturnsShort = (-1)*(uncondCallReturns[lookback:] + uncondPutReturns[lookback:])
turnoverLong               = turnoverLong[lookback:]
turnoverShort              = turnoverShort[lookback:]

totalTurnover     = np.abs(turnoverLong) + np.abs(turnoverShort)

LSStraddleReturns = straddleReturnsLong + straddleReturnsShort



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
straddleReturnsLongScaled  = straddleReturnsLong*scale
straddleReturnsShortScaled = straddleReturnsShort*scale
straddleReturnsLSScaled    = LSStraddleReturns*scale

uncondCallReturnsScaled          = uncondCallReturns*scale
uncondPutReturnsScaled           = uncondPutReturns*scale
uncondStraddleReturnsShortScaled = uncondStraddleReturnsShort*scale
uncondStraddleReturnsLongScaled  = uncondStraddleReturnsLong*scale

#Transaction cost
c = 0.001
straddleReturnsLongTC  = straddleReturnsLongScaled - np.abs(turnoverLong)*c
straddleReturnsShortTC = straddleReturnsShortScaled - np.abs(turnoverShort)*c
straddleReturnsLSTC    = straddleReturnsLSScaled - totalTurnover*c

#Cumulative Returns
cumCallReturns          =  np.cumprod(1 + callReturnsScaled)
cumPutReturns           =  np.cumprod(1 + putReturnsScaled)
cumStraddleReturnsLong  =  np.cumprod(1 + straddleReturnsLongScaled)
cumStraddleReturnsShort =  np.cumprod(1 + straddleReturnsShortScaled)
cumStraddleReturnsLS    =  np.cumprod(1 + straddleReturnsLSScaled)

cumStraddleReturnsLongTC  =  np.cumprod(1 + straddleReturnsLongTC)
cumStraddleReturnsShortTC =  np.cumprod(1 + straddleReturnsShortTC)
cumStraddleReturnsLSTC    =  np.cumprod(1 + straddleReturnsLSTC)



#cumOverlayReturns  = np.cumprod(1 + overlayReturns)
cumUnderlyingReturns     = np.cumprod(1 + underlyingXsReturns)
cumUncondCallReturns     = np.cumprod(1 + uncondCallReturnsScaled)
cumUncondPutReturns      = np.cumprod(1 + uncondPutReturnsScaled)
cumUncondStraddleReturnsShort = np.cumprod(1 + uncondStraddleReturnsShortScaled)
cumUncondStraddleReturnsLong = np.cumprod(1 + uncondStraddleReturnsLongScaled)


dates4fig = pd.to_datetime(AggregateDates, format = '%Y%m%d')
dates4fig = dates4fig[lookback:]

#Plot Equity Lines
plt.figure()
plt.plot(dates4fig, cumStraddleReturnsLong,  color = prefColor, alpha = 0.8, label = "Long Only Gamma-Timed")
plt.plot(dates4fig, cumStraddleReturnsShort, color = "red",     alpha = 0.8, label = "Short Only Gamma-Timed")
plt.plot(dates4fig, cumStraddleReturnsLS,    color = "black",   alpha = 1.0, label = "Long-Short Gamma-Timed")
plt.title("Gamma Timed Index Straddle Strategies, " + UnderlyingTicker)
plt.ylabel("Cumulative Excess Returns")
plt.legend()

#Plot Equity Lines
plt.figure()
plt.plot(dates4fig, cumUncondStraddleReturnsShort,  color = "red", alpha = 0.8, label = "Short Straddle")
plt.plot(dates4fig, cumStraddleReturnsShort, color = prefColor,     alpha = 0.8, label = "Short Straddle Gamma-Timed")
plt.title("Short Straddle Strategies, " + UnderlyingTicker)
plt.ylabel("Cumulative Excess Returns")
plt.legend()

#Plot Equity Lines
plt.figure()
plt.plot(dates4fig, cumUncondStraddleReturnsLong,  color = "red", alpha = 0.8, label = "Long Straddle")
plt.plot(dates4fig, cumStraddleReturnsLong, color = prefColor, alpha = 0.8, label = "Long Straddle Gamma-Timed")
plt.title("Long Straddle Strategies, " + UnderlyingTicker)
plt.ylabel("Cumulative Excess Returns")
plt.legend()

#Plot Equity Lines
plt.figure()
plt.plot(dates4fig, cumStraddleReturnsLong,  color = prefColor, alpha = 0.8, label = "Long Only Gamma-Timed")
plt.plot(dates4fig, cumStraddleReturnsLongTC,  color = prefColor, linestyle = "--",  alpha = 0.8, label = "Long Only Gamma-Timed")

plt.plot(dates4fig, cumStraddleReturnsShort, color = "red",     alpha = 0.8, label = "Short Only Gamma-Timed")
plt.plot(dates4fig, cumStraddleReturnsShortTC, color = "red", linestyle = "--", alpha = 0.8, label = "Short Only Gamma-Timed")

plt.plot(dates4fig, cumStraddleReturnsLS,    color = "black",   alpha = 1.0, label = "Long-Short Gamma-Timed")
plt.plot(dates4fig, cumStraddleReturnsLSTC,    color = "black", linestyle = "--",  alpha = 1.0, label = "Long-Short Gamma-Timed")
plt.title("Gamma Timed Index Straddle Strategies, " + UnderlyingTicker)
plt.ylabel("Cumulative Excess Returns")
plt.legend()




#Investigate errors

checkDate = disapearingOptionFront_un[0]
checkDateOptions = OptionDataArr[(OptionDates == checkDate[0]), :]


#Compute performance
callPerformance     = bt.ComputePerformance(straddleReturnsLongScaled, RfDaily, 0, 255)
putPerformance      = bt.ComputePerformance(straddleReturnsShortScaled, RfDaily, 0, 255)
straddlePerformance = bt.ComputePerformance(straddleReturnsLSScaled, RfDaily, 0, 255)

#Construct Latex Table
def constructPerformanceDf(performanceList, colNames, to_latex = True):
    legend      = performanceList[0][0]
    nStrategies = len(performanceList)
    performanceMat = np.zeros((len(legend), nStrategies))
    
    for i in np.arange(0, nStrategies):
        performanceMat[:, i]  = performanceList[i][1].reshape(-1,)
            
    performanceMat = np.vstack(performanceMat).astype(np.float)
    performanceMat = np.round(performanceMat, decimals = 2) 
    performanceMat  = np.concatenate((legend.reshape(-1, 1), performanceMat), axis = 1)
    performanceDf   = pd.DataFrame.from_records(performanceMat, columns = colNames)

    if to_latex == True:
        print(performanceDf.to_latex(index=False))
        
    return performanceDf


colNames = np.array(["statistic", "Long Only", "Short Only", "Long-Short"])
perfList = [callPerformance, putPerformance, straddlePerformance]
test     = constructPerformanceDf(perfList, colNames = colNames, to_latex = True)


sys.exit()



day = 19981012
dayOptions = OptionDataArr[OptionDates == day, :]











