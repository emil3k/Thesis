# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:34:19 2020

@author: ekblo
"""
#Split option data by Put and Call
cp_flag    = OptionData["cp_flag"].to_numpy()
call_flag  = (cp_flag == "C") #Create call_flag boolean
CallData   = OptionDataTr[call_flag]
PutData    = OptionDataTr[~call_flag]
CallDates  = OptionDates[call_flag]
PutDates   = OptionDates[~call_flag]

#Get contract change indices
def getContractTypeChanges(OptionStrikes):
    nOptions = np.size(OptionStrikes, 0)
    isNewContract = np.zeros((nOptions, 1))

    for i in np.arange(0, nOptions):
        if OptionStrikes[i] < OptionStrikes[i + 1]:
            isNewContract[i + 1] = 1
        
    newContractList = np.nonzero(isNewContract)

    return newContractList

## TEST FUNCTION!!!!

##Need to adjust loop below to go over each contract expiration style
##Use function above

#OTM / ATM flag
nDays   = np.size(UniqueDates)

ATMCall_flag  = np.zeros((1, 1))
ATMPut_flag   = np.zeros((1, 1))
OTMCall_flag  = np.zeros((1, 1))
UnderlyingVec = np.zeros((1, 1))

for i in np.arange(0, nDays):

    CurrentDate       = UnderlyingDates[i] #Grab current date
    CurrentUnderlying = UnderlyingPrice[i] #Grab underlying price
    CallStrikes       = CallData[:, 0]
    PutStrikes        = PutData[:, 0]
    
    #Calls
    isRightDate        = (CallDates == CurrentDate)    #Boolean for each date    
    CurrentStrikes     = CallStrikes[isRightDate]      #Grab Current date Strikes
    nCalls             = np.size(CurrentStrikes, 0)    #number of calls for given date
    diff               = np.abs(CurrentStrikes - CurrentUnderlying) #Compute difference between strikes and underlying
    
    ATM_dummy            = np.zeros((nCalls, 1)) #Create ATM dummy vector
    ATM_index            = np.nonzero(diff == np.min(diff)) #find strike which is closest to underlying
    ATM_index            = ATM_index[0] #grab indices of options closest to underlying price
    ATM_dummy[ATM_index] = 1 #flag options 
    ATMCall_flag         = np.concatenate((ATMCall_flag, ATM_dummy), axis = 0) #Concatenate with previous dates
    
    OTM_dummy            = (CurrentStrikes < CurrentUnderlying)
    OTMCall_flag         = np.concatenate((OTMCall_flag, OTM_dummy.reshape(nCalls, 1)), axis = 0)
    
    Underlying_dummy     = CurrentUnderlying * np.ones((nCalls, 1))
    UnderlyingVec        = np.concatenate((UnderlyingVec, Underlying_dummy), axis = 0)
    
    
ATMCall_flag = ATMCall_flag[1:]
OTMCall_flag = OTMCall_flag[1:]
UnderlyingVec = UnderlyingVec[1:]

CallDataAug = np.concatenate((CallData, ATMCall_flag, OTMCall_flag, UnderlyingVec), axis = 1)



sys.exit()


ex_style = test["exercise_style"].to_numpy()
american_flag = ex_style == "A"
european_flag = ex_style == "E"

sys.exit()
data = pd.read_fwf(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\OptionData\SPXOptionTest.txt")
aapl = pd.read_fwf(r"C:\Users\ekblo\Documents\MScQF\Masters Thesis\Data\OptionData\AAPLOptionTest.txt")
colnames = np.array(["date", "cp_flag", "exercise_style", "index_flag", "exdate", "open_interest", \
                     "secid", "best_bid", "best_offer", "strike_price", "forward_price", "gamma",])
#colnames = np.array(["date", "cp_flag", "exercise_style", "index_flag", "exdate", "open_interest", \
#                     "secid", "best_bid", "best_offer", "strike_price", "forward_price", "gamma", "delta"])

headers   = aapl.iloc[3, :].to_numpy() #Grab part of the headers
dateIndex = np.nonzero(headers == "date") #find date index (this is the column to start from)
startCol  = int(dateIndex[0]) #get date index
startRow  = 5

aapl_keep = aapl.iloc[5:, 2:] #Data to keep
aapl_keep.columns = colnames
aapl_arr  = aapl_keep.to_numpy()

call_flag = aapl_arr[:, 1] == "C"
european_flag = aapl_arr[:, 2] == "E"
american_flag = aapl_arr[:, 2] == "A"
not_american_flag = np.nonzero(american_flag == 0)