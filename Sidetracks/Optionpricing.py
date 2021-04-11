# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:16:21 2021

@author: ekblo
"""
#Option Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
days = 30
T = days / 360  #Time to maturity
N = 1000 #ticks per day
tick = T / N
CurrentStockPrice = 100
rf = 0.01

#Simulate geometric brownian motion (under Q)

def SimulateGeoBM(CurrentStock, T, N, rf):
    tick = T/N
    S_t = np.zeros((N,))
    S_t[0] = CurrentStockPrice
    for i in np.arange(1, N):
        noise  = np.sqrt(tick) * np.random.normal()
        drift  = rf * tick
        S_t[i] = S_t[i-1]*np.exp(drift + noise)
    return S_t

S_t = SimulateGeoBM(CurrentStockPrice, T, N, rf)

# nSims = 1000
# StockPaths = np.zeros((N, nSims))
# for i in np.arange(0, nSims):
#     StockPaths[:, i] = SimulateGeoBM(CurrentStockPrice, T, N, rf)
#x = np.linspace(0, days, N)
#plt.plot(x, StockPaths)

def ComputeBS(S, K, T, r, sigma, cp_flag):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/ (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp_flag == 1:
        Price = S * norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
        Delta = norm.cdf(d1)
    else:
        Price = np.exp(-r*T)*K*norm.cdf(-d2) - S * norm.cdf(-d1)
        Delta = norm.cdf(-d1)
    
    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))    
    return Price, Delta, Gamma

K      = 100
sigma  = 0.20
rf     = 0.01
S_list = np.arange(80, 120)
T      = 30 / 360
Delta_list = np.zeros((len(S_list),))
Gamma_list = np.zeros((len(S_list),))

for i in np.arange(0, len(S_list)):
    S = S_list[i]
    Price, Delta, Gamma = ComputeBS(S, K, T, rf, sigma, 0)
    Delta_list[i] = Delta
    Gamma_list[i] = Gamma

#Plot relationships

fig, ax = plt.subplots()
ax.plot(S_list, Gamma_list, color = "red", alpha = 0.8, label = "Gamma")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Gamma")
ax.set_title("Delta and Gamma vs Stock Price")

ax2 = ax.twinx()
ax2.plot(S_list, Delta_list, color = '#0504aa', alpha = 0.8, label = "Delta")
ax2.set_ylabel("Delta")
#Legend
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)


plt.figure()
plt.plot(S_list, Delta_list, color = '#0504aa', alpha = 0.8, label = "Delta")
plt.title("Delta vs Stock Price")
plt.ylabel("Delta")
plt.xlabel("Stock")
plt.legend()

plt.figure()
plt.plot(S_list, Gamma_list, color = "red", alpha = 0.8, label = "Gamma")
plt.title("Gamma vs Stock Price")
plt.ylabel("Gamma")
plt.xlabel("Stock")
plt.legend()

plt.figure()
plt.plot(Delta_list, Gamma_list)
plt.title("Gamma vs Delta")
plt.ylabel("Gamma")
plt.xlabel("Delta")

#Simulate Delta and Gamma 

norm.pdf(1)
norm.cdf(1)


sys.exit()
terminalPrices = StockPaths[-1, :]
CallPayoffs = np.zeros((N,))
for i in np.arange(0, N):
    CallPayoffs[i]    = np.max((terminalPrices[i] - K, 0))

SimPrice = np.mean(CallPayoffs)
print(SimPrice)

##Simulate strategy payoff
def OptionPayoff(S, K, cp_flag):
    if np.size(S) == 1:
        CallPayoff = np.max(S - K, 0)
        PutPayoff  = np.max(K - S, 0)
    else:
        CallPayoff = np.zeros((np.size(S)))
        PutPayoff  = np.zeros((np.size(S)))
        for i in np.arange(0, np.size(S)):
            CallPayoff[i] = np.max((S[i] - K, 0))
            PutPayoff[i]  = np.max((K- S[i], 0))

    if cp_flag == 1:
        return CallPayoff
    else:
        return PutPayoff
       

K = 100
S = 100
Svec = np.arange(80, 120)
sigma = 0.2
r = 0.01
[CallPrice, Delta] = ComputeBS(Svec, K, T, r, sigma, 1)
[PutPrice, Delta] = ComputeBS(Svec, K, T, r, sigma, 0)

Call = OptionPayoff(Svec, K, 1) 
Put  = OptionPayoff(Svec, K, 0)
StraddleProfit = Call + Put - 10
PortfolioValue = CallPrice + PutPrice


plt.figure()
plt.plot(Svec, StraddleProfit)
plt.plot(Svec, np.zeros((np.size(Svec))), "--b")
plt.plot(Svec, PortfolioValue)






