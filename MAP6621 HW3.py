import numpy as np
from numpy import random as rnd
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%%

def euler_monte_carlo(b,σ,S_0,T,M,K,N):
    Δt = T/M
    V = np.zeros(N)
    a1 = 1 + b*Δt # constant calculated outside of loop to reduce flops required
    a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
    a1T = (b - .5*σ**2)*T # constant calculated outside of loop to reduce flops required
    a2T = σ*np.sqrt(T) # constant calculated outside of loop to reduce flops required
    for n in tqdm(range(N)):
        S_t = np.zeros(M+1)
        S_t[0] = S_0
        for k in range(M):
            S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal()) # simulating S_t via euler
        S_T = S_0*np.exp(a1T + a2T*rnd.normal()) # simulating GBM S_T at time T
        V[n] = max(S_t[M] - K,0) - max(S_T - K,0)
    return V.mean() # return average deviance over N simulations

#%%
b = .1
σ = .2
S_0 = 100
T = 1
M = 10000
K = 100
N = 10000
    
avg_V = euler_monte_carlo(b,σ,S_0,T,M,K,N)
print('avg deviance:', avg_V)

#%%

# black-scholes price
def BS(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    return S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 

# compute hedge values (in units of stocks and bonds)
def h(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    h_S = norm.cdf(d1) # # of stocks to buy is delta of option
    a = S*h_S # constant calculated to reduce flops required
    c = a - K*np.exp(-r*(T-t))*norm.cdf(d2) 
    h_B = (c - a)/np.exp(r*t) # of bonds to buy according to equation (8.20)
    return h_S, h_B

def hedge_error(b,r,σ,S_0,T,M,K,N):
    Δt = T/M
    t_k = np.linspace(0,T,M+1)
    gap_sq = np.zeros(N)
    a1 = 1 + b*Δt # constant calculated outside of loop to reduce flops required
    a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
    for n in tqdm(range(N)):
        S_t = np.zeros(M+1)
        V = np.zeros(M)
        S_t[0] = S_0
        for k in range(M):
            h_S, h_B = h(S_t[k],K,T,t_k[k],r,σ) # compute hedges
            V[k] = h_S*S_t[k] + h_B*np.exp(r*t_k[k]) # compute value of replicating portfolio
            S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal()) # simulate stock discretely using euler method
        gap_sq[n] = (V[M-1] - max(0, S_t[M] - K))**2 # compute wealth gap squared
    return gap_sq.mean() # return average wealth gap squared across N simulations

#%%
b = .1
r = .05
σ = .2
S_0 = 100
T = 1
M = 10000
K = 100
N = 100

avg_gap_sq = hedge_error(b,r,σ,S_0,T,M,K,N)
print('avg wealth gap squared:', avg_gap_sq)

#%%
b = .1
r = .05
σ = .2
S_0 = 100
T = 1
M = 10000
K = 100
N = 10000

Δt = T/M
t_k = np.linspace(0,T,M+1)
gap_sq = np.zeros(N)
a1 = 1 + b*Δt
a2 = σ*np.sqrt(Δt)
# for n in tqdm(range(N)):
S_t = np.zeros(M+1)
V = np.zeros(M)
C = np.zeros(M+1)
S_t[0] = S_0
C[0] = BS(S_0,K,T,0,r,σ)
for k in tqdm(range(M)):
    S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal())
    h_S, h_B = h(S_t[k],K,T,t_k[k],r,σ)
    V[k] = h_S*S_t[k] + h_B*np.exp(r*t_k[k])
    if k == M-1:
        C[k+1] = max(0,S_t[k+1]-K)
    else:
        C[k+1] = BS(S_t[k+1],K,T,t_k[k+1],r,σ)
    
plt.plot(C)
plt.plot(V)