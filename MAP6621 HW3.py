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
    a1 = 1 + b*Δt
    a2 = σ*np.sqrt(Δt)
    a1T = (b - .5*σ**2)*T
    a2T = σ*np.sqrt(T)
    for n in tqdm(range(N)):
        S_t = np.zeros(M+1)
        S_t[0] = S_0
        for k in range(M):
            S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal())  
        S_T = S_0*np.exp(a1T + a2T*rnd.normal())
        V[n] = max(S_t[M] - K,0) - max(S_T - K,0)
        
    avg_V = [V[:n+1].mean() for n in range(N)]
    plt.plot(avg_V)
    return V.mean()

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

def BS(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    return S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 

def h(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    h_S = norm.cdf(d1)
    a = S*h_S
    c = a - K*np.exp(-r*(T-t))*norm.cdf(d2) 
    h_B = (c - a)/np.exp(r*t)
    return h_S, h_B

def hedge_error(b,r,σ,S_0,T,M,K,N):
    Δt = T/M
    t_k = np.linspace(0,T,M+1)
    gap_sq = np.zeros(N)
    a1 = 1 + b*Δt
    a2 = σ*np.sqrt(Δt)
    for n in tqdm(range(N)):
        S_t = np.zeros(M+1)
        V = np.zeros(M)
        S_t[0] = S_0
        for k in range(M):
            S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal())
            h_S, h_B = h(S_t[k+1],K,T,t_k[k+1],r,σ)
            V[k] = h_S*S_t[k+1] + h_B*np.exp(r*t_k[k+1])
        gap_sq[n] = (V[M-1] - max(0, S_t[M] - K))**2
    return gap_sq.mean()

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
# V = np.zeros((M,N))
gap_sq = np.zeros(N)
a1 = 1 + b*Δt
a2 = σ*np.sqrt(Δt)
# for n in tqdm(range(N)):
S_t = np.zeros(M+1)
V = np.zeros(M)
C = np.zeros(M)
S_t[0] = S_0
for k in tqdm(range(M)):
    S_t[k+1] = S_t[k]*(a1 + a2*rnd.normal())
    h_S, h_B = h(S_t[k+1],K,T,t_k[k+1],r,σ)
    # V[k,n] = h_S*S_t[k+1] + h_B*np.exp(r*t_k[k+1])
    V[k] = h_S*S_t[k+1] + h_B*np.exp(r*t_k[k+1])
    if k == M-1:
        C[k] = max(0,S_t[k+1]-K)
    else:
        C[k] = BS(S_t[k+1],K,T,t_k[k+1],r,σ)
    # gap_sq[n] = (V[M-1,n] - max(0, S_t[M] - K))**2
    # gap_sq[n] = (V[M-1] - max(0, S_t[M] - K))**2
    
plt.plot(C)
plt.plot(V)