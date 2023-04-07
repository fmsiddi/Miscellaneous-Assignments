import numpy as np
from numpy import random as rnd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

b = .1
σ = .2
S_0 = 100
T = 1
M = 10000
K = 100
N = 10000
Δt = T/M

V = np.zeros(N)
for n in tqdm(range(N)):
    S_t = np.zeros(M+1)
    S_t[0] = S_0
    for k in range(M):
        S_t[k+1] = S_t[k]*(1 + b*Δt + σ*np.sqrt(Δt)*rnd.normal())
        
    S_T = S_0*np.exp((b - .5*σ**2)*T + σ*np.sqrt(T)*rnd.normal())
    V[n] = max(S_t[M] - K,0) - max(S_T - K,0)
    
avg_V = [(1/(1+n))*V[n] for n in range(N)]
plt.plot(avg_V)

#%%
