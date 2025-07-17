import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import pandas as pd
from scipy.interpolate import CubicSpline

def crank_nicholson(type, expiration, sigma, r, strike, NAS, NTS, s):
    S_min = strike/3
    S_max = strike*2

    dS = (S_max-S_min)/NAS
    dt = expiration/NTS

    S = np.arange(0, NAS+1)* dS +S_min

    V = np.zeros((NAS + 1, NTS + 1))

    payoff = np.maximum((S - strike), 0)  # Call payoff
    V[:, -1] = payoff
    V[0, :] = 0  # Call value at S_min is 0
    V[-1, :] = (S_max - strike) * np.exp(-r * np.linspace(0, expiration, NTS + 1)[::-1])  # Call value at S_max
    I = np.arange(0,NAS+1)

    alpha = 0.25 * dt * ((sigma**2) * (I**2) - r*I)
    beta = -dt * 0.5 * (sigma**2 * (I**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (I**2) + r * I)

    ML = diags([-alpha[2:], 1-beta[1:], -gamma[1:]], [-1,0,1], shape=(NAS-1, NAS-1)).tocsc()
    MR = diags([alpha[2:], 1+beta[1:], gamma[1:]], [-1,0,1], shape=(NAS-1, NAS-1)).tocsc()

    for t in range(NTS - 1, -1, -1):
        boundary_t = np.zeros(NAS - 1)
        boundary_t[0] = alpha[1] * (V[0, t] + V[0, t + 1]) -alpha[0] * V[0, t + 1]
        boundary_t[-1] = gamma[NAS - 1] * (V[NAS, t] + V[NAS, t + 1])
        b = MR.dot(V[1:NAS, t + 1]) + boundary_t
        V[1:NAS, t] = spsolve(ML, b)
        #V[0, t] = 2 * V[1, t] - V[2, t]


    cs = CubicSpline(S, V[:,0], bc_type='natural')  # Clamped BCs
    price = cs(s)
    print(price)
    return price

"""
if __name__ == "__main__":
    K =95
    sigma = 0.2
    r = 0.05
    expiration = 21/252
    NAS = 1000
    NTS = 1000
    type = "call"

    option_df = crank_nicholson(type = type, strike = K, sigma = sigma, r = r, 
                                expiration = expiration, 
                                NAS = NAS, NTS = NTS, s = 100)
    
    print(option_df)
"""