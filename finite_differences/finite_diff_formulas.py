import numpy as np
from scipy.stats import norm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline

class call_option:
    def __init__(self, strike_price):
        self.strike_price = strike_price

    def expirey_price(self, S):
        return np.maximum(S - self.strike_price, 0)
    
def black_scholes_call(S, K, T, r, sigma):
    """
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility
    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def partial_v_s(Vps, Vfs, delS):
    #This is a second order accurate method, backward difference
    return (Vfs - Vps) / (2 * delS)

def partial_squared_v_s(Vps, Vcs, Vfs, delS):
    return (Vfs - 2 * Vcs + Vps) / (delS ** 2)

def finite_approx_prevV(S, sigma, r, V_curr, Vps, Vfs, delt, delS):
    dvds_squared = partial_squared_v_s(Vps, V_curr, Vfs, delS)
    dvds = partial_v_s(Vps, Vfs, delS)
    return  V_curr + (delt / 2) * (sigma ** 2) * S ** 2 * dvds_squared + delt * r * S  * dvds - delt * r * V_curr

def finite_difference_call(time_steps, price_steps, k, tfinal, S, sigma, r):
    """
    Calculate call price using finite differences.
    
    """
    #Create a grid of 0 with the number of timesteps indicating value of option at that time and with that stock price 
    Vat = np.zeros((2 * (time_steps - 1) + 1, time_steps))
    delS = S/price_steps
    s_expirey_price = S + delS * np.arange(-(time_steps - 1), time_steps)
    #Initial grid with expirey dates
    #print(s_expirey_price)
    Vat[:,-1] = np.maximum(s_expirey_price - k, 0)
    #print(Vat[:,-1])
    #Vectorized implementation
    for i in range(time_steps - 2, -1, -1):
        center = (time_steps - 1)
        lo = center - i 
        hi = center + i + 1

        S_c = s_expirey_price[lo : hi]
        Vcent = Vat[lo : hi, i + 1]
        Vnext = Vat[lo + 1 : hi + 1, i + 1]
        Vpast = Vat[lo - 1 : hi - 1, i + 1]
        delt = tfinal / time_steps
        Vat[lo : hi , i] = finite_approx_prevV(S_c, sigma, r, Vcent, Vpast, Vnext, delt, delS)
    #print(Vat)
    return Vat[time_steps - 1, 0]

def undo_transform(W, a, b, tau, lnS, r):
    exp_r = np.exp(-r * tau).reshape(1, -1) 
    exp_s = np.exp(a * lnS).reshape(-1, 1)    
    exp_b = np.exp(b * tau).reshape(1, -1)  
    return exp_s * exp_b * exp_r * W   

def transformed_option_price_at_expirey(logS, k, a):
    return np.exp(-a * logS) * np.maximum(np.exp(logS) - k, 0)

def transformed_option_price_at_xmax(x, tau, K, r, a, b):
    lnSmax = x[-1]
    return (np.exp(r * tau) * np.exp(-(a * lnSmax + b * tau)) * (np.exp(lnSmax) - K* np.exp(-r * tau)))

def create_grid(time_steps, price_steps, K, r, tfinal, a, b, S, sigma):
    """
    Create space over which option value is to be computed
    Parameters
    ----------
    time_steps (int):
        specifies time subdivisions in grid
    price_steps (int):
        specifies asset price subdivisions in grid

    Returns
    -------
    Grid (time_steps * price_steps array):
        grid space over which option value is to be computed
    """

    V = np.zeros((price_steps, time_steps))
    #Test parameters to see which is most efficient
    s_min = K / 2
    s_max = 3 * K
    s_range = np.linspace(s_min, s_max, price_steps)

    V[:, 0] = np.maximum(s_range - K, 0)
    tau = tfinal / time_steps
    t = np.array([tau * i for i in range(time_steps)])

    V[-1,:] = s_max - K * np.exp(-r * (tfinal - tau))
    return V, s_range, t, s_min, s_max

def differentiation_matrix(n, s_span, dt, sigma, r, S): 
    I = np.arange(n)
    dS = (s_span[-1] - s_span[0]) / (n - 1)
    alpha = 0.25 * dt * (sigma**2 * (S[I]**2 / dS**2) - r * S[I] / dS)
    beta = -0.5 * dt * (sigma**2 * (S[I]**2 / dS**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (S[I]**2 / dS**2) + r * S[I] / dS)
    #print(diags(-alpha[1:-1], offsets=-1, shape=(n, n)))
    ML = (
        diags(-alpha[1:-1], offsets=-1, shape=(n-2, n-2)) +
        diags(1 - beta[1:-1], offsets=0, shape=(n-2, n-2)) +
        diags(-gamma[1:-1], offsets=1, shape=(n-2, n-2))
    )
    
    # Explicit matrix (MR)
    MR = (
        diags(alpha[1:-1], offsets=-1, shape=(n-2, n-2)) +
        diags(1 + beta[1:-1], offsets=0, shape=(n-2, n-2)) +
        diags(gamma[1:-1], offsets=1, shape=(n-2, n-2))
    )
    
    return ML.tocsc(), MR.tocsc(), alpha, gamma

#No drift used in this equation
def adaptive_boundary(c, r, sigma, tfinal, S0):
    return S0 * np.exp(sigma * np.sqrt(tfinal)*c)

def price_derivative_cn(n, m, K, r, tfinal, a, b, S, sigma):
    """
    Prices a derivative using the Crank-Nicolson method.

    Parameters
    ----------
    n : int
        The number of time steps to be used in the time grid.
    m : int
        The number of asset price steps (spatial grid points).

    Returns
    -------
    float
        The price of the option at the final time, given the initial condition. 
        Specifically, this gives the value of the option at the strike price
        and the corresponding final time in the computed grid.
    """

    #Begin by first creating grid 
    V, s_range, tau, smin, smax = create_grid(n, m, K, r, tfinal, a, b, S, sigma)
    delt = tfinal / n
    ML, MR, alpha, gamma  = differentiation_matrix(m, (smin, smax), delt, sigma, r, s_range)
    for i in range(n - 1):
        Vy = MR @ V[1:-1, i]  
        Vy[0] += alpha[1] * (V[0, i] + V[0, i+1])  
        Vy[-1] += gamma[-2] * (V[-1, i] + V[-1, i+1])  
        V[1:-1, i+1] = spsolve(ML, Vy)  
        V[0, i+1] = 0  
        V[-1, i+1] = smax - K * np.exp(-r * (tfinal - tau[i+1]))  # Call at S_max

    #Now that we have our full discrete function, this becomes a root finding problem.
    cs = CubicSpline(s_range, V[:,-1], bc_type='natural')  # Clamped BCs
    price = cs(S)

    return price
