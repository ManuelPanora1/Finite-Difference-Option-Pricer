import numpy as np
from scipy.stats import norm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
    
def black_scholes_call(S, K, T, r, sigma):
    """
    Analytically compute the price of Europeans call option using BSM

    Parameters
    ----------

    S : float
        Current stock price
    K : float
        Strike price
    T : float 
        Time to maturity (years)
    r : float 
        Risk-free interest rate
    sigma : float
        Volatility
    Returns:
        float
            call_price: float
                Option price
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

def finite_approx_prevV(S, sigma, r, V_curr, Vps, Vfs, delt, delS):
    """
    Vectorized computation of option value at previous time step using naive finite differences

    Parameters
    ----------
    S : np.ndarray
        Asset price values at previous time step
    sigma : float
        Volatility
    r : flaot 
        Risk-free interest rate
    V_curr : np.ndarray
        Option price at previous time step for current asset price
    Vps : np.ndarray
        Option price at previous time step for previous asset price
    Vfs : np.ndarray
        Option price at previous time step for future asset price
    delt : float
        discrete time step
    delS : float
        discrete asset price step

    Returns
    -------
    np.ndarray
        option_price : np.ndarray
            Values for grid with asset prices

    """
    dvds_squared = (Vfs - 2 * V_curr + Vps) / (delS ** 2)
    dvds = (Vfs - Vps) / (2 * delS)
    option_price = V_curr + (delt / 2) * (sigma ** 2) * S ** 2 * dvds_squared + delt * r * S  * dvds - delt * r * V_curr
    return option_price

def finite_difference_call(time_steps, price_steps, k, tfinal, S, sigma, r):
    """
    Calculate European call price using finite differences efficiently by only 
    computing values directly used for computation of intial option price

    Parameters
    ----------
    time_steps : int
        Number of timesteps used in grid
    price_steps : int
        Number of pricesteps used in grid
    k : float
        Strike price
    tfinal : float
        Time to maturity (years)
    S : float
        Current asset price
    sigma : float
        Volatility
    r : float
        Risk-free interest rate
    
    Return
    ----------
    float
        Vat[time_steps - 1, 0] : float
            Option price at current time
    """

    #Initialize grid with twice the time steps to capture widest range
    Vat = np.zeros((2 * (time_steps - 1) + 1, time_steps))
    
    #Enforce boundary conditions
    delS = S/price_steps
    s_expirey_price = S + delS * np.arange(-(time_steps - 1), time_steps)
    Vat[:,-1] = np.maximum(s_expirey_price - k, 0)

    #Loop to numerically solve for option price
    for i in range(time_steps - 2, -1, -1):
        #For efficient computation option price computed through time in triangular form
        center = (time_steps - 1)
        lo = center - i 
        hi = center + i + 1

        #Adaptive ranges store necessary values at previous time steps to use in computation of current time step value
        S_c = s_expirey_price[lo : hi]
        Vcent = Vat[lo : hi, i + 1]
        Vnext = Vat[lo + 1 : hi + 1, i + 1]
        Vpast = Vat[lo - 1 : hi - 1, i + 1]
        delt = tfinal / time_steps

        Vat[lo : hi , i] = finite_approx_prevV(S_c, sigma, r, Vcent, Vpast, Vnext, delt, delS)

    #Final option price stored in center of computed grid
    return Vat[time_steps - 1, 0]

def create_grid_cn(time_steps, price_steps, K, r, tfinal, s_min, s_max):
    """
    Initializes the grid for the Crank-Nicolson PDE solver.

    Parameters
    ----------
    time_steps : int
        specifies time subdivisions in grid
    price_steps : int
        specifies asset price subdivisions in grid
    K : float
        strike price used in computing boundary conditions
    r : float
        risk free interest rate (annualized)
    tfinal : float
        time to expirey (years)

    Returns
    -------
    tuple
        V : np.ndarray
            Initialized grid with payoff conditions (shape: (price_steps, time_steps)).
        s_range : np.ndarray
            Array of asset price values (from S_min to S_max).
        t : np.ndarray
            Array of time step values (time_steps time steps from 0 to tfinal)
    """

    V = np.zeros((price_steps, time_steps))

    s_range = np.linspace(s_min, s_max, price_steps)

    #Implement boundary conditions at expirey
    V[:, 0] = np.maximum(s_range - K, 0)

    #Enforce boundary conditions at s_max
    delt = tfinal / time_steps
    t = np.array([delt * i for i in range(time_steps)])
    V[-1,:] = s_max - K * np.exp(-r * t)

    return V, s_range, t

def differentiation_matrix(n, s_span, dt, sigma, r, S): 
    """
    Constructs the matrices (ML and MR) for the Crank-Nicolson method.
    
    Parameters
    ----------
    n : int
        Number of price steps.
    s_span : tuple
        (S_min, S_max) asset price range.
    dt : float
        Time step size.
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    S : np.ndarray
        Asset price grid.
    
    Returns
    -------
    tuple
        ML : scipy.sparse.csc_matrix
            Implicit (left-hand side) matrix.
        MR : scipy.sparse.csc_matrix
            Explicit (right-hand side) matrix.
        alpha : np.ndarray
            Coefficients for lower diagonal.
        gamma : np.ndarray
            Coefficients for upper diagonal.
    """

    I = np.arange(n)

    dS = (s_span[-1] - s_span[0]) / (n - 1)

    #Diagonal matrices computing constants for semi-discretization
    alpha = 0.25 * dt * (sigma**2 * (S[I]**2 / dS**2) - r * S[I] / dS)
    beta = -0.5 * dt * (sigma**2 * (S[I]**2 / dS**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (S[I]**2 / dS**2) + r * S[I] / dS)

    ML = (
        diags(-alpha[1:-1], offsets=-1, shape=(n-2, n-2)) +
        diags(1 - beta[1:-1], offsets=0, shape=(n-2, n-2)) +
        diags(-gamma[1:-1], offsets=1, shape=(n-2, n-2))
    )
    
    MR = (
        diags(alpha[1:-1], offsets=-1, shape=(n-2, n-2)) +
        diags(1 + beta[1:-1], offsets=0, shape=(n-2, n-2)) +
        diags(gamma[1:-1], offsets=1, shape=(n-2, n-2))
    )
    
    return ML.tocsc(), MR.tocsc(), alpha, gamma

#No drift used in this equation
def adaptive_boundary(c, sigma, tfinal, K):
    return K * np.exp(sigma * np.sqrt(tfinal)*c)

def price_derivative_cn(n, m, K, r, tfinal, S, sigma, test = False, grid_range = (0,0)):
    """
    Prices a European call option using Crank-Nicolson method.
    
    Parameters
    ----------
    n : int
        Time steps.
    m : int
        Price steps.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    tfinal : float
        Time to expiration (years).
    S0 : float
        Current asset price.
    sigma : float
        Volatility.
    s_min : float
        Lower boundary for created grid
    s_max : float
        Upper boundary for created grid
    
    Returns
    -------
    float
        Option price at S0.
    """

    #Adaptive grid boundary definition
    if (test):
        s_min = grid_range[0] 
        s_max = grid_range[1]
    else: 
        #Hardcode adaptive boundary depending on initial boundaries
        std_dev = 3.5 if sigma*np.sqrt(tfinal) > 1 else 3
        s_min = adaptive_boundary(-std_dev, sigma, tfinal, K)
        s_max = adaptive_boundary(std_dev, sigma, tfinal, K)


    #Begin by first creating option grid, price_grid, and time_grid
    V, s_range, t = create_grid_cn(n, m, K, r, tfinal, s_min, s_max)

    #Constants
    smin = s_range[0]
    smax = s_range[-1]
    delt = tfinal / n

    ML, MR, alpha, gamma  = differentiation_matrix(m, (smin, smax), delt, sigma, r, s_range)

    #Main loop for PDE solving
    for i in range(n - 1):
        Vy = MR @ V[1:-1, i]  

        #Enforce boundary conditions before solving to increase stability
        V[0, i+1] = 0  
        V[-1, i+1] = smax - K * np.exp(-r * (tfinal - t[i+1]))  

        Vy[0] += alpha[1] * (V[0, i] + V[0, i+1])  
        Vy[-1] += gamma[-2] * (V[-1, i] + V[-1, i+1])  

        #Fill grid with solution to system
        V[1:-1, i+1] = spsolve(ML, Vy)  
       
    #Now that we have our full discrete set of points, this becomes a root finding problem.
    cs = CubicSpline(s_range, V[:,-1], bc_type='natural')  
    price = cs(S)

    return price