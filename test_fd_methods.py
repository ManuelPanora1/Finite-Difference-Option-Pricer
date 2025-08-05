import finite_differences.finite_diff_formulas as f
import test_implementation as ti
import numpy as np

def test_convergence_all(K, r, tfinal, S, sigma, error_range, f1, f2, *args):
    """
    Purpose of method is two compare the convergence of two methods by varying time steps and asset price steps

    Parameters
    ----------
    K : float
        option strike price
    r : float
        risk free interest rate
    tfinal : float
        time of expiration
    S : float
        initial stock price
    sigma : float
        volatility
    error_range : list
        list storing int power we exponentiate time steps and asset steps with
    f1 : method
        first method used to price the option
    f2 : method
        second method used to price option
    
    Return
    ----------
    tuple
        cnerrors : np.ndarray
            matrix of errors for method 1 computed with differing step sizes
        fderrors : np.ndarray
            matrix of errors for method 2 computed with differing step sizes
        values : np.ndarray
            matrix of computed option price values at differeing step sizes using method 1
    
    """

    #timesteps arbitrary, will refine to get better results
    x, y = error_range
    #Time steps
    n = [2 + 4** i for i in range(x)]
    #Asset price steps
    m = [2 + 4** i for i in range(y)]
    bs_price = f.black_scholes_call(S, K, tfinal, r, sigma)

    cnerrors = np.zeros((x,y))
    fderrors = np.zeros((x,y))
    values = np.zeros((x,y))
    neuman_conds = np.zeros((x,y))

    for i in range(len(n)):
        for j in range(len(m)):
            #Compute whether Nueman Inequality holds
            if (len(args) > 0):
                maxS = args[1][1]
                delt = tfinal / n[i]
                dels = (args[1][1] - args[1][0]) / m[j]

                #Compute neuman condition using change in time and change in asset price
                Neumann_holds = delt <= (.5 * dels ** 2) / (sigma**2 * maxS**2)
                print(dels, delt)
                print(Neumann_holds)
                print(m[j], n[i])
                print()
                #If the Neumann conditions do not hold, skip entry
                if not Neumann_holds:
                    continue
                neuman_conds[i, j] = Neumann_holds

            f1_price = f1(n[i], m[j], K, r, tfinal, S, sigma, *args)
            f2_price = f2("call", tfinal , sigma, r, K, m[j], n[i], S)
            cnerrors[i, j] = abs(f1_price - bs_price)
            fderrors[i, j] = abs(f2_price - bs_price)
            values[i, j] = f1_price
            print(cnerrors[i, j])

    return cnerrors, fderrors, values, neuman_conds

def test_grid_asset_boundaries(f1, f2, K, r, tfinal, S, sigma, error_range, boundaries):
    """
    Purpose of method is to find optimal grid boundaries by passing in the minimum and maximum boundary values to function directly

    Parameters
    ----------
    f1 : method
        first method used to price the option
    f2 : method
        second method used to price option
    K : float
        option strike price
    r : float
        risk free interest rate
    tfinal : float
        time of expiration
    S : float
        initial stock price
    sigma : float
        volatility
    error_range : list
        list storing int power we exponentiate time steps and asset steps with
    boundaries : np.ndarray
        tuple containing upper and lower boundaries of created grid
    
    Return
    ----------
    tuple
        cnerrors : np.ndarray
            matrix of errors for method 1 computed with differing step sizes
        fderrors : np.ndarray
            matrix of errors for method 2 computed with differing step sizes
        conditions : np.ndarray
            matrix containing information if combination of parameters for f1 satisfy Neumann conditions
    """

    #Values for ranges for grid to test
    f1_errors = []
    f2_errors = []

    #Array storing validation of Neumann conditions
    conditions = []
    #For each of these values run the test for testing convergence, but pass in a direct grid
    for s_range in boundaries:
        f1_err_mat, f2_err_mat, f1_vals, Neumann_conds = test_convergence_all(K, r, tfinal, S, sigma, error_range, f1, f2, True, s_range)
        f1_errors.append(f1_err_mat)
        f2_errors.append(f2_err_mat)
        conditions.append(Neumann_conds)

    return f1_errors, f2_err_mat, conditions

def satisfies_advection_term(K, r, tfinal, S, sigma, error_range, f1):
    """
    The purpose of this function is to price the option and measure its accuracy only if it satisfies the advection conditions
    
    Parameters
    ----------
    K : float
        option strike price
    r : float
        risk free interest rate
    tfinal : float
        time of expiration
    S : float
        initial stock price
    sigma : float
        volatility
    error_range : list
        list storing int power we exponentiate time steps and asset steps with
    f1 : method
        Method used to price the option

    Returns
    -------
    np.ndarray
        f1errors : np.ndarray
            Matrix containing option price given initial conditions for varied NAS and NTS and 0 if advection condition does not hold
    """

    x, y = error_range
    #Time steps
    n = [2 + 4** i for i in range(x)]
    #Asset price steps
    m = [2 + 4** i for i in range(y)]
    bs_price = f.black_scholes_call(S, K, tfinal, r, sigma)

    #Errors associated with function1 
    f1errors = np.zeros((x,y))

    for i in range(len(n)):
        for j in range(len(m)):
            std_dev = 3.5 if sigma*np.sqrt(tfinal) > 1 else 3
            Smin = adaptive_boundary(-std_dev, sigma, tfinal, K)
            Smax = adaptive_boundary(std_dev, sigma, tfinal, K)
            delt = tfinal / n[i]
            dels = (Smax - Smin) / m[j]
            if (advection_condition(delt, dels, r, Smax)):
                f1_price = f1(n[i], m[j], K, r, tfinal, S, sigma)
                f1errors[i, j] = abs(f1_price - bs_price)

    return f1errors

def adaptive_boundary(c, sigma, tfinal, K):
    return K * np.exp(sigma * np.sqrt(tfinal)*c)

def advection_condition(delt, dels, r, Smax):
    """
    Purpose of this function is to check if advection conditions hold given the starting conditions

    Parameters
    ----------
    delt : float
        Grid time step
    dels : float
        Grid asset price step
    r : float
        Risk free interest rate
    Smax : float
        Maximum asset price used in grid

    Returns
    -------
    bool
        True if advection holds, false else
    """
    return (delt) < (r * Smax / dels)

def optimal_NAS_NTS(K, tfinal, S, sigma, r, f1 = f.price_derivative_cn):
    """
    Purpose of this method is to return the optimal number of time steps and asset price steps given initial starting conditions

    Parameters
    ----------

    S : float
        Current stock price
    K : float
        Strike price
    tfinal : float 
        Time to maturity (years)
    r : float 
        Risk-free interest rate
    sigma : float
        Volatility
    Returns:
        tuple:
            NAS, NTS: (int, int)
                NAS is the optimal asset price steps and NTS is the optimal time steps
    """

    #Generate list of values to test. Log scale to cluster time-steps around larger values
    #From previous experimentation, we have found that 2 ** 8, has been sufficient for time steps
    NAS_upper = 2 ** 10
    NAS_lower = 2 ** 5
    NTS_upper = 2 ** 8
    NTS_lower = 2 ** 5
    NAS_intervals = 10
    NTS_intervals = 10

    NAS = np.round(np.sqrt(np.linspace((NAS_lower)**2, (NAS_upper)**2, NAS_intervals)).astype(int))
    NTS = np.round(np.sqrt(np.linspace((NTS_lower)**2, (NTS_upper)**2, NTS_intervals)).astype(int))
    
    #For each possible pair, if it satisfies the Neumann condition and the advection condition, we compute the error
    bs_price = f.black_scholes_call(S, K, tfinal, r, sigma)

    #Errors associated with function1 
    f1errors = np.zeros((NAS_intervals ,NTS_intervals))

    min_error = np.inf

    min_steps = (0,0)

    std_dev = 3.5 if sigma*np.sqrt(tfinal) > 1 else 3
    Smin = adaptive_boundary(-std_dev, sigma, tfinal, K)
    Smax = adaptive_boundary(std_dev, sigma, tfinal, K)

    for i in range(NAS_intervals):
        for j in range(NTS_intervals):
            
            delt = tfinal / NTS[j]
            dels = (Smax - Smin) / NAS[i]
            Neumann_holds = (delt <= (.5 * dels ** 2) / (sigma**2 * Smax **2))
            print(NAS[j], NTS[i])
            print(dels, delt)
            print(Neumann_holds)
            print()
            if (advection_condition(delt, dels, r, Smax) and  Neumann_holds):
                f1_price = f1(NTS[j], NAS[i], K, r, tfinal, S, sigma)
                #Measure in relative error terms
                #TODO FOUND LARGE BUG WHAT IF IT IS WORTHLESS? THINK ABOUT IT!
                f1errors[i, j] = abs(f1_price - bs_price) / bs_price
                if (f1errors[i, j] < min_error):
                    min_error = f1errors[i, j]
                    min_steps = [i,j]
    
    print(f1errors)
    mask = f1errors > 0
    f1errors = f1errors[mask]

    return NAS[min_steps[0]], NTS[min_steps[1]]

    #Return the NAS and NTS that minimizes the error