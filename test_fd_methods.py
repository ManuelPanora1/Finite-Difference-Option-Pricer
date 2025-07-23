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
    n = [2 + 4** i for i in range(x)]
    m = [2 + 4** i for i in range(y)]
    bs_price = f.black_scholes_call(S, K, tfinal, r, sigma)

    cnerrors = np.zeros((x,y))
    fderrors = np.zeros((x,y))
    values = np.zeros((x,y))

    for i in range(len(n)):
        for j in range(len(m)):
            f1_price = f1(n[i], m[j], K, r, tfinal, S, sigma, *args)
            f2_price = f2("call", tfinal , sigma, r, K, m[j], n[i], S)
            cnerrors[i, j] = abs(f1_price - bs_price)
            fderrors[i, j] = abs(f2_price - bs_price)
            values[i, j] = f1_price

    return cnerrors, fderrors, values


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
        values : np.ndarray
            matrix of computed option price values at differeing step sizes using method 1
    """

    #Values for ranges for grid to test
    f1_errors = []
    f2_errors = []
    #For each of these values run the test for testing convergence, but pass in a direct grid
    for s_range in boundaries:
        f1_err_mat, f2_err_mat, f1_vals = test_convergence_all(K, r, tfinal, S, sigma, error_range, f1, f2, True, s_range)
        f1_errors.append(f1_err_mat)
        f2_errors.append(f2_err_mat)

    return f1_errors, f2_err_mat


    