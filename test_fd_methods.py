import finite_differences.finite_diff_formulas as f
import test_implementation as ti
import numpy as np

def test_convergence_all(K, r, tfinal, a, b, S, sigma, error):
    """
    Parameters
    ----------
    """
    #timesteps
    x, y = error
    n = [2 + 4** i for i in range(x)]
    m = [2 + 4** i for i in range(y)]
    bs_price = f.black_scholes_call(S, K, tfinal, r, sigma)
    cnerrors = np.zeros((x,y))
    fderrors = np.zeros((x,y))
    values = np.zeros((x,y))
    for i in range(len(n)):
        for j in range(len(m)):
            cn_price = f.price_derivative_cn(n[i], m[j], K, r, tfinal, a, b, S, sigma)
            fd_price = ti.crank_nicholson("call", tfinal,sigma,r, K, m[j], n[i], S)
            cnerrors[i, j] = abs(cn_price - bs_price)
            fderrors[i, j] = abs(fd_price - bs_price)
            values[i, j] = cn_price

    return cnerrors, fderrors, values
