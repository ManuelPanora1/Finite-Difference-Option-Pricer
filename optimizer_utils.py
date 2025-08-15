import pandas as pd
import numpy as np
import test_fd_methods as t
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator

def compute_optimal_steps(args):
    K, T, S, sigma = args
    NAS, NTS, err = t.optimal_NAS_NTS(K, T, S, sigma, r=0.05)
    return {"K": K, "T": T, "S": S, "sigma": sigma, "NAS": NAS, "NTS": NTS, "Error": err}


def run_parallel_processing(param_grid, chunk_size=1000):
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(param_grid), chunk_size):
            print(i)
            chunk = param_grid[i:i + chunk_size]
            results.extend(list(executor.map(compute_optimal_steps, chunk)))

    return pd.DataFrame(results)

def fit_to_data(data):
    """
    Creates an interpolating function that predicts optimal NAS and NTS for given option parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing columns ["K", "T", "S", "sigma", "NAS", "NTS", "Error"]

    Returns
    -------
    predictor : Callable
        Function that takes (K, T, S, sigma) and returns (NAS, NTS)
    """

    #Begin by scaling the data
    #Prepare precomputed data to pass into spline for fitting
    X = data[["K", "T", "S", "sigma",]].values 
    y_NAS = data["NAS"].values                       
    y_NTS = data["NTS"].values   

    # Scales all features to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)     

    # For n (time steps)
    rbf_NAS = LinearNDInterpolator(X_scaled, y_NAS)  

    # For m (asset steps)
    rbf_NTS = LinearNDInterpolator(X_scaled, y_NTS) 

    def predictor(K, T, S, sigma):
        """
        Predicts optimal grid sizes with safe handling of edge cases.
        
        Parameters
        ----------
        K : float
            Strike price
        T : float 
            Time to maturity (years)
        S : float
            Underlying price
        sigma : float
            Volatility
            
        Returns
        -------
        Tuple
            (NAS, NTS) : grid sizes
                Optimal computed grid sizes
        """

        #Create new transformation of recieved data
        new_x = scaler.transform([[K, T, S, sigma]])[0]
        print(new_x)
        print(rbf_NAS(new_x), rbf_NTS(new_x))
        #Discretize continous prediction
        NAS = int(np.round(rbf_NAS(new_x)))
        NTS = int(np.round(rbf_NTS(new_x)))

        return NAS, NTS
    
    return predictor