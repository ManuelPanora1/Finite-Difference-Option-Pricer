# Finite-Difference-Option-Pricer

6/21/25 Setup the finite difference computations

6/23/25 We will measure the convergence of these finite differnces

6/24/25 Finished loop to compute derivative according to pde, maybe find more efficient way to implement? Crank Nicolson is the way in the future, currently working on vectorizing all operations! We have completed the vectorization on my own as well! Very nice!

Created option pricer using finite differences. Implemented using vectorized approach to save on computational overhead. Filled in initial grid and accounted for boundary conditions. I first implemented the naive method for pricing the option. True implementations utilize the Crank-Nicolson implementation.


6/25/25 Implement and investigate Crank-Nicolson method for boundary value problems. A good question to ask is "for this surface is it ok to assume that the volatility is constant?". Spent today investigating IVBVP.

6/26/25 Today will implement the Crank-Nicolson method for IVBVP

7/1/25 Today I will fully implement it and then I will rigorously test it an ensure that I have built the correct item!

7/11/25 Today we have encountered an issue here. We have assumed uniform spacing in the grid for S but then this results in non constant spacing for delta x complicating the calculation of the differentiation matrix. To fix this I am going to recreate the grid to handle this non uniform spacing.

7/12/25 We have created a working function of the Crank Nicolson method. I understand everything, though it is numerically unstable and I must work to improve those instabilities! We have to optimize across two variables, we will see which is the best for now and see what happens.

7/14/25 Asset price steps matter the most in our CN method as increasing them reduces the error. This makes sense

7/23/25 The next step is to make it adaptive to different conditions. That is what I am going to test now, for the rest of this session that is what I will focus on and the rest of the time I will focus on a machine learning program and make life great no matter what! I am going to make the best choices and life is going to be great!

8/1/25 I am going to break this project for a little bit, but if I can finish this with the testing today, that would be amazing! Let me get to work and I will make this happen! Advection conditions is satisfied in most cases, but should still be used in finding the optimal initial conditions. This may become an optimization problem. We can now validate the Neumann condition so that even if it does satisfy it, the residual must be shorter. The best solution for now is to 

8/2/25 Shocking discovery made, we are not using the correct number of time steps, we are making tremendous gaps, so we need more clustering near larger values. I think that exp could do this for us?

# Summary of findings

## Boundary Conditions
Empirically validated that dynamic boundaries:  
`S_min = K * exp(-3σ√T)`, `S_max = K * exp(3σ√T)`  
yield errors < .01% across and covers 99.7% possible ending price trajectories, and is most numerically stable:  
- Strikes (`K`): 10 to 500  
- Volatilities (`σ`): 0.1 to 0.5  
- Maturities (`T`): 0.1 to 10 years  

For extreme cases (e.g., `σ > 0.5` or `T > 10`), use:  
`n_std = 4 * sqrt(min(T, 1.0))`.

## Optimal Discretization
- **Time Step**: `Δt ≤ 0.5 * (ΔS)² / (σ² S_max²)`.
In satisfying the Neumann condition, we created necessary but insufficient conditions for the stability in this option pricing model for all cases.
- **Boundaries**: `n_std = 3.5` if `σ√T > 1`, else `3.0`.

- **Asset Grid**: Use `sinh_grid` with clustering near `K`.

## Future Directions
- Continue Improving European Stock Option Pricer