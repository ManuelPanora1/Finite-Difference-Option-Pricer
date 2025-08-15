# Finite-Difference-Option-Pricer

This project implements a vectorized European option pricer using finite-difference methods, with a particular focus on the Crank–Nicolson scheme. The primary goals were to validate boundary conditions, determine optimal discretizations for stability and accuracy, and explore a machine learning approach for adaptive grid selection.

---

## Project Timeline

- **6/21/25:** Set up basic finite difference computations.  
- **6/23/25:** Began measuring convergence of finite differences.  
- **6/24/25:** Completed vectorized derivative calculations; initial naive method implemented. Crank–Nicolson method planned for later improvement.  
- **6/25–6/26/25:** Implemented and investigated Crank–Nicolson (CN) for boundary value problems; started exploring stability under constant volatility assumption.  
- **7/1/25:** Fully implemented CN method; began rigorous testing.  
- **7/11–7/12/25:** Addressed non-uniform spacing in the asset grid; CN method functional but numerically unstable.  
- **7/14/25:** Found asset price steps (NAS) have the largest effect on error.  
- **7/23/25:** Tested adaptive grids for different conditions. Started integrating optimization approach for initial grid selection.  
- **8/1–8/4/25:** Validated Neumann and advection conditions, refined time steps and boundaries. Identified need for finer asset step clustering near key strike regions. Explored parallel processing for faster computations.

---

## Summary of Findings

### Boundary Conditions

Empirically validated that **dynamic boundaries**:

\[
S_\text{min} = K \cdot e^{-3\sigma\sqrt{T}}, \quad S_\text{max} = K \cdot e^{3\sigma\sqrt{T}}
\]

- Yield errors < 0.01% across a wide range of strikes, volatilities, and maturities.  
- Cover 99.7% of possible ending price trajectories.  
- Most numerically stable.

**Ranges tested:**  
- Strikes (`K`): 10–500  
- Volatilities (`σ`): 0.1–0.5  
- Maturities (`T`): 0.1–10 years  

**Extreme cases:** For `σ > 0.5` or `T > 10`, increase number of standard deviations (`n_std`) to `4 * sqrt(min(T, 1.0))`.

---

### Optimal Discretization

- **Time Step:**  

\[
\Delta t \leq \frac{(\Delta S)^2}{2\sigma^2 S_\text{max}^2}
\]

Necessary but insufficient condition for stability; ensures Crank–Nicolson residuals remain controlled.

- **Boundary Selection:**  

\[
n_\text{std} = 
\begin{cases} 
3.5 & \text{if } \sigma\sqrt{T} > 1 \\
3.0 & \text{otherwise} 
\end{cases}
\]

- **Asset Grid:** Use sinh-based spacing with clustering near `K` for higher accuracy.

- **Observation:** Asset price steps (NAS) significantly impact numerical error; time steps (NTS) less so if within stability bounds.

---

### Adaptive Grid via Continuous Mapping

- Generated a **lookup table** of optimal `(NAS, NTS)` for ~12,000 cases (reduced from ~3 million to focus on realistic regions).  
- Fitted a **continuous mapping** from initial conditions `(K, T, S0, σ)` to `(NAS, NTS)`.  
- Advantage: can predict near-optimal grid parameters for **unseen points** inside the convex hull of training data.  

**Testing Results:**  
- Evaluated CN pricing against Black–Scholes prices.  
- Relative errors remained **under 10%** for unseen cases.  
- Confirms the mapping generalizes well and provides accurate grid selection.

---

### Future Directions

1. Extend to **American or barrier options** using implicit finite difference schemes.  
2. Investigate **non-constant volatility models** (local or stochastic volatility).  
3. Optimize parallel processing and further reduce computation time for large grids.  
4. Explore **dynamic grid refinement** driven by machine learning or error estimates to maximize accuracy with minimal computational cost.

---

### Conclusion

This project demonstrates a robust framework for vectorized European option pricing using Crank–Nicolson finite differences. By carefully selecting dynamic boundaries, discretization steps, and adaptive grids, the method achieves high accuracy while remaining computationally efficient. The continuous mapping approach allows practical application to previously unseen option parameters, making this framework versatile for real-world scenarios.
