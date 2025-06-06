#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats


# In[4]:


class RandomNumberGenerator:
    """
    Wrapper for generating standard normal random numbers.
    Inputs:
      - n_paths: number of Monte Carlo paths.
      - n_steps: number of time steps per path.
    """
    def __init__(self):
        self.rng = None

    def normals(self, n_paths, n_steps, seed=2025):
        """Generate an array of shape (n_paths, n_steps) of standard normal variates."""
        self.rng = np.random.default_rng(seed)
        return self.rng.standard_normal(size=(n_paths, n_steps))


# In[2]:


n_paths = 100
n_steps = 250
seed = 2025

rng = RandomNumberGenerator()
normals = rng.normals(n_paths, n_steps, seed=seed)
print(normals, "\n") 


# In[5]:


class PathGenerator:
    """
    Generates asset price paths under geometric Brownian motion:
        dS = S * ((r - q) dt + sigma dW)
    Inputs:
      - r: risk-free rate
      - q: dividend yield
      - sigma: volatility
      - S0: initial spot price
    """
    def __init__(self, r, q, sigma, S0):
        self.r = r
        self.q = q
        self.sigma = sigma
        self.S0 = S0

    def simulate(self, times, normals):
        """
        Simulate paths for given time grid and pre-generated normals.

        - times: 1D array of time points, shape (N+1,)
        - normals: array of shape (n_paths, N) of standard normals
        Returns: array of simulated paths shape (n_paths, N+1)
        """
        dt = np.diff(times)
        n_paths, N = normals.shape
        S_paths = np.empty((n_paths, N+1))
        S_paths[:, 0] = self.S0
        for i in range(N):
            drift = (self.r - self.q - 0.5 * self.sigma**2) * dt[i]
            diffusion = self.sigma * np.sqrt(dt[i]) * normals[:, i]
            S_paths[:, i+1] = S_paths[:, i] * np.exp(drift + diffusion)
        return S_paths


# In[64]:


r = 0.03
q = 0.01
sigma = 0.30
S0 = 100.0

path_gen = PathGenerator(r, q, sigma, S0)
times = np.linspace(0.0, 1.0, n_steps + 1)
S_paths = path_gen.simulate(times, normals)
print(S_paths, "\n")


# In[6]:


class PathInterpolator:
    """
    Geometric (Brownian-bridge) interpolation of log-prices between coarse points.
    Inputs:
      - times_coarse: (M+1,) array of times t_0 < t_1 < … < t_M
      - S_coarse:   (n_paths, M+1) array of prices at those times
      - L:           integer number of subintervals between each coarse interval
    Output:
      - (n_paths, M*L+1) array of interpolated prices
    """
    @staticmethod
    def interpolate(times_coarse, S_coarse, L):
        """
        For each coarse interval [t_j, t_{j+1}], insert L subintervals by Brownian-bridge:
          log S(t) = log S(t_j) + alpha (log S(t_{j+1}) - log S(t_j)),
        where alpha = (t - t_j)/(t_{j+1}-t_j).
        Returns array of shape (n_paths, M*L+1).
        """
        # Logs of coarse prices
        logS = np.log(S_coarse)           # (n_paths, M+1)
        n_paths, M1 = S_coarse.shape
        M = M1 - 1

        # Build fine time grid uniformly subdividing each coarse interval
        times_fine = np.empty(M * L + 1)
        idx = 0
        for j in range(M):
            # subdivide [t_j, t_{j+1}] into L steps
            t_j = times_coarse[j]
            t_j1 = times_coarse[j+1]
            for k in range(L):
                times_fine[idx] = t_j + (t_j1 - t_j) * (k / L)
                idx += 1
        times_fine[idx] = times_coarse[-1]

        # For each fine time, find its coarse interval index j_k
        # since uniform, j_k = k//L
        S_fine = np.empty((n_paths, M * L + 1))
        # Interpolate in log-space
        for j in range(M):
            x0 = logS[:, j:j+1]            # (n_paths,1)
            x1 = logS[:, j+1:j+2]          # (n_paths,1)
            # alphas shape (L,)
            alphas = np.linspace(0, 1, L+1)[:-1][np.newaxis, :]
            block = np.exp(x0 + alphas * (x1 - x0))  # (n_paths,L)
            # place block
            start_idx = j * L
            S_fine[:, start_idx:start_idx+L] = block
        # Last point
        S_fine[:, -1] = S_coarse[:, -1]
        return S_fine


# In[16]:


class PathInterpolator:

    @staticmethod
    def interpolate(
        S_paths: np.ndarray,
        T: float,
        n_coarse_steps: int,
        L: int
    ) -> np.ndarray:
        
        # (1) Build coarse‐time grid (if needed elsewhere; here only length matters):
        times_coarse = np.linspace(0, T, n_coarse_steps + 1)

        # (2) Verify that S_paths has the expected number of columns:
        n_paths, full_len = S_paths.shape
        expected_len = n_coarse_steps * L + 1
        if full_len != expected_len:
            raise ValueError(
                f"S_paths has shape[1]={full_len}, but expected {expected_len} "
                f"for (n_coarse_steps*L + 1) = ({n_coarse_steps}*{L} + 1)."
            )

        # (3) Extract the coarse‐grid prices by taking every L-th column:
        #     indices = [0, L, 2L, …, n_coarse_steps*L]
        indices = np.arange(0, full_len, L)        # shape = (n_coarse_steps + 1,)
        S_coarse = S_paths[:, indices]             # shape = (n_paths, n_coarse_steps+1)

        # (4) Perform log‐space linear interpolation on each coarse interval:
        logS = np.log(S_coarse)                    # shape = (n_paths, M+1)
        M = n_coarse_steps                         # number of coarse intervals

        # Allocate the output array:
        S_fine = np.empty((n_paths, M * L + 1))

        for j in range(M):
            # Beginning‐of‐interval log price:
            x0 = logS[:, j    : j+1]               # shape = (n_paths, 1)
            # End‐of‐interval log price:
            x1 = logS[:, j+1  : j+2]               # shape = (n_paths, 1)

            # α‐values = [0, 1/L, 2/L, …, (L−1)/L], shape = (1, L)
            alphas = np.linspace(0, 1, L+1)[:-1][np.newaxis, :]

            # Broadcast to (n_paths, L):
            block = np.exp(x0 + alphas * (x1 - x0)) # shape = (n_paths, L)

            start_idx = j * L
            S_fine[:, start_idx : start_idx + L] = block

        # Finally, copy the very last coarse price into the final column:
        S_fine[:, -1] = S_coarse[:, -1]
        return S_fine


# In[67]:


n_coarse_steps = 50
L               = 5
T               = 1.0
S_fine = PathInterpolator.interpolate(
    S_paths=S_paths,
    T=T,
    n_coarse_steps=n_coarse_steps,
    L=L
)
S_fine


# In[7]:


class DecrementIndex:
    """
    Computes the decrement index I_t from simulated stock paths.

    Rule:
        I_{n} = max(0,
                    I_{n-1} * (1 - delta1) * (S_n / S_{n-1} + q * dt)
                    - delta2)
    Inputs:
      - delta1: percentage decrement (0 < delta1 < 1, 0.05)
      - delta2: point decrement (>0, 0.01 * S0)
      - q: scalar dividend yield
      - dt: time step size (scalar)
      - I0: initial index level
    """
    def __init__(self, delta1, delta2, q, dt, I0=100.0):
        self.delta1 = delta1
        self.delta2 = delta2
        self.q = q
        self.dt = dt
        self.I0 = I0

    def compute(self, S_paths):
        """
        Compute decrement index for each simulated path.

        - S_paths: array of shape (n_paths, N+1)
        Returns: I_paths of same shape.
        """
        n_paths, N1 = S_paths.shape
        I_paths = np.zeros_like(S_paths)
        I_paths[:, 0] = self.I0
        # Iterate time steps
        for n in range(1, N1):
            ratio = S_paths[:, n] / S_paths[:, n-1]
            I_prev = I_paths[:, n-1]
            I_paths[:, n] = np.maximum(
                0,
                I_prev * (1 - self.delta1) * (ratio + self.q * self.dt)
                - self.delta2
            )
        return I_paths


# In[72]:


delta1 = 0.05
delta2 = 0.01 * S0   # e.g. 1% of initial price
q = 0.01
I0 = 100.0

decrement_index = DecrementIndex(delta1, delta2, q, dt, I0)
I_paths_full = decrement_index.compute(S_paths)
I_paths_interp = decrement_index.compute(S_fine)
I_paths_full
I_paths_interp


# In[8]:


class Payoff:
    """
    Base class for payoffs.
    """
    def __call__(self, I_paths):
        raise NotImplementedError


# In[9]:


class VanillaCall(Payoff):
    """Vanilla call option on the decrement index."""
    def __init__(self, K):
        self.K = K
    
    def __call__(self, I_paths):
        payoffs = np.maximum(I_paths[:, -1] - self.K, 0)
        return payoffs


# In[10]:


class VanillaPut(Payoff):
    """Vanilla put option on the decrement index."""
    def __init__(self, K):
        self.K = K
    
    def __call__(self, I_paths):
        payoffs = np.maximum(self.K - I_paths[:, -1], 0)
        return payoffs


# In[11]:


class BarrierDownOut(Payoff):
    """
    Down-and-out barrier on index: zero if I falls below barrier,
    else vanilla call payoff on final level.
    """
    def __init__(self, K, barrier):
        self.K = K
        self.barrier = barrier
    
    def __call__(self, I_paths):
        payoffs = np.maximum(I_paths[:, -1] - self.K, 0)
        touched = (I_paths <= self.barrier).any(axis=1)
        payoffs[touched] = 0
        return payoffs


# In[12]:


class AutoCallable(Payoff):
    """
    Auto-callable on the decrement index I_t with:
      - Periodic digital coupon at quarterly observations
      - Up-and-out barrier enforced only at each observation date
      - European down-in put at maturity (gap put)
    """
    
    def __init__(self, times_obs, coupon, barrier_up, barrier_down, K_put, dt=1/250):
        """
        Parameters:
        -----------
        times_obs : list of float
            Observation times in years (e.g. [0.25, 0.5, 0.75, 1.0])
        coupon : float
            Digital coupon amount paid at each observation date
        barrier_up : float
            Up-and-out barrier level (triggers early exercise)
        barrier_down : float
            Down-in barrier level for the European put (barrier < K_put for gap put)
        K_put : float
            Strike price for the put option
        dt : float
            Time step in years (default: 1/250 for daily)
        """
        self.times_obs = np.array(times_obs)
        self.obs_idx = [int(round(t_year / dt)) for t_year in times_obs]
        self.coupon = coupon
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down
        self.K_put = K_put
        self.dt = dt
    
    def __call__(self, I_paths):
        """
        Evaluate payoffs for given index paths.
        
        Parameters:
        -----------
        I_paths : np.ndarray
            Array of shape (n_paths, n_steps) containing index paths
            
        Returns:
        --------
        np.ndarray
            Payoffs for each path
        """
        n_paths, n_steps = I_paths.shape
        payoffs = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        
        # We'll determine down-in status at maturity based on final price only
        
        # Check each observation date
        for i, obs_idx in enumerate(self.obs_idx):
            if obs_idx >= n_steps:
                continue
                
            # For alive paths, pay coupon at each observation
            payoffs[alive] += self.coupon
            
            # Check which alive paths breach the up barrier (knock-out)
            trigger = alive & (I_paths[:, obs_idx] >= self.barrier_up)
            
            # Terminate these paths (they keep their accumulated coupons)
            alive[trigger] = False
        
        # Handle survivors at final observation
        if len(self.obs_idx) > 0:
            final_idx = min(self.obs_idx[-1], n_steps - 1)
            survivors = alive
            
            # Check down-in condition only at final price
            final_prices = I_paths[survivors, final_idx]
            down_in_condition = final_prices <= self.barrier_down
            
            # Paths where final price is at or below down-in barrier get put protection
            put_payoff = np.maximum(self.K_put - final_prices, 0.0)
            payoffs[survivors] += np.where(down_in_condition, put_payoff, 1.0)
        
        return payoffs


# In[15]:


class MonteCarlo:
    """
    Monte Carlo pricer for decrement index derivatives.
    Handles both full simulation and interpolation-based acceleration.
    """

    def __init__(self, path_generator, decrement_index, rng=None):
        self.path_generator = path_generator
        self.decrement_index = decrement_index
        self.rng = rng if rng is not None else RandomNumberGenerator()
        self._cached_stock_paths = None
        self._cached_times = None
        self._cached_seed = None

    def price_full_simulation(self, payoff, T, n_paths, n_steps, seed=2025):
        
        start_time = time.time()

        # Generate time grid and normals
        times = np.linspace(0, T, n_steps + 1)
        normals = self.rng.normals(n_paths, n_steps, seed)
        S_paths = self.path_generator.simulate(times, normals)
        I_paths = self.decrement_index.compute(S_paths)

        # Cache for potential interpolation
        self._cached_stock_paths = S_paths.copy()
        self._cached_times = times.copy()
        self._cached_seed = seed

        # Compute payoffs
        payoffs = payoff(I_paths)

        # Statistics
        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        end_time = time.time()
        computation_time = end_time - start_time

        final_I = I_paths[:, -1]
        I_stats = {
            'mean': np.mean(final_I),
            'std': np.std(final_I),
            'min': np.min(final_I),
            'max': np.max(final_I),
            'median': np.median(final_I),
            'percentile_5': np.percentile(final_I, 5),
            'percentile_95': np.percentile(final_I, 95)
        }

        payoff_stats = {
            'mean': price,
            'std_error': std_error,
            'std': np.std(payoffs),
            'min': np.min(payoffs),
            'max': np.max(payoffs),
            'non_zero_fraction': np.mean(payoffs > 0),
            'median': np.median(payoffs)
        }

        payoff_name = payoff.__class__.__name__

        return {
            'payoff_name': payoff_name,
            'price': price,
            'std_error': std_error,
            'computation_time': computation_time,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'final_I_stats': I_stats,
            'payoff_stats': payoff_stats,
            'I_paths': I_paths,
            'payoffs': payoffs
        }

    def price_with_interpolation(self, payoff, T, n_paths, n_coarse_steps,
                                 L, seed=2025, use_cached_paths=True):
       
        start_time = time.time()

        # Coarse time grid
        times_coarse = np.linspace(0, T, n_coarse_steps + 1)

        # Use cached stock paths if possible
        if (use_cached_paths and
            self._cached_stock_paths is not None and
            self._cached_seed == seed):
            step_size = (self._cached_stock_paths.shape[1] - 1) // n_coarse_steps
            indices = np.arange(0, self._cached_stock_paths.shape[1], step_size)[:n_coarse_steps + 1]
            indices[-1] = self._cached_stock_paths.shape[1] - 1
            S_coarse = self._cached_stock_paths[:, indices]
        else:
            normals_coarse = self.rng.normals(n_paths, n_coarse_steps, seed)
            S_coarse = self.path_generator.simulate(times_coarse, normals_coarse)

        # Interpolate to fine grid
        S_fine = PathInterpolator.interpolate(times_coarse, S_coarse, L)
        I_paths = self.decrement_index.compute(S_fine)
        payoffs = payoff(I_paths)

        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        end_time = time.time()
        computation_time = end_time - start_time

        final_I = I_paths[:, -1]
        I_stats = {
            'mean': np.mean(final_I),
            'std': np.std(final_I),
            'min': np.min(final_I),
            'max': np.max(final_I),
            'median': np.median(final_I),
            'percentile_5': np.percentile(final_I, 5),
            'percentile_95': np.percentile(final_I, 95)
        }

        payoff_stats = {
            'mean': price,
            'std_error': std_error,
            'std': np.std(payoffs),
            'min': np.min(payoffs),
            'max': np.max(payoffs),
            'non_zero_fraction': np.mean(payoffs > 0),
            'median': np.median(payoffs)
        }

        payoff_name = payoff.__class__.__name__

        return {
            'payoff_name': payoff_name,
            'price': price,
            'std_error': std_error,
            'computation_time': computation_time,
            'n_paths': n_paths,
            'n_coarse_steps': n_coarse_steps,
            'L': L,
            'final_I_stats': I_stats,
            'payoff_stats': payoff_stats,
            'I_paths': I_paths,
            'payoffs': payoffs
        }

    def compare_results(self, result_full, result_interp):
        """
        Compare two Monte Carlo result dictionaries and compute:
          - Absolute price difference
          - Relative price error (relative to the full‐simulation price)
          - Speed‐up factor (full_time / interp_time)
        """
        price_full = result_full.get('price')
        price_interp = result_interp.get('price')
        std_full = result_full.get('std_error')
        std_interp = result_interp.get('std_error')
        time_full = result_full.get('computation_time')
        time_interp = result_interp.get('computation_time')

        abs_price_diff = abs(price_full - price_interp)
        rel_price_error = abs_price_diff / price_full if price_full != 0 else np.nan
        speedup = time_full / time_interp if time_interp != 0 else np.nan

        return {
            'price_full': price_full,
            'price_interp': price_interp,
            'abs_price_diff': abs_price_diff,
            'rel_price_error': rel_price_error,
            'std_error_full': std_full,
            'std_error_interp': std_interp,
            'time_full': time_full,
            'time_interp': time_interp,
            'speedup_factor': speedup
        }

    def spot_ladder_analysis(self, payoffs_dict, T, n_paths, n_steps, spot_range, seed=2025):
        """
        Generate spot ladder with computation time ratios.
        
        Parameters:
        -----------
        payoffs_dict : dict
            Dictionary of payoff_name -> Payoff instance
        T : float
            Time to maturity in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path (for full simulation)
        spot_range : list or array-like
            Range of S0 values to test
        seed : int
            Base random seed
        """
        original_S0 = self.path_generator.S0
        spots = np.array(spot_range)

        results = {}
        for payoff_name in payoffs_dict.keys():
            results[payoff_name] = {
                'spots': spots,
                'prices': np.zeros_like(spots, dtype=float),
                'std_errors': np.zeros_like(spots, dtype=float),
                'comp_times': np.zeros_like(spots, dtype=float)
            }

        # Baseline computation times at S0 = 100
        baseline_times = {}
        for payoff_name, payoff in payoffs_dict.items():
            self.path_generator.S0 = 100.0
            result = self.price_full_simulation(payoff, T, n_paths, n_steps, seed)
            baseline_times[payoff_name] = result['computation_time']

        # Run spot ladder
        for i, spot in enumerate(spots):
            self.path_generator.S0 = spot
            for payoff_name, payoff in payoffs_dict.items():
                result = self.price_full_simulation(payoff, T, n_paths, n_steps, seed + i)
                results[payoff_name]['prices'][i] = result['price']
                results[payoff_name]['std_errors'][i] = result['std_error']
                results[payoff_name]['comp_times'][i] = result['computation_time']

        # Calculate time ratios
        for payoff_name in payoffs_dict.keys():
            results[payoff_name]['time_ratios'] = (
                results[payoff_name]['comp_times'] / baseline_times[payoff_name]
            )

        # Restore original S0
        self.path_generator.S0 = original_S0

        return results


# In[16]:


r = 0.03
q = 0.01
sigma = 0.30
S0 = 100.0

path_gen = PathGenerator(r, q, sigma, S0)

# ── Here: Smaller decrements ──────────────────────────────────────────────────
delta1 = 0.0005           # 0.05% daily decrement  (was 0.005)
delta2 = 0.001 * S0       # 0.1% of S0 = 0.10 every day  (was 1.0)
dt = 1 / 250
dec_index = DecrementIndex(delta1, delta2, q, dt)
# ───────────────────────────────────────────────────────────────────────────────

rng = RandomNumberGenerator()
mc = MonteCarlo(path_gen, dec_index, rng)

# 2) Define payoffs
K = 100.0
barrier_down = 90.0

times_obs = [0.25, 0.50, 0.75, 1.00]  # quarterly
coupon = 1.0
barrier_up = 105.0
barrier_down_for_put = 85.0
K_put = 95.0

vanilla_call = VanillaCall(K)
barrier_do    = BarrierDownOut(K, barrier_down)
auto_call     = AutoCallable(
    times_obs=times_obs,
    coupon=coupon,
    barrier_up=barrier_up,
    barrier_down=barrier_down_for_put,
    K_put=K_put,
    dt=dt
)

payoffs_dict = {
    'VanillaCall':    vanilla_call,
    'BarrierDownOut': barrier_do,
    'AutoCallable':   auto_call
}

# 3) Simulation settings
T = 1.0            # 1-year maturity
n_paths = 50000    # 50k Monte Carlo paths
n_steps = 250      # daily stepping
n_coarse_steps = 50
L = n_steps // n_coarse_steps   # = 250/50 = 5

# 4a) Full‐grid simulation
results_full = {}
for name, payoff in payoffs_dict.items():
    results_full[name] = mc.price_full_simulation(payoff, T, n_paths, n_steps, seed=2025)

# 4b) Interpolation‐based simulation
results_interp = {}
for name, payoff in payoffs_dict.items():
    results_interp[name] = mc.price_with_interpolation(
        payoff,
        T,
        n_paths,
        n_coarse_steps,
        L,
        seed=2025,
        use_cached_paths=False
    )

# 4c) Compare
comparison = {}
for name in payoffs_dict.keys():
    comparison[name] = mc.compare_results(results_full[name], results_interp[name])

# 5) Print results
for name in payoffs_dict.keys():
    print(f"─── {name} ───")
    print("Full‐grid Simulation:")
    print(f"  Price:      {results_full[name]['price']:.6f}")
    print(f"  Std Error:  {results_full[name]['std_error']:.6f}")
    print(f"  Time (sec): {results_full[name]['computation_time']:.6f}")
    print()
    print("Interpolation Simulation:")
    print(f"  Price:      {results_interp[name]['price']:.6f}")
    print(f"  Std Error:  {results_interp[name]['std_error']:.6f}")
    print(f"  Time (sec): {results_interp[name]['computation_time']:.6f}")
    print()
    print("Comparison Metrics:")
    comp = comparison[name]
    print(f"  Abs Price Diff:   {comp['abs_price_diff']:.6f}")
    print(f"  Rel Price Error:  {comp['rel_price_error']:.6%}")
    print(f"  Speed‐up Factor:  {comp['speedup_factor']:.6f}")
    print(f"  StdErr Full:      {comp['std_error_full']:.6f}")
    print(f"  StdErr Interp:    {comp['std_error_interp']:.6f}")
    print("────────────────────────────────────────────────\n")


# In[ ]:




