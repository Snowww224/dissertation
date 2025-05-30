import numpy as np
import time
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
class Payoff:
    """
    Base class for payoffs.
    """
    def __call__(self, I_paths):
        raise NotImplementedError
class VanillaCall(Payoff):
    """Vanilla call option on the decrement index."""
    def __init__(self, K):
        self.K = K
    
    def __call__(self, I_paths):
        payoffs = np.maximum(I_paths[:, -1] - self.K, 0)
        return payoffs
class VanillaPut(Payoff):
    """Vanilla put option on the decrement index."""
    def __init__(self, K):
        self.K = K
    
    def __call__(self, I_paths):
        payoffs = np.maximum(self.K - I_paths[:, -1], 0)
        return payoffs
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
class AutoCallable(Payoff):
    """
    Auto-callable on the decrement index I_t with:
      - Periodic digital coupon at quarterly observations
      - Bermudan-style knock-in put if ever <= barrier_ki
      - Up-and-out barrier enforced only at each observation date
      - Optional Broadie-Glasserman correction for barrier monitoring
    """
    
    def __init__(self, times_obs, coupon, barrier_up, barrier_ki, K_put, 
                 dt=1/250, use_bg_correction=True, volatility=None):
        """
        Parameters:
        -----------
        times_obs : list of float
            Observation times in years (e.g. [0.25, 0.5, 0.75, 1.0])
        coupon : float
            Digital coupon amount paid on early exercise
        barrier_up : float
            Up-and-out barrier level (triggers early exercise)
        barrier_ki : float
            Knock-in barrier level (activates put protection)
        K_put : float
            Strike price for the put option
        dt : float
            Time step in years (default: 1/250 for daily)
        use_bg_correction : bool
            Whether to apply Broadie-Glasserman correction
        volatility : float
            Annualized volatility (required for BG correction)
        """
        self.times_obs = np.array(times_obs)
        self.obs_idx = [int(round(t_year / dt)) for t_year in times_obs]
        self.coupon = coupon
        self.barrier_up = barrier_up
        self.barrier_ki = barrier_ki
        self.K_put = K_put
        self.dt = dt
        self.use_bg_correction = use_bg_correction
        self.volatility = volatility
        
        if use_bg_correction and volatility is None:
            raise ValueError("Volatility must be provided for Broadie-Glasserman correction")
    
    def broadie_glasserman_correction(self, barrier, dt, volatility):
        """
        Apply Broadie-Glasserman correction factor for discrete barrier monitoring.
        
        For up-and-out barriers, the correction factor is:
        β = exp(λ * σ√dt) where λ ≈ 0.5826
        
        Returns the adjusted barrier level.
        """
        lambda_bg = 0.5826  # Broadie-Glasserman constant
        correction_factor = np.exp(lambda_bg * volatility * np.sqrt(dt))
        return barrier / correction_factor  # Lower the barrier for up-and-out
    
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
        
        # Determine knock-in status (continuous monitoring)
        knocked_in = (I_paths <= self.barrier_ki).any(axis=1)
        
        # Apply Broadie-Glasserman correction to up barrier if requested
        effective_barrier_up = self.barrier_up
        if self.use_bg_correction:
            effective_barrier_up = self.broadie_glasserman_correction(
                self.barrier_up, self.dt, self.volatility
            )
        
        # Check up-and-out barrier only at observation dates
        for i, obs_idx in enumerate(self.obs_idx):
            if obs_idx >= n_steps:
                continue
                
            # Check which alive paths breach the up barrier
            trigger = alive & (I_paths[:, obs_idx] >= effective_barrier_up)
            
            # Pay coupon and terminate these paths
            payoffs[trigger] = self.coupon * (i + 1)  # Cumulative coupon
            alive[trigger] = False
        
        # Handle survivors at final observation
        if len(self.obs_idx) > 0:
            final_idx = min(self.obs_idx[-1], n_steps - 1)
            survivors = alive
            
            # Paths that were knocked in get put protection
            knocked_in_survivors = survivors & knocked_in
            payoffs[knocked_in_survivors] = np.maximum(
                self.K_put - I_paths[knocked_in_survivors, final_idx], 0.0
            )
            
            # Paths that were never knocked in get their principal back
            never_knocked_in = survivors & ~knocked_in
            payoffs[never_knocked_in] = 1.0  # Assuming normalized principal
        
        return payoffs
        
class MonteCarlo:
    """
        Monte Carlo pricer for decrement index derivatives.
    Handles both full simulation and interpolation-based acceleration.
    """
    
    def __init__(self, path_generator, decrement_index, rng=None):
        """
        Parameters:
        -----------
        path_generator : PathGenerator
            Generator for underlying asset paths
        decrement_index : DecrementIndex
            Decrement index calculator
        rng : RandomNumberGenerator, optional
            Random number generator (creates default if None)
        """
        self.path_generator = path_generator
        self.decrement_index = decrement_index
        self.rng = rng if rng is not None else RandomNumberGenerator()
        # Store stock paths for reuse between methods
        self._cached_stock_paths = None
        self._cached_seed = None
    
    def price_full_simulation(self, payoff, T, n_paths, n_steps, seed=2025):
        """
        Price using full simulation (no interpolation).
        
        Parameters:
        -----------
        payoff : Payoff
            Payoff function to evaluate
        T : float
            Time to maturity in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int
            Random seed
            
        Returns:
        --------
        dict
            Dictionary containing price, standard error, and timing info
        """
        start_time = time.time()
        
        # Generate time grid
        times = np.linspace(0, T, n_steps + 1)
        
        # Generate random numbers and simulate stock paths
        normals = self.rng.normals(n_paths, n_steps, seed)
        S_paths = self.path_generator.simulate(times, normals)
        
        # Cache stock paths for reuse
        self._cached_stock_paths = S_paths
        self._cached_seed = seed
        
        # Compute decrement index paths
        I_paths = self.decrement_index.compute(S_paths)
        
        # Compute payoffs
        payoffs = payoff(I_paths)
        
        # Calculate statistics
        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        
        end_time = time.time()
        
        return {
            'price': price,
            'std_error': std_error,
            'computation_time': end_time - start_time,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'method': 'full_simulation'
        }
    
    def price_with_interpolation(self, payoff, T, n_paths, n_coarse_steps, 
                                L, seed=2025, use_cached_paths=True):
        """
        Price using interpolation-based acceleration.
        
        Parameters:
        -----------
        payoff : Payoff
            Payoff function to evaluate
        T : float
            Time to maturity in years
        n_paths : int
            Number of Monte Carlo paths
        n_coarse_steps : int
            Number of coarse time steps for simulation
        L : int
            Number of interpolation points between each coarse step
        seed : int
            Random seed
        use_cached_paths : bool
            Whether to use cached stock paths from previous full simulation
            
        Returns:
        --------
        dict
            Dictionary containing price, standard error, and timing info
        """
        start_time = time.time()
        
        # Generate coarse time grid
        times_coarse = np.linspace(0, T, n_coarse_steps + 1)
        
        # Use cached stock paths if available
        if (use_cached_paths and self._cached_stock_paths is not None and 
            self._cached_seed == seed):
            # Extract coarse points directly from cached full paths
            step_size = (self._cached_stock_paths.shape[1] - 1) // n_coarse_steps
            indices = np.arange(0, self._cached_stock_paths.shape[1], step_size)[:n_coarse_steps + 1]
            indices[-1] = self._cached_stock_paths.shape[1] - 1  # Ensure final point
            S_coarse = self._cached_stock_paths[:, indices]
        else:
            # Generate new simulation
            normals_coarse = self.rng.normals(n_paths, n_coarse_steps, seed)
            S_coarse = self.path_generator.simulate(times_coarse, normals_coarse)
        
        # Interpolate to fine grid
        S_fine = PathInterpolator.interpolate(times_coarse, S_coarse, L)
        
        # Compute decrement index paths on fine grid
        I_paths = self.decrement_index.compute(S_fine)
        
        # Compute payoffs
        payoffs = payoff(I_paths)
        
        # Calculate statistics
        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        
        end_time = time.time()
        
        return {
            'price': price,
            'std_error': std_error,
            'computation_time': end_time - start_time,
            'n_paths': n_paths,
            'n_coarse_steps': n_coarse_steps,
            'n_fine_steps': n_coarse_steps * L,
            'interpolation_factor': L,
            'method': 'interpolation',
            'used_cached_paths': use_cached_paths and self._cached_stock_paths is not None
        }
    
    def compare_methods(self, payoff, T, n_paths, n_steps_full, n_coarse_steps, 
                       L, seed=2025):
        """
        Compare full simulation vs interpolation method using the same random numbers.
        
        Returns:
        --------
        dict
            Dictionary with results from both methods and comparison metrics
        """
        # Full simulation (this will cache the normals)
        result_full = self.price_full_simulation(
            payoff, T, n_paths, n_steps_full, seed
        )
        
        # Interpolation method using cached stock paths
        result_interp = self.price_with_interpolation(
            payoff, T, n_paths, n_coarse_steps, L, seed, use_cached_paths=True
        )
        
        # Calculate comparison metrics
        price_diff = abs(result_full['price'] - result_interp['price'])
        speedup = result_full['computation_time'] / result_interp['computation_time']
        
        return {
            'full_simulation': result_full,
            'interpolation': result_interp,
            'price_difference': price_diff,
            'speedup_factor': speedup,
            'relative_error': price_diff / result_full['price'] if result_full['price'] != 0 else 0
        }
    
    def spot_ladder(self, payoff, T, n_paths, n_steps, spot_range, seed=2025):
        """
        Generate a spot ladder showing price sensitivity to initial spot.
        
        Parameters:
        -----------
        payoff : Payoff
            Payoff function to evaluate
        T : float
            Time to maturity
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps
        spot_range : array-like
            Range of initial spot prices to test
        seed : int
            Random seed
            
        Returns:
        --------
        dict
            Dictionary with spots and corresponding prices
        """
        original_S0 = self.path_generator.S0
        spots = np.array(spot_range)
        prices = np.zeros_like(spots)
        std_errors = np.zeros_like(spots)
        
        for i, spot in enumerate(spots):
            # Update initial spot
            self.path_generator.S0 = spot
            
            # Price the option
            result = self.price_full_simulation(payoff, T, n_paths, n_steps, seed)
            prices[i] = result['price']
            std_errors[i] = result['std_error']
        
        # Restore original spot
        self.path_generator.S0 = original_S0
        
        return {
            'spots': spots,
            'prices': prices,
            'std_errors': std_errors
        }
if __name__ == "__main__":
    # Market parameters
    r = 0.03
    q = 0.01
    sigma = 0.3
    S0 = 100.0
    delta1 = 0.05
    delta2 = 1.0
    dt = 1 / 250
    T = 1.0
    K = 95.0
    path_gen = PathGenerator(r, q, sigma, S0)
    decr_index = DecrementIndex(delta1, delta2, q, dt)
    mc = MonteCarlo(path_gen, decr_index)
    vanilla_call = VanillaCall(K)
    barrier_down_out = BarrierDownOut(K, barrier=80.0)
    auto_callable = AutoCallable(
        times_obs=[0.25, 0.5, 0.75, 1.0],
        coupon=0.10,
        barrier_up=110.0,
        barrier_ki=75.0,
        K_put=100.0,
        dt=dt,
        use_bg_correction=True,
        volatility=sigma
    )
    n_paths = 10000
    n_steps = 250
    n_coarse_steps = 50
    L = 5
    print("Monte Carlo Pricing System for Decrement Index Derivatives")
    print("=" * 60)

    # Vanilla Call pricing
    print(f"\nPricing Vanilla Call (K={K}):")
    result_vanilla = mc.price_full_simulation(vanilla_call, T, n_paths, n_steps)
    print(f"Price: {result_vanilla['price']:.4f} ± {result_vanilla['std_error']:.4f}")
    print(f"Time: {result_vanilla['computation_time']:.2f}s")

    # Barrier Down-and-Out Call pricing
    print(f"\nPricing Barrier Down-and-Out Call (K={K}, Barrier=80):")
    result_barrier = mc.price_full_simulation(barrier_down_out, T, n_paths, n_steps)
    print(f"Price: {result_barrier['price']:.4f} ± {result_barrier['std_error']:.4f}")
    print(f"Time: {result_barrier['computation_time']:.2f}s")

    # Compare full simulation vs interpolation
    print("\nComparing full simulation vs interpolation:")
    compare_res = mc.compare_methods(vanilla_call, T, n_paths, n_steps, n_coarse_steps, L)
    full = compare_res['full_simulation']
    interp = compare_res['interpolation']
    print(f"Full price: {full['price']:.4f}, Interp price: {interp['price']:.4f}")
    print(f"Price diff: {compare_res['price_difference']:.6f}")
    print(f"Speedup factor: {compare_res['speedup_factor']:.2f}x")

    # Spot ladder sensitivity
    print("\nSpot Ladder Sensitivity (Vanilla Call):")
    spot_range = [80, 90, 100, 110, 120]
    ladder = mc.spot_ladder(vanilla_call, T, n_paths, n_steps, spot_range)
    for spot, price, se in zip(ladder['spots'], ladder['prices'], ladder['std_errors']):
        print(f"S0={spot:.0f}: Price={price:.4f} ± {se:.4f}")
