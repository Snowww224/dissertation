import numpy as np


class RandomNumberGenerator:
    """
    Wrapper for generating standard normal random numbers.
    Inputs:
      - n_paths: number of Monte Carlo paths.
      - n_steps: number of time steps per path.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def normals(self, n_paths, n_steps):
        """Generate an array of shape (n_paths, n_steps) of standard normal variates."""
        return self.rng.standard_normal(size=(n_paths, n_steps))


# In[5]:


rng = RandomNumberGenerator(seed=42)

n_paths = 10000  
n_steps = 252

#生成随机数矩阵，形状 (n_paths, n_steps)
Z = rng.normals(n_paths, n_steps)

print(Z.shape) 
print(Z[:3,:5])


# In[29]:


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


# In[30]:


# 1. 设置参数
r = 0.03       # 无风险利率
q = 0.01       # 股息率
sigma = 0.3    # 波动率
S0 = 100.0     # 初始价格

# 2. 构造时间格点（例如一年 252 个交易日）
n_steps = 252
times = np.linspace(0, 1, n_steps + 1)  # [0, 1/252, 2/252, ..., 1]

# 3. 生成随机数
n_paths = 10000
rng = RandomNumberGenerator(seed=123)
Z = rng.normals(n_paths, n_steps)  # 形状 (10000, 252)

# 4. 模拟路径
pg = PathGenerator(r, q, sigma, S0)
paths = pg.simulate(times, Z)      # 形状 (10000, 253)

# 5. 查看结果
print(paths.shape)      # → (10000, 253)
print(paths[0:5, :5])     # 打印第一条路径的前 5 个价格


# In[35]:


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


# In[10]:


class PathInterpolator:
    """
    Geometric interpolation of log-prices between periodically simulated stock values.
    Inputs:
      - S_coarse: array of shape (n_paths, M+1) with stock prices at coarse time points.
      - L: number of subintervals between each pair of coarse points.
    Outputs:
      - array of shape (n_paths, M*L + 1) with fully interpolated paths.
    """
    @staticmethod
    def interpolate(S_coarse, L):
        n_paths, M1 = S_coarse.shape
        M = M1 - 1
        full = []
        for j in range(M):
            start = S_coarse[:, j:j+1]
            end = S_coarse[:, j+1:j+2]
            # geometric interpolation via exponents from (0,1]
            exponents = np.linspace(0, 1, L+1)[1:]
            block = start * (end / start) ** exponents[np.newaxis, :]
            if j == 0:
                full.append(start)
            full.append(block)
        return np.concatenate(full, axis=1)


# In[20]:


class DecrementIndex:
    """
    Computes the decrement index I_t from simulated stock paths.

    Rule:
        I_{n} = max(0,
                    I_{n-1} * (1 - delta1) * (S_n / S_{n-1} + q * dt)
                    - delta2)
    Inputs:
      - delta1: percentage decrement (0 < delta1 < 1)
      - delta2: point decrement (>0)
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


# In[21]:


class Payoff:
    """
    Base class for payoffs.
    """
    def __call__(self, I_paths):
        raise NotImplementedError


# In[31]:


class VanillaCall(Payoff):
    def __init__(self, K):
        self.K = K

    def __call__(self, I_paths):
        payoffs = np.maximum(I_paths[:, -1] - self.K, 0)
        return payoffs


# In[33]:


class BarrierDownOut(Payoff):
    """
    Down-and-out barrier on index: zero if I falls below barrier,
    else vanilla call payoff on final level.
    """
    def __init__(self, K, barrier):
        self.K = K
        self.barrier = barrier

    def __call__(self, I_paths):
        n_paths = I_paths.shape[0]
        payoffs = np.maximum(I_paths[:, -1] - self.K, 0)
        touched = (I_paths <= self.barrier).any(axis=1)
        payoffs[touched] = 0
        return payoffs


# In[34]:


class AutoCallable(Payoff):
    """
    Auto-callable product on the decrement index I_t:
    periodic digital coupon plus Bermudan type knock-in put,
    both subject to periodic up-and-out barriers on I_t.
    """
    def __init__(self, times_obs, coupon, barrier_up, barrier_ki, K_put):
        self.times_obs = times_obs
        self.coupon = coupon
        self.barrier_up = barrier_up
        self.barrier_ki = barrier_ki
        self.K_put = K_put

    def __call__(self, I_paths):
        n_paths, N1 = I_paths.shape
        payoffs = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        knocked_in = (I_paths <= self.barrier_ki).any(axis=1)

        for obs in self.times_obs:
            trigger = alive & (I_paths[:, obs] >= self.barrier_up)
            payoffs[trigger] = self.coupon
            alive[trigger] = False

        final_obs = self.times_obs[-1]
        survivors = alive
        exec_idx = survivors & knocked_in
        payoffs[exec_idx] = np.maximum(self.K_put - I_paths[exec_idx, final_obs], 0)
        return payoffs


# In[ ]:


class IndexPricer(MCPricer):
    """
    Monte Carlo pricer specialized for decrement index payoffs.
    Inherits RNG, interpolator, and uses a Payoff object on I_paths.
    """
    def __init__(self, rng, interpolator):
        super().__init__(rng, interpolator, None)

    def price_index(self, times_coarse, L, n_paths, r, q, sigma, S0, delta1, delta2, dt, I0, payoff_obj):
        """
        1. Simulate and interpolate stock paths to fine grid.
        2. Compute decrement index paths.
        3. Evaluate payoff_obj on I_paths and return PV and variance.

        - payoff_obj: instance of Payoff subclass operating on I_paths
        """
        # 1. Stock simulation
        M = len(times_coarse) - 1
        normals = self.rng.normals(n_paths, M)
        pg = PathGenerator(r, q, sigma, S0)
        S_coarse = pg.simulate(times_coarse, normals)
        times_fine = np.linspace(times_coarse[0], times_coarse[-1], M * L + 1)
        S_fine = self.interpolator.interpolate(times_coarse, S_coarse, times_fine)

        # 2. Compute index paths
        decr = DecrementIndex(delta1, delta2, q, dt, I0)
        I_paths = decr.compute(S_fine)

        # 3. Payoff evaluation
        pay_vals = payoff_obj(I_paths)
        # Compute present value (PV) as the average payoff
        pv = np.mean(pay_vals)
        # Optionally estimate variance of the payoff
        var = np.var(pay_vals, ddof=1)
        return pv, var

