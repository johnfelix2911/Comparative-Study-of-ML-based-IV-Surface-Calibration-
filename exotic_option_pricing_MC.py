# %%
import yfinance as yf
import pandas as pd
import numpy as np

# Configuration
tickers = ["NVDA", "AMZN", "TSLA"]
risk_free_rate = 0.05  # Assumption
target_Ts = [0.5, 1.0] # desired maturities in years

# Fetch Calibration Data
option_data_list = []

for symbol in tickers:
    ticker_obj = yf.Ticker(symbol)
    expirations = ticker_obj.options
    
    # Finding the closest maturities to the desired maturities
    selected_expirations = []
    for target in target_Ts:
        available_Ts = []
        for exp in expirations:
            days = (pd.to_datetime(exp) - pd.Timestamp.today()).days
            available_Ts.append(days / 365.0)
        
        # Find the index of the T closest to desired maturities (0.5 or 1.0)
        closest_idx = np.argmin([abs(t - target) for t in available_Ts])
        selected_expirations.append((expirations[closest_idx], available_Ts[closest_idx]))

    # Fetch data only for these selected expirations
    for exp_date, actual_T in selected_expirations:
        opt_chain = ticker_obj.option_chain(exp_date)
        calls = opt_chain.calls

        for _, row in calls.iterrows():
            option_data_list.append({
                "Asset": symbol,
                "K": row['strike'],
                "T": round(actual_T, 4),
                "Price": row['lastPrice'] 
            })

df_calibration = pd.DataFrame(option_data_list)

# Fetch Historical Data (For Correlation Matrix)
print("Fetching historical prices for correlation: ")
hist_prices = yf.download(tickers, period="3y")['Close'] 
returns = np.log(hist_prices).diff().dropna()
corr_matrix = returns.corr()

print("\n--- Calibration Data Sample ---")
print(df_calibration[['Asset', 'T', 'K', 'Price']].drop_duplicates('T'))

print("\n--- Estimated Correlation Matrix ---")
print(corr_matrix)

# %%
tickers = ["AMZN", "NVDA", "TSLA"] # in the order in which they are present in the correlation matrix

# Fetching exact current spot prices
S0 = {}
for ticker in tickers:
    # Fetching the most recent 1-day history to get the last closing price
    ticker_data = yf.Ticker(ticker)
    recent_history = ticker_data.history(period="1d")
    
    if not recent_history.empty:
        # The spot price is the last 'Close' price in the series
        S0[ticker] = recent_history['Close'].iloc[-1]
    else:
        print(f"Warning: Could not fetch price for {ticker}")

print("Exact Spot Prices (S0):")
for asset, price in S0.items():
    print(f"{asset}: {price:.2f}")

# %%
# calculating the cholesky decomp half matrix L from LL^T=rho
# This matrix helps in making the independent normal variables to become correlated in such a way that it mimics the correlated behavior of the actual assets 
eigen_vals=np.linalg.eigvals(corr_matrix)
print(f"The eigen values of the correlation matrix are: ",eigen_vals)
if np.all(eigen_vals>0):
    cholesky_L=np.linalg.cholesky(corr_matrix)
else:
    print(f"correlation matrix must be positive semi definite")

# %%
import scipy.stats as st
from scipy.optimize import brentq

# The black-Scholes-Merton pricing model
def bs_call_price(S,K,T,r,sigma):
    if sigma<=0 or T<=0:
        return max(S-K,0)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return S*st.norm.cdf(d1)-K*np.exp(-r*T)*st.norm.cdf(d2)

# This function returns the implied volatility using the root finding algorithm brentq
def implied_volatility(price,S,K,T,r):
    f=lambda sigma:bs_call_price(S,K,T,r,sigma)-price
    return brentq(f,1e-6,5.0)

r=risk_free_rate
implied_vols = []

for index, row in df_calibration.iterrows():
    S = S0[row['Asset']]
    K, T, price = row['K'], row['T'], row['Price']
    
    # # Apply Strict Filtering
    # if price < 0.10:
    #     continue  # Skip unstable/illiquid data
        
    # Arbitrage Bound Check
    lower_bound = max(S - K * np.exp(-risk_free_rate * T), 0)
    if price <= lower_bound + 0.01 or price >= S:
        continue

    # Solve for IV
    try:
        iv = implied_volatility(price, S, K, T, risk_free_rate)
        implied_vols.append((row['Asset'], T, K, iv))
    except ValueError:
        print("skipped")
        pass # Skip if still failing to converge
        # But as you can see nothing was skipped here because of the arbitrage bound check above

df_iv = pd.DataFrame(implied_vols, columns=['Asset', 'T', 'K', 'ImpliedVol'])

# %%
df_iv.head(10)

# %%
from scipy.interpolate import griddata

vol_grids={}

for asset in ["AMZN", "NVDA", "TSLA"]:
    df_a = df_iv[df_iv.Asset == asset]
    
    # points (pts) must be shape (N, 2)
    pts = df_a[['K', 'T']].values 
    vals = df_a['ImpliedVol'].values
    
    vol_grids[asset] = (pts, vals)

# %%
vol_surface={}
for asset, (pts, vals) in vol_grids.items():
    current_S = S0[asset]
    
    # Create a grid from 80% to 120% of the current spot price
    strike_grid = np.linspace(current_S * 0.8, current_S * 1.2, 5)
    time_grid = np.array([0.1,0.5,1.0])
    
    surf = np.zeros((len(time_grid), len(strike_grid)))
    
    for i, T in enumerate(time_grid):
        for j, K in enumerate(strike_grid):
            res = griddata(pts, vals, (K, T), method='linear', fill_value=np.mean(vals))
            surf[i, j] = res
            
    vol_surface[asset] = surf
    print(f"{asset} Surface Calibrated centered at {current_S:.2f}")
    print(f"{asset} vol surface (T vs K):\n {surf.round(3)}")

# %%
import matplotlib.pyplot as plt

def monte_carlo_basket_barrier(S0_vec, K_opt, H_barrier, T, r, vol_surface, corr_L, N_paths=20000, M_steps=250, plot_n=150):
    dt = T / M_steps
    S_paths = np.tile(S0_vec, (N_paths, 1)) 
    alive = np.ones(N_paths, dtype=bool) 
    
    basket_history = np.zeros((M_steps + 1, N_paths))
    basket_history[0] = np.mean(S0_vec) # Initial basket value

    for step in range(M_steps):
        Z = np.random.normal(size=(N_paths, 3))
        dW = Z.dot(corr_L.T) 

        t = step * dt
        tau = T - t

        for i, asset in enumerate(["AMZN", "NVDA", "TSLA"]):
            S_prev = S_paths[:, i]
            pts, vals = vol_surface[asset]
            
            sigma = griddata(pts, vals, (S_prev, tau), method='linear', fill_value=vals.mean())
            
            S_paths[:, i] = S_prev * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW[:, i])
 
        current_basket = S_paths.mean(axis=1)
        basket_history[step + 1] = current_basket
        
        knocked = (current_basket >= H_barrier)
        alive &= ~knocked 

    # Plotting
    time_axis = np.linspace(0, T, M_steps + 1)
    plt.figure(figsize=(12, 6))
    
    # Plot a subset of paths (plot_n)
    for p in range(min(plot_n, N_paths)):
        color = 'green' if alive[p] else 'red' # Green for surviving, red for knocked-out [cite: 6]
        plt.plot(time_axis, basket_history[:, p], color=color, alpha=0.5, linewidth=1)

    # Plot Barrier and Strike lines [cite: 240, 248]
    plt.axhline(y=H_barrier, color='black', linestyle='--', label=f'Knock-out Barrier ({H_barrier})')
    plt.axhline(y=K_opt, color='blue', linestyle=':', label=f'Strike Price ({K_opt})')
    
    plt.title(f"Monte Carlo Basket Paths ({plot_n} samples)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Basket Average Price")
    plt.legend()
    plt.show()

    final_basket = S_paths.mean(axis=1)
    payoffs = np.where(alive, np.maximum(final_basket - K_opt, 0.0), 0.0)
    price = np.exp(-r * T) * payoffs.mean()
    std_err = (np.exp(-r * T) * payoffs).std(ddof=1) / np.sqrt(N_paths)
    
    return price, std_err

# %%
S0=[S0['AMZN'],S0['NVDA'],S0['TSLA']]

# %% [markdown]
# Here, there I have demostrated two cases to show the effect of the variance reduction techniques (such as antithetic variate and control variate methods).
# 
# In the first case where:  
# K=300  
# barrier=1000  
# T=1.0  
# We can see that the price of the option is high because this probability of this option getting knocked out is very less. Another thing to notice is that the antithetic variance reduction technique worked well in this case because the "lucky" and the "unlucky" twins both were not knocked out because of the high barrier hence not interfering with this variance reduction technique and giving a reduced standard error as compared to the standard error from the vanilla monte carlo
# 
# In the second case where:  
# K=300  
# barrier=500  
# T=1.0  
# Basically I only changed the barrier value to 500. This brings down the price of the option by a lot because now there is a much higher chance of the spot price crossing the barrier and thus knocking it out. Also another thing to notice here is that antithetic variate variancec reduction technique did not work as well here because a significant number of the "lucky" twins were knocked out thus interfering with the fast convergence of this variance reduction technique

# %%
K_opt=300        # basket strike
H_barrier=1000   # knockout barrier on basket
T=1.0            # 1 year
price, stderr = monte_carlo_basket_barrier(S0, K_opt, H_barrier, T, r, vol_grids, cholesky_L)

# %%
print(f"Basket call price: {price:.2f} ± {1.96*stderr:.2f} (95% CI)")

# %%
K_opt=300        # basket strike
H_barrier=500    # knockout barrier on basket
T=1.0            # 1 year
price, stderr = monte_carlo_basket_barrier(S0, K_opt, H_barrier, T, r, vol_grids, cholesky_L)

# %%
print(f"Basket call price: {price:.2f} ± {1.96*stderr:.2f} (95% CI)")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def monte_carlo_basket_barrier_antithetic(S0_vec, K_opt, H_barrier, T, r, vol_surface, corr_L, N_paths=20000, M_steps=250):
    dt = T / M_steps
    num_pairs = N_paths // 2
    
    S1 = np.tile(S0_vec, (num_pairs, 1))
    S2 = np.tile(S0_vec, (num_pairs, 1))
    
    alive1 = np.ones(num_pairs, dtype=bool)
    alive2 = np.ones(num_pairs, dtype=bool)

    for step in range(M_steps):
        Z = np.random.normal(size=(num_pairs, 3))
        dW1 = Z.dot(corr_L.T)
        dW2 = -dW1  # antithetic mirror

        tau = T - (step * dt)

        for i, asset in enumerate(["AMZN", "NVDA", "TSLA"]):
            pts, vals = vol_surface[asset]
            v_mean = vals.mean()
            
            sig1 = griddata(pts, vals, (S1[:, i], tau), method='linear', fill_value=v_mean)
            sig2 = griddata(pts, vals, (S2[:, i], tau), method='linear', fill_value=v_mean)
            
            S1[:, i] *= np.exp((r - 0.5 * sig1**2) * dt + sig1 * np.sqrt(dt) * dW1[:, i])
            S2[:, i] *= np.exp((r - 0.5 * sig2**2) * dt + sig2 * np.sqrt(dt) * dW2[:, i])

        alive1 &= (S1.mean(axis=1) < H_barrier)
        alive2 &= (S2.mean(axis=1) < H_barrier)

    payoff1 = np.where(alive1, np.maximum(S1.mean(axis=1) - K_opt, 0.0), 0.0)
    payoff2 = np.where(alive2, np.maximum(S2.mean(axis=1) - K_opt, 0.0), 0.0)
    
    pair_payoffs = (payoff1 + payoff2) / 2.0
    
    discounted_payoffs = np.exp(-r * T) * pair_payoffs
    price = discounted_payoffs.mean()
    
    std_err = discounted_payoffs.std(ddof=1) / np.sqrt(num_pairs)
    
    return price, std_err

# %%

K_opt=300        # basket strike
H_barrier=1000   # knockout barrier on basket
T=1.0            # 1 year
price,stdee=monte_carlo_basket_barrier_antithetic(S0, K_opt, H_barrier, T, r, vol_grids, cholesky_L)
print(f"Basket call price: {price:.2f} ± {1.96*stdee:.2f} (95% CI)")

# %%
K_opt=300        # basket strike
H_barrier=500    # knockout barrier on basket
T=1.0            # 1 year
price,stdee=monte_carlo_basket_barrier_antithetic(S0, K_opt, H_barrier, T, r, vol_grids, cholesky_L)
print(f"Basket call price: {price:.2f} ± {1.96*stdee:.2f} (95% CI)")

# %% [markdown]
# ## Vanilla Option Pricing Engine (Tesla)
# A plain European call on a single asset (TSLA). This reuses the same
# calibrated implied-vol surface (`vol_grids['TSLA']`) as the exotic engine,
# but drops the three features that made the option exotic: the basket
# averaging across assets, the knockout barrier, and the correlated
# multi-asset simulation (Cholesky). What remains is a single-asset GBM
# Monte Carlo with a per-step local-vol lookup.

# %%
def monte_carlo_vanilla(S0_scalar, K_opt, T, r, vol_surface_asset, N_paths=20000, M_steps=250):
    dt = T / M_steps
    S = np.full(N_paths, S0_scalar)      # one price path per Monte Carlo sample
    pts, vals = vol_surface_asset
    v_mean = vals.mean()

    for step in range(M_steps):
        Z = np.random.normal(size=N_paths)   # independent draws, no correlation
        tau = T - (step * dt)
        sigma = griddata(pts, vals, (S, tau), method='linear', fill_value=v_mean)
        S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Vanilla call payoff: no barrier, no basket averaging
    discounted_payoffs = np.exp(-r * T) * np.maximum(S - K_opt, 0.0)
    price = discounted_payoffs.mean()
    std_err = discounted_payoffs.std(ddof=1) / np.sqrt(N_paths)
    return price, std_err

# %%
S0_tsla = S0[2]          # TSLA spot (S0 is [AMZN, NVDA, TSLA] at this point)
K_opt = S0_tsla          # at-the-money strike
T = 1.0                  # 1 year
price, stderr = monte_carlo_vanilla(S0_tsla, K_opt, T, r, vol_grids['TSLA'])
print(f"TSLA vanilla call price (K={K_opt:.2f}, T={T}): {price:.2f} ± {1.96*stderr:.2f} (95% CI)")


