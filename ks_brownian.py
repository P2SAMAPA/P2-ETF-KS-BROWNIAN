import numpy as np
from scipy.stats import norm, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def brownian_cdf(x, mu=0.0, sigma=1.0):
    """CDF of a Brownian motion increment (normal distribution)."""
    return norm.cdf(x, mu, sigma)

def ks_brownian_score(returns, macro_factor=None):
    """
    Compute Kolmogorov-Smirnov distance between empirical return distribution
    and Brownian motion (normal) with macro-dependent parameters.
    """
    if len(returns) < 10:
        return 0.0
    # If macro factor provided, use it to adjust the Brownian parameters
    if macro_factor is not None:
        # Higher macro -> higher volatility (sigma) and drift (mu)
        mu = macro_factor * 0.01
        sigma = 0.01 + macro_factor * 0.02
    else:
        mu = 0.0
        sigma = np.std(returns)
    # Empirical CDF
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    empirical_cdf = np.arange(1, n+1) / n
    # Theoretical CDF
    theoretical_cdf = brownian_cdf(sorted_returns, mu, sigma)
    # KS statistic: max absolute difference
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    return float(ks_stat)
