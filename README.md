# KS Distance to Brownian Motion for ETFs

Compares the empirical distribution of ETF returns to a Brownian motion (normal) distribution using the Kolmogorov-Smirnov test. Brownian parameters are adjusted using the composite macro factor. The per‑ETF score is the KS statistic – a measure of deviation from randomness.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Empirical CDF vs theoretical normal CDF
- Macro‑adjusted drift and volatility
- Score = KS statistic (higher = more structured/predictable)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-ks-brownian-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High KS statistic → returns deviate from Brownian motion → structured/predictable.
- Low KS statistic → close to random walk.

## Requirements

See `requirements.txt`.
