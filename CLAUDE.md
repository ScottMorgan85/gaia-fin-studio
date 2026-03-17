# CLAUDE.md — GAIA Fin Studio

Developer reference for Claude Code and human contributors.

---

## Known bugs / fix log

| ID  | Status  | Description |
|-----|---------|-------------|
| BUG 1 | ✅ Fixed | `_log_decision` NameError — function was called on Accept/Reject but never defined. Added definition after `REC_LOG_PATH`. |

---

## Key helpers

### `get_groq_key() -> str`
Defined at the top of `gaia_pages.py`.

```python
return os.environ.get("GROQ_API_KEY", "")
```

**Rule:** Every place in `gaia_pages.py` that needs the Groq API key must call `get_groq_key()`. No `st.secrets` — env var only.

### `get_fred_key() -> str`
Defined in `utils.py`.

```python
return os.environ.get("FRED_API_KEY", "f4ac14beb82a2e5cf49e141465baa458")
```

Hardcoded fallback is the registered public key. **DigitalOcean:** add `FRED_API_KEY` as an env var in App Platform settings.

### `_flag(name, default)` — feature flags
Defined in both `app.py` and `gaia_pages.py`. Reads only from `os.getenv()` — no `st.secrets`.

### RULE — no `st.secrets` anywhere
`st.secrets` is banned across all files. DigitalOcean App Platform does not ship a `secrets.toml`, causing "No secrets files found" log spam and potential crashes. All secrets and feature flags must use `os.environ.get()` only.

### RULE — `set_page_config` must be first
`st.set_page_config()` is called at line 7 of `app.py`, immediately after `import streamlit as st`, before any other `st.*` call including `st.session_state`, `st.sidebar`, or imported module code. Never move it or add any `st.*` call above it.

---

## Models in use

| Constant / variable  | Model ID | Where |
|----------------------|----------|-------|
| `model_primary` (Forecast Lab) | `llama-3.3-70b-versatile` | `display_forecast_lab` |
| `model_fallback` (Forecast Lab) | `meta-llama/llama-4-scout-17b-16e-instruct` | `display_forecast_lab` |
| DTD commentary primary | `llama-3.3-70b-versatile` | `generate_dtd_commentary` |
| DTD commentary fallback | `meta-llama/llama-4-scout-17b-16e-instruct` | `generate_dtd_commentary` |
| Scenario Allocator trade ideas | `llama-3.3-70b-versatile` | `display_scenario_allocator` |
| LLM recs for strategy | `llama-3.3-70b-versatile` | `_llm_recs_for_strategy` |

**Rule:** The old fallback `llama-3.1-8b-instant` has been retired. Use
`meta-llama/llama-4-scout-17b-16e-instruct` as the fallback everywhere.

---

## Secrets

Local dev: `.streamlit/secrets.toml` (never commit).
DigitalOcean: Settings → Environment Variables (flat `GROQ_API_KEY=...`).

| Key | Required | Where to get |
|-----|----------|--------------|
| `GROQ_API_KEY` | Yes | console.groq.com |
| `FRED_API_KEY` | Optional | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) — free registration |

**FRED_API_KEY note:** The FRED REST API requires a real registered key. Without it, `get_macro_data()` returns an empty DataFrame and Forecast Lab / Market Pulse fall back to synthetic data gracefully. The demo key `b722a33d9fe927f7fe3e494aeeed3e0e` in the spec is **not valid** — register at FRED for a free key and add it to `secrets.toml` as `FRED_API_KEY = "..."`.

---

## Features

### Quantum Studio
A quantum-inspired portfolio optimization PoC that uses simulated annealing to mimic
QUBO-style optimization across six synthetic sleeve allocations anchored to live strategy returns.

| Field | Value |
|-------|-------|
| Function | `display_quantum_studio()` in `gaia_pages.py` |
| Tab name | `Quantum Studio` |
| Route key | `quantum` |
| Dependencies | `numpy`, `plotly` (already in requirements.txt) |
| Status | PoC — simulated annealing mimicking QUBO-style optimization; no new pip packages |

### Forecast Lab
Full probabilistic simulation and macro analysis overlay for the selected client strategy.

| Field | Value |
|-------|-------|
| Function | `display_forecast_lab()` in `gaia_pages.py` |
| Tab name | `Forecast Lab` |
| Route key | `forecast` |
| Dependencies | `numpy`, `plotly`, `groq` |

**Key implementation details:**

| Fix | Description |
|-----|-------------|
| FIX 1 | Level→return detection: if `abs().mean() > 5.0`, apply `pct_change()`. Returns clipped to `[-20%, +20%]`. |
| FIX 2 | Macro table shows YoY% for CPI/GDP; columns formatted via `st.column_config.NumberColumn` with `format="%.1f%%"`. |
| FIX 3 | Block bootstrap simulation (block_size=6 months) with 1,000 paths over 60-month horizon to preserve autocorrelation. |
| FIX 4 | Scenario table shows $10,000 terminal values (base / bull / bear / custom) — not percentages. |
| FIX 5 | Fan chart y-axis in dollars ($8k–$25k expected range); 10/25/50/75/90th percentile bands. |
| FIX 6 | Terminal value violin plot in dollars. |
| FIX 7 | Regime analysis: 2×2 GDP growth × CPI momentum matrix → Goldilocks / Reflation / Stagflation / Deflation. |
| FIX 8 | AI trade ideas behind `st.button("Generate AI Ideas")` with macro-aware prompt (no auto-run on page load). |
| FIX 9 | Methodology & Statistical Disclosures expander with block-bootstrap description. |

**Level-vs-return detection rule:** `abs().mean() > 5.0` implies the series is in price/index levels — apply `pct_change()` before any simulation.

**Regime matrix:**
```
            CPI mom ≥ 0          CPI mom < 0
GDP > 0   Goldilocks / Reflation
GDP ≤ 0   Stagflation            Deflation
```

---

## Data Layer — utils.py functions

All five functions are in `utils.py` (appended at end). All cache with `@st.cache_data`, return empty df/dict on failure, never crash.

| Function | TTL | Description |
|----------|-----|-------------|
| `get_market_data()` | 1hr | yfinance: monthly prices+returns for 19 tickers, daily VIX, factor returns |
| `get_macro_data()` | 24hr | FRED REST API: 14 macro series → monthly DataFrame. **Requires `FRED_API_KEY` in secrets.toml.** Falls back to empty df. |
| `get_derived_signals()` | 1hr | Computes momentum (12-1), regime score (-2→+2), vol regime, yield curve shape, 24-month rolling corr matrix |
| `enrich_client_data()` | 1hr | Per-strategy risk metrics: 1/3/5yr returns, sharpe, sortino, calmar, beta, alpha, up/down capture vs SPY |
| `get_upcoming_events()` | 12hr | yfinance earnings calendar for 10 tickers + hardcoded FOMC dates |
| `get_benchmark_returns()` | 1hr | Total-return performance table for 6 benchmarks: S&P 500, MSCI EAFE, US Agg Bond, DJ Commodity, HY Credit (HYG proxy), Lev Loan (BKLN proxy). Columns: DTD, MTD, QTD, YTD (cumulative); 1yr/3yr/5yr (annualized). |
| `load_strategy_returns()` | 1hr | Loads `strategy_returns.xlsx`, strips timestamp artifacts, auto-detects cumulative vs period returns (diff() if cumulative), clamps to ±20%. Do NOT call directly — use `get_strategy_returns()`. |
| `extend_strategy_returns(base_df)` | 24hr | Appends live monthly returns from yfinance ETF proxies from last xlsx date through current month. 8 weighted baskets (SPY/QQQ/IWM for Equity, etc.). Internal — called by `get_strategy_returns()`. |
| `get_strategy_returns()` | 1hr | **Master function.** Returns clean monthly period returns Aug 2014 → current month. Use everywhere instead of `load_strategy_returns()` directly. |

**Integrations:**
- **Market Pulse sidebar** — added to `display_market_commentary_and_overview()`: VIX/vol regime badge, HY spread, yield curve shape, regime score, next FOMC countdown
- **Forecast Lab** — macro section replaced with `utils.get_macro_data()`; vol regime badge next to scenario selector; T10Y2Y and HY spread available for regime callout
- **Client 360** — enriched risk metrics (Sharpe, Sortino, max DD, beta, alpha, up/down capture) added below AUM/Age/Risk Profile cards via `utils.enrich_client_data()`
- **Quantum Studio** — sleeve correlation multipliers calibrated from `utils.get_derived_signals()["rolling_corr"]`
- **Market Pulse (Performance Snapshot)** — `display_performance_snapshot()` renders a 6-row benchmark table at the top of the Portfolio Pulse page via `display_market_commentary_and_overview()`. Proxy note shown in UI: HY Credit = HYG, Lev Loan = BKLN.

**FOMC dates:** Hardcoded in `get_upcoming_events()` through July 2026. Update when stale (see function comment).

---

## Data files

| File | Purpose |
|------|---------|
| `data/client_data.csv` | 6 HNW clients — client_id, client_name, total_aum, age, risk_profile, primary_advisor, inception_date, tax_bracket, state, filing_status, time_horizon_yrs, next_review_date. **Note: AUM column is `total_aum`, not `aum`.** `load_client_data_csv()` maps it to `aum` in the returned DataFrame. |
| `data/accounts.csv` | 23 accounts across 6 clients (2–5 per client). Columns: account_id, client_id, account_name, account_type, custodian, strategy, aum, inception_date, is_taxable. Strategies match strategy_returns.xlsx column names exactly. Account AUM sums verified to equal client total_aum. |
| `data/client_mapping.py` | Hardcoded client→strategy dict. 6 clients with diverse primary strategies: Warren Miller (GovB), Patricia Huang (L/S Eq HF), David Brown (Equity), Elena Rodriguez (HYB), James Whitfield (PE), Aisha Johnson (LL). Update this when onboarding new clients. |
| `data/rec_log.csv` | Accept/Reject audit log written by `_log_decision()` |
| `data/visitor_log.csv` | Access-request log written by `log_visitor()` |
