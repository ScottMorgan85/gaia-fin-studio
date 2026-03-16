# GAIA: Generative AI Investment Analytics

**Live:** https://gaia-fin-studio-umumr.ondigitalocean.app/

> **Production-grade AI analytics platform built to replace $50k/yr vendor tools with a $5/month open stack.**

GAIA is a full-stack wealth management intelligence platform demonstrating what a modern data/AI engineering practice looks like in financial services — real-time market data, probabilistic forecasting, LLM-driven research, and quantum-inspired optimization, all deployed on commodity cloud infrastructure.

Built as a direct alternative to **Bloomberg**, **FactSet**, **PitchBook**, **Morningstar Direct**, **Capital IQ**, and **YCharts** — platforms that charge enterprise licensing fees for capabilities that a well-engineered open stack can match or exceed.

---

## Why This Exists

Legacy financial platforms share the same failure modes:

- Rigid AI roadmaps — you wait 18 months for features that can be shipped in a weekend
- Black-box models you can't audit, prompt, or extend
- Per-seat pricing that punishes scale
- No path to custom integrations with your own data

GAIA is the counter-thesis: open LLMs, a Python-native analytics stack, live market + macro data, and full infrastructure ownership. The entire platform runs on DigitalOcean App Platform for ~$5/month.

---

## Platform Capabilities

### Client 360
Full relationship intelligence per client — AUM, risk profile, interaction history, transaction ledger, and enriched risk metrics (Sharpe, Sortino, Calmar ratio, beta, alpha, up/down capture vs. SPY) computed from live yfinance data.

### Market Pulse
Real-time market intelligence dashboard with:
- **Performance Snapshot** — DTD/MTD/QTD/YTD + 1/3/5yr annualized returns for 6 key benchmarks (S&P 500, MSCI EAFE, US Agg Bond, DJ Commodity, HY Credit, Lev Loan)
- **Macro signals** — VIX/vol regime badge, HY OAS spread (bps), yield curve shape, regime score
- **LLM commentary** — AI-generated day-to-day market narrative via Groq-hosted Llama 3.3 70B
- **FOMC countdown** — days to next Fed meeting

### Forecast Lab
Probabilistic simulation engine for client strategy analysis:
- Block bootstrap Monte Carlo (1,000 paths, 60-month horizon, 6-month blocks) — preserves autocorrelation
- Fan chart with 10/25/50/75/90th percentile bands, denominated in dollars
- Scenario table: base / bull / bear / custom terminal values from $10,000
- Regime overlay: 2×2 GDP growth × CPI momentum matrix → Goldilocks / Reflation / Stagflation / Deflation
- Live macro inputs from FRED (14 series: CPI, GDP, unemployment, yield curve, credit spreads)
- AI trade ideas via `st.button` — macro-aware prompt, no auto-run

### Scenario Allocator
Strategy-level allocation recommendations with LLM-generated trade ideas contextualized to current macro regime.

### Quantum Studio
Quantum-inspired portfolio optimization PoC — simulated annealing on a QUBO-style objective across six sleeve allocations anchored to live strategy returns. Sleeve correlation multipliers calibrated from 24-month rolling correlations via `get_derived_signals()`.

### Recommendation Engine
Streamed AI recommendations per client strategy with Accept/Reject audit logging to `data/rec_log.csv`.

---

## Data Layer

All functions live in `utils.py`, cache with `@st.cache_data`, return empty df/dict on failure, and never crash the app.

| Function | TTL | Source | Description |
|---|---|---|---|
| `get_market_data()` | 1hr | yfinance | Monthly prices + returns for 19 tickers, daily VIX, factor returns |
| `get_macro_data()` | 24hr | FRED REST API | 14 macro series → monthly DataFrame (CPI, GDP, T10Y2Y, HY spreads, etc.) |
| `get_derived_signals()` | 1hr | Computed | 12-1 momentum, regime score (−2→+2), vol regime, yield curve shape, 24-month rolling corr matrix |
| `enrich_client_data()` | 1hr | yfinance | Sharpe, Sortino, Calmar, beta, alpha, up/down capture vs. SPY per strategy |
| `get_upcoming_events()` | 12hr | yfinance + hardcoded | Earnings calendar for 10 tickers + FOMC dates through Jul 2026 |
| `get_benchmark_returns()` | 1hr | yfinance | DTD/MTD/QTD/YTD + 1/3/5yr annualized for 6 benchmarks |

---

## Architecture

```mermaid
flowchart TD
  subgraph Live Data
    A1[yfinance — prices, VIX, earnings]
    A2[FRED REST API — 14 macro series]
  end

  subgraph utils.py — Data Layer
    B1[get_market_data]
    B2[get_macro_data]
    B3[get_derived_signals]
    B4[enrich_client_data]
    B5[get_benchmark_returns]
    B6[get_upcoming_events]
  end

  subgraph Static Data
    C1[strategy_returns.xlsx]
    C2[client_data.csv]
    C3[client_transactions.csv]
  end

  A1 --> B1 & B4 & B5 & B6
  A2 --> B2 & B3

  subgraph gaia_pages.py — Pages
    P1[Market Pulse + Performance Snapshot]
    P2[Client 360]
    P3[Forecast Lab]
    P4[Scenario Allocator]
    P5[Quantum Studio]
    P6[Recommendation Engine]
  end

  B1 & B2 & B3 --> P1 & P3
  B4 --> P2
  B5 --> P1
  B6 --> P1 & P3
  C1 --> P3 & P4 & P5
  C2 --> P2
  C3 --> P6

  subgraph LLM Layer
    G[Groq API — llama-3.3-70b-versatile]
    GF[Fallback — llama-4-scout-17b-16e]
  end

  P1 & P3 & P4 & P6 --> G
  G -->|on error| GF

  P1 & P2 & P3 & P4 & P5 & P6 --> UI[GAIA Streamlit UI]
```

---

## Project Structure

```
gaia-fin-studio/
├── app.py                  ← Entry point; set_page_config at line 7
├── gaia_pages.py           ← All page functions and LLM orchestration
├── commentary.py           ← DTD commentary generation
├── utils.py                ← Full data layer + helpers
├── landing.py              ← Access gate / visitor log
├── assets/
│   ├── Collector.py        ← yfinance wrappers
│   ├── Portfolio.py        ← Portfolio object
│   └── Stock.py            ← Stock object
├── data/
│   ├── client_data.csv
│   ├── client_transactions.csv
│   ├── strategy_returns.xlsx
│   ├── rec_log.csv         ← Accept/Reject audit log
│   └── visitor_log.csv     ← Access-request log
├── .streamlit/
│   └── config.toml         ← Theme config (secrets.toml is gitignored)
├── requirements.txt
└── Dockerfile
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM inference | Groq API (llama-3.3-70b-versatile / llama-4-scout fallback) |
| Market data | yfinance |
| Macro data | FRED REST API |
| Visualization | Plotly |
| Optimization | NumPy simulated annealing (QUBO-style) |
| Hosting | DigitalOcean App Platform |
| Container | Docker |

---

## Data Asset Catalog

### `client_data.csv`
Client demographics for Client 360.
| Column | Description | Example |
|---|---|---|
| `client_name` | Unique client identifier | "Acme Family" |
| `aum` | Assets under management | 1000000 |
| `age` | Client age | 55 |
| `risk_profile` | Risk tolerance label | "Moderate" |

### `client_interactions.csv`
CRM-style interaction history.
| Column | Description | Example |
|---|---|---|
| `client_name` | Must match client_data.csv | "Acme Family" |
| `date` | Interaction date | "2025-06-15" |
| `interaction_type` | Call / Meeting / Email | "Quarterly Review" |
| `notes` | Summary and next steps | "Discussed rebalance" |

### `client_transactions.csv`
Recent buys/sells used in commentary and recommendation engine.
| Column | Description | Example |
|---|---|---|
| `Name` | Security name | "Apple Inc." |
| `Transaction Type` | Buy or Sell | "Buy" |
| `Total Value ($)` | Trade notional | 25000 |
| `Selected_Strategy` | Strategy label | "Equity" |
| `Commentary` | Analyst note | "Q2 positioning" |

### `strategy_returns.xlsx`
Monthly strategy performance time series.
| Column | Description | Example |
|---|---|---|
| `as_of_date` | Month-end date | "2025-06-30" |
| `<strategy>` | Return series per column | 0.012 |

### `rec_log.csv`
Audit log of user decisions on recommendations.
| Column | Description |
|---|---|
| `timestamp` | ISO datetime |
| `client` | Client name |
| `strategy` | Strategy name |
| `decision` | Accept or Reject |
| `ml_score` | Model confidence score |

### `visitor_log.csv`
Access-request and approval trail.
| Column | Description |
|---|---|
| `timestamp` | ISO datetime |
| `name` | Visitor display name |
| `email` | Visitor email |

---

## Setup

### Prerequisites
- Python 3.9+
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- FRED API key (free registration at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))

### Local development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Environment variables
```bash
export GROQ_API_KEY=your_groq_key
export FRED_API_KEY=your_fred_key   # optional — falls back to public key
export GAIA_GATE_ON=false           # set true to enable access gate
```

Or place in `.streamlit/secrets.toml` locally (never commit — gitignored).

### Docker
```bash
docker build -t gaia-dashboard .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  gaia-dashboard
```

---

## Deploy

### DigitalOcean App Platform (~$5/mo)
1. Connect GitHub repo
2. Service type: **Python**
3. Build: `pip install -r requirements.txt`
4. Run: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add env vars: `GROQ_API_KEY`, `FRED_API_KEY`, `GAIA_GATE_ON`

App Platform handles HTTPS, auto-deploy on push to `main`, and horizontal scaling.

### DigitalOcean Droplet (~$5/mo)
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv nginx
git clone https://github.com/YOUR_USERNAME/gaia-fin-studio.git && cd gaia-fin-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Systemd service for process management + Nginx reverse proxy for TLS via `certbot`.

---

## Security Notes
- All secrets via `os.environ.get()` — `st.secrets` is not used anywhere (incompatible with DigitalOcean App Platform)
- `secrets.toml` is gitignored
- Recommendation decisions logged with timestamp to `rec_log.csv`
- Access requests logged to `visitor_log.csv`

---

## License
MIT — see `LICENSE`.

---

*Built by Scott Morgan — demonstrating that production-grade financial AI doesn't require a seven-figure vendor contract.*
