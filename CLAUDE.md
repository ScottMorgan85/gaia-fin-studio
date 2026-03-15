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
Defined at the top of `gaia_pages.py` (after constants, before page functions).

Tries these sources in order, returns the first non-empty string:
1. `os.environ.get("GROQ_API_KEY", "")`
2. `st.secrets.get("GROQ_API_KEY", "")` — flat key in `secrets.toml`
3. `st.secrets.get("env", {}).get("GROQ_API_KEY", "")` — nested under `[env]`

Both `st.secrets` reads are wrapped in `try/except` so the app doesn't crash on
DigitalOcean App Platform where `secrets.toml` is absent.

**Rule:** Every place in `gaia_pages.py` that needs the Groq API key must call
`get_groq_key()` — never read `os.environ` or `st.secrets` directly for this key.

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

## Data files

| File | Purpose |
|------|---------|
| `data/rec_log.csv` | Accept/Reject audit log written by `_log_decision()` |
| `data/visitor_log.csv` | Access-request log written by `log_visitor()` |
