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

## Data files

| File | Purpose |
|------|---------|
| `data/rec_log.csv` | Accept/Reject audit log written by `_log_decision()` |
| `data/visitor_log.csv` | Access-request log written by `log_visitor()` |
