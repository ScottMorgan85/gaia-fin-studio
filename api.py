"""
GAIA Financial Intelligence API
================================
Exposes core market signals, benchmark returns, client risk metrics, and macro
data as a REST service — decoupling the analytics layer from the Streamlit UI.

Run:
    uvicorn api:app --port 8502 --reload

Docs (auto-generated):
    http://localhost:8502/docs       ← Swagger UI
    http://localhost:8502/redoc      ← ReDoc

Auth:
    Set GAIA_API_KEY env var to enable key enforcement.
    Pass the key as:  X-API-Key: <your-key>
    If GAIA_API_KEY is unset, auth is disabled (open access).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GAIA Financial Intelligence API",
    description=(
        "Live market signals, benchmark returns, client risk metrics, and "
        "macro data from the GAIA analytics platform."
    ),
    version="1.0.0",
    contact={"name": "Scott Morgan"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Auth ─────────────────────────────────────────────────────────────────────

_API_KEY      = os.environ.get("GAIA_API_KEY", "")
_api_key_hdr  = APIKeyHeader(name="X-API-Key", auto_error=False)


def _auth(key: str = Security(_api_key_hdr)) -> bool:
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return True


# ── Data layer import ─────────────────────────────────────────────────────────
# utils.py uses @st.cache_data; outside Streamlit it falls back to in-memory
# cache automatically (MemoryCacheStorageManager).  GROQ_API_KEY must be set.

try:
    import utils as _u
except Exception as _import_err:
    _u = None
    print(f"[GAIA API] utils import failed: {_import_err}", flush=True)


def _require_utils():
    if _u is None:
        raise HTTPException(
            status_code=503,
            detail="Data layer unavailable — check server logs for import errors.",
        )
    return _u


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(v):
    """Convert a value to JSON-safe float or None."""
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, 6)
    except Exception:
        return None


def _series_tail(s: pd.Series, n: int = 6) -> dict:
    return {str(idx.date() if hasattr(idx, "date") else idx): _safe(val)
            for idx, val in s.tail(n).items()}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Liveness check",
    tags=["Meta"],
)
def health():
    """Returns service status and version. No auth required."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "service": "GAIA Financial Intelligence API",
    }


@app.get(
    "/v1/signals",
    summary="Live derived market signals",
    tags=["Market Data"],
    dependencies=[Depends(_auth)],
)
def get_signals():
    """
    Returns the latest derived signals computed by GAIA:
    - **vol_regime**: current volatility regime (Low / Elevated / High)
    - **regime_score**: macro regime score −2 (bearish) → +2 (bullish)
    - **hy_spread**: HY OAS spread in basis points
    - **yield_curve**: yield curve shape label
    - **momentum**: 12-1 month momentum scores for tracked tickers
    """
    u = _require_utils()
    try:
        signals = u.get_derived_signals()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Signal computation failed: {e}")

    result: dict = {}
    for k, v in signals.items():
        if k in ("rolling_corr", "corr_matrix"):
            continue                          # skip large matrices
        if isinstance(v, (int, float)):
            result[k] = _safe(v)
        elif isinstance(v, str):
            result[k] = v
        elif isinstance(v, pd.Series):
            result[k] = _series_tail(v)
        elif isinstance(v, pd.DataFrame):
            pass                              # omit DataFrames from this endpoint

    return {"data": result, "source": "GAIA derived signals (utils.get_derived_signals)"}


@app.get(
    "/v1/benchmarks",
    summary="Benchmark performance table",
    tags=["Market Data"],
    dependencies=[Depends(_auth)],
)
def get_benchmarks():
    """
    Total-return performance for 6 key market benchmarks:
    S&P 500, MSCI EAFE, US Agg Bond, DJ Commodity, HY Credit (HYG), Lev Loan (BKLN).

    Columns: DTD, MTD, QTD, YTD (cumulative); 1yr Ann, 3yr Ann, 5yr Ann (annualized).
    """
    u = _require_utils()
    try:
        df = u.get_benchmark_returns()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Benchmark fetch failed: {e}")

    if df.empty:
        raise HTTPException(status_code=503, detail="Benchmark data unavailable — yfinance may be rate-limited.")

    return {"data": df.to_dict(orient="records"), "count": len(df)}


@app.get(
    "/v1/macro",
    summary="Latest macro snapshot",
    tags=["Macro"],
    dependencies=[Depends(_auth)],
)
def get_macro():
    """
    Latest values for 14 FRED macro series: CPI, GDP, unemployment rate,
    10-2yr yield spread (T10Y2Y), HY OAS, fed funds rate, and more.
    Requires FRED_API_KEY to be set; returns 503 if unavailable.
    """
    u = _require_utils()
    try:
        df = u.get_macro_data()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"FRED fetch failed: {e}")

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Macro data unavailable — check FRED_API_KEY environment variable.",
        )

    latest = df.iloc[-1]
    clean  = {k: (_safe(v) if not isinstance(v, str) else v) for k, v in latest.items()}
    return {"data": clean, "as_of": str(df.index[-1].date())}


@app.get(
    "/v1/clients",
    summary="List clients",
    tags=["Clients"],
    dependencies=[Depends(_auth)],
)
def list_clients():
    """Returns all client names available in the platform."""
    try:
        from data.client_mapping import get_client_names
        return {"clients": get_client_names(), "count": len(get_client_names())}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(
    "/v1/clients/{client_name}/risk",
    summary="Client risk metrics",
    tags=["Clients"],
    dependencies=[Depends(_auth)],
)
def get_client_risk(client_name: str):
    """
    Enriched risk metrics for the strategy associated with a client:
    Sharpe, Sortino, Calmar ratio, max drawdown, beta, alpha,
    up-capture and down-capture vs. SPY.  Lookback: 1/3/5yr.
    """
    u = _require_utils()
    try:
        from data.client_mapping import client_strategy_risk_mapping
        strategy = client_strategy_risk_mapping.get(client_name)
        if isinstance(strategy, dict):
            strategy = strategy.get("strategy_name") or strategy.get("strategy")
        if not strategy:
            raise HTTPException(
                status_code=404, detail=f"Client '{client_name}' not found."
            )

        df = u.enrich_client_data()
        if df.empty:
            raise HTTPException(status_code=503, detail="Risk metrics unavailable.")
        if strategy not in df.index:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy}' not in risk index. Available: {list(df.index)}",
            )

        row   = df.loc[strategy].to_dict()
        clean = {k: _safe(v) for k, v in row.items()}
        return {"client": client_name, "strategy": strategy, "metrics": clean}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(
    "/v1/clients/{client_name}/factor-exposures",
    summary="Client factor exposures (FF5)",
    tags=["Clients"],
    dependencies=[Depends(_auth)],
)
def get_factor_exposures(client_name: str):
    """
    Fama-French 5-factor exposures for the client's strategy:
    Market (Mkt-RF), Size (SMB), Value (HML), Profitability (RMW), Investment (CMA).
    Includes loadings, t-statistics, R², adj-R², annualized alpha.
    """
    u = _require_utils()
    try:
        from data.client_mapping import client_strategy_risk_mapping
        strategy = client_strategy_risk_mapping.get(client_name)
        if isinstance(strategy, dict):
            strategy = strategy.get("strategy_name") or strategy.get("strategy")
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Client '{client_name}' not found.")

        result = u.get_factor_exposures(strategy)
        if not result:
            raise HTTPException(
                status_code=503,
                detail="Factor data unavailable — Ken French Data Library may be unreachable.",
            )

        return {
            "client":           client_name,
            "strategy":         strategy,
            "loadings":         result["loadings"],
            "t_stats":          result["t_stats"],
            "r2":               result["r2"],
            "adj_r2":           result["adj_r2"],
            "alpha_annualized": result["alpha_annualized"],
            "alpha_t":          result["alpha_t"],
            "n_months":         result["n_months"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8502, reload=True)
