# Local Development Setup

This file documents the local Python environment used during development on the ThinkStation workstation.
It is for reference only — the authoritative dependency list for deployment is `requirements.txt`.

## Environment

- Python 3.9 (Anaconda/conda base environment)
- Platform: Linux (WSL2 on Windows)
- CUDA: 12.1 (nvidia-cuda-runtime-cu12)

## Key Packages (production-relevant)

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.36.0 | App framework |
| yfinance | 0.2.61 | Market data |
| pandas | 1.5.3 | Data layer |
| numpy | 1.26.4 | Numerics |
| plotly | 5.22.0 | Charts |
| groq | 0.9.0 | LLM API client |
| openpyxl | 3.1.5 | strategy_returns.xlsx reader |
| fastapi | — | REST sidecar (api.py) |
| scikit-learn | 1.5.1 | ML utilities |
| scipy | 1.13.1 | Statistical functions |

## Notable Local-Only Packages

These were installed locally but are **not** required for the deployed app:

- `torch==2.4.0` / `triton==3.0.0` — GPU/PyTorch (DigitalOcean deploy uses CPU-only)
- `faiss-gpu==1.7.2` — GPU vector search
- `sentence-transformers==3.0.1` / `transformers==4.42.4` — local embedding experiments
- `langchain==0.2.14` / `langchain-community` / `langchain-groq` — LangChain stack (not used in current app)
- `stable_baselines3==2.6.0` / `gymnasium==1.1.1` — RL experiments
- `duckdb==1.0.0` / `pandasai==2.2.11` — analytics experiments
- `jupyterlab` / `ipykernel` — Jupyter for local notebooks
- `boto3==1.39.1` — AWS SDK (not used in current app)

## Full pip freeze

Generated from `pip freeze` on ThinkStation (2024, conda base env):

```
aiohttp==3.9.5
altair==5.3.0
annotated-types==0.7.0
beautifulsoup4
blinker==1.8.2
boto3==1.39.1
cachetools==5.3.3
certifi
click==8.1.7
cloudpickle==3.1.1
curl_cffi==0.11.1
duckdb==1.0.0
faiss-gpu==1.7.2
fastjsonschema
fpdf==1.7.2
frozendict==2.4.4
groq==0.9.0
gymnasium==1.1.1
huggingface-hub==0.23.4
langchain==0.2.14
langchain-community==0.2.12
langchain-core==0.2.33
langchain-groq==0.1.6
langsmith==0.1.90
matplotlib==3.9.1
Nasdaq-Data-Link==1.0.4
numpy==1.26.4
openai==1.35.15
openpyxl==3.1.5
pandas==1.5.3
pandas-datareader==0.10.0
pandasai==2.2.11
pdfkit==1.0.0
pillow==10.4.0
plotly==5.22.0
protobuf==5.27.2
pyarrow==16.1.0
pydantic==1.10.17
pydeck==0.9.1
reportlab==4.2.2
requests
rich==13.7.1
scikit-learn==1.5.1
scipy==1.13.1
seaborn==0.13.2
sentence-transformers==3.0.1
SQLAlchemy==2.0.31
stable_baselines3==2.6.0
streamlit==1.36.0
streamlit-extras==0.4.3
tenacity==8.5.0
tokenizers==0.19.1
torch==2.4.0
transformers==4.42.4
triton==3.0.0
validators==0.33.0
watchdog==4.0.1
websockets==15.0.1
yfinance==0.2.61
```
