# NexusAI

NexusAI is a Streamlit analytics demonstration with interactive dashboards for healthcare, fintech, e-commerce, energy, and automated machine learning.

## Requirements

- Python 3.10 or newer
- pip

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The app creates its sample CSV datasets and X-ray images under `assets/` when needed. Those files are generated and are excluded from Git.

## Project layout

```text
app.py                  Streamlit entry point and navigation
core/                   Shared styling and sample-data generators
modules/                Healthcare, fintech, e-commerce, energy, and AutoML pages
requirements.txt        Python dependencies
```

## Demo limitations

This project is intended for demonstration and learning. Some outputs are based on synthetic data, simple heuristics, or simulated model results; they are not validated for clinical, financial, or operational decision-making. AutoDataLab trains models on the selected dataset, but its results should not be treated as production validation.