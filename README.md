# 🏠 House Price Prediction — End-to-End ML Web App

An end-to-end machine-learning product: a messy, real-world dataset of Indian property
listings is cleaned and modeled in a Jupyter notebook, served through a FastAPI backend,
and consumed by a React + TypeScript frontend where a user enters property details and
gets an instant price estimate.

## Overview

| Layer | Tech |
|---|---|
| Modeling | pandas, scikit-learn (Pipeline + ColumnTransformer), Jupyter |
| Backend | FastAPI, pydantic-settings, joblib |
| Frontend | React 18, TypeScript, Vite, React Router |
| Packaging | Docker (backend) |

## Architecture

```
                    ┌──────────────────────┐
 Kaggle dataset ───▶│  notebooks/           │──▶ house_price.pkl
 (house_prices.csv) │  house_price_model    │──▶ locations.json
                    │  .ipynb (clean/train) │──▶ metrics.json
                    └──────────────────────┘
                               │  (copy artifacts)
                               ▼
   ┌─────────────────┐   HTTP JSON   ┌──────────────────────┐
   │  React frontend  │ ────────────▶│   FastAPI backend     │
   │  (Vite, :5173)   │◀──────────── │   (uvicorn, :8000)    │
   └─────────────────┘  predicted    │  loads Pipeline once  │
                          price      │  at startup           │
                                     └──────────────────────┘
```

The exported artifact is a **single scikit-learn `Pipeline`** containing both
preprocessing (imputation, scaling, one-hot encoding) and the trained regressor, so the
backend never re-implements feature engineering — it just calls `.predict()`.

## Project structure

```
house-price-project/
├── notebooks/
│   ├── house_price_model.ipynb   # cleaning, EDA, training, export
│   └── data/                     # put house_prices.csv here (gitignored)
├── backend/
│   ├── app/
│   │   ├── main.py                       # FastAPI app, CORS, lifespan model loading
│   │   ├── api/routes/prediction.py      # GET /health, POST /predict
│   │   ├── core/config.py                # settings from .env (pydantic-settings)
│   │   ├── schemas/prediction.py         # PredictionRequest / PredictionResponse
│   │   ├── services/
│   │   │   ├── preprocessing.py          # request -> one-row DataFrame
│   │   │   └── inference.py              # loads .pkl once, runs predict
│   │   └── utils/logging_config.py
│   ├── models/                   # house_price.pkl + locations.json go here
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/predictionClient.ts
│   │   ├── components/PredictionForm.tsx
│   │   ├── pages/{HomePage,ResultPage,NotFoundPage}.tsx
│   │   ├── types/prediction.ts
│   │   └── App.tsx
│   ├── public/locations.json     # replace with the notebook's export
│   └── .env.example
├── .gitignore
└── README.md
```

## Dataset

**[House Price](https://www.kaggle.com/datasets/juhibhojani/house-price)** by Juhi
Bhojani — ~187k real property listings from India (`house_prices.csv`).

Download it either:

**Manually** — click *Download* on the dataset page, unzip, and place the CSV at
`notebooks/data/house_prices.csv`.

**Or via the Kaggle CLI:**
```bash
pip install kaggle
# Get an API token: Kaggle -> Settings -> API -> "Create New Token"
# Place kaggle.json in ~/.kaggle/ (or C:\Users\<you>\.kaggle\ on Windows)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

## 1. Run the notebook

```bash
cd house-price-project
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib

cd notebooks
jupyter notebook house_price_model.ipynb
# Kernel -> Restart & Run All
```

This produces `house_price.pkl`, `locations.json` and `metrics.json` inside `notebooks/`.
Copy them into place:

```bash
cp house_price.pkl locations.json ../backend/models/
cp locations.json ../frontend/public/locations.json
```

> ⚠️ **Version pinning:** note the scikit-learn version printed at the end of the
> notebook (`sklearn.__version__`) and make sure `backend/requirements.txt` pins the
> **same** version — a pickle only reliably loads with the scikit-learn version it was
> saved with.

## 2. Backend setup (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env             # adjust MODEL_PATH / CORS_ORIGINS if needed

uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Run the tests:
```bash
pytest
```

### Environment variables (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/house_price.pkl` | Path to the exported Pipeline |
| `LOCATIONS_PATH` | `models/locations.json` | Path to the allowed-locations list |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `APP_NAME` | `House Price Prediction API` | Shown in `/docs` |
| `APP_VERSION` | `1.0.0` | Shown in `/docs` |

### API reference

**`GET /health`**
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**`POST /predict`**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "thane",
    "area_sqft": 650,
    "floor_num": 3,
    "bathroom": 2,
    "balcony": 1,
    "car_parking": 1,
    "status": "Ready to Move",
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "ownership": "Freehold"
  }'
# {"predicted_price": 4123456.78}
```

### Run with Docker

```bash
cd backend
docker build -t house-price-backend .
docker run -p 8000:8000 house-price-backend
```

## 3. Frontend setup (React + TypeScript + Vite)

```bash
cd frontend
npm install
cp .env.example .env             # points VITE_API_BASE_URL at the backend

npm run dev
# open http://localhost:5173
```

### Environment variables (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Base URL of the FastAPI backend |

Build for production with `npm run build` (output in `frontend/dist/`).

## Verify the full flow

1. Backend running on `:8000` (`uvicorn app.main:app --reload`).
2. Frontend running on `:5173` (`npm run dev`).
3. Open the app, fill in the form, submit — you should see a real predicted price on
   the result page.

## Model metrics

*(filled in from `notebooks/metrics.json` after running the notebook — sample results
from a reference run below)*

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~3.99M | ~6.14M | ~0.71 |
| **Random Forest (chosen)** | **~1.06M** | **~3.30M** | **~0.92** |

The Random Forest Regressor was selected as the final model: it captures non-linear
relationships between location, area, and property attributes far better than the
linear baseline, at the cost of a heavier (but still fast) `.pkl` file.

## Screenshots

*(add screenshots of the running app here before submitting)*

- `docs/screenshot-home.png` — the estimator form
- `docs/screenshot-result.png` — the price result page

## Publishing to GitHub

```bash
git init
git add .
git commit -m "House price prediction: notebook, FastAPI backend, React frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-app.git
git push -u origin main
```

Before pushing, double-check:
- [ ] `.gitignore` is in place (excludes `.venv/`, `node_modules/`, `.env`, the raw CSV)
- [ ] `backend/models/house_price.pkl` is committed only if it's under 50 MB
- [ ] The repository is **public** and accessible

## License

For educational use as part of the House Price Prediction student project.
