# Nigerian Real Estate — ML Prediction API

A machine learning pipeline and REST API for predicting rental costs, property types, and affordability tiers across Nigerian cities.

---

## Overview

This project trains three ML models on merged Nigerian real estate datasets and serves predictions through a FastAPI application.

| Model | Type | Output |
|---|---|---|
| `gradboost_rental_price_regressor` | Regression (Gradient Boosting) | Annual & monthly rent (NGN) |
| `xgboost_property_type_classifier` | Classification (XGBoost) | Property type (5 classes) |
| `xgboost_price_tier_classifier` | Classification (XGBoost) | Affordability tier (Low / Mid / High) |

---

## Project Structure

```
property-price-model/
├── api/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # API dependencies
├── model/
│   ├── label_encoders.pkl                   # Fitted LabelEncoders
│   ├── gradboost_rental_price_regressor.pkl
│   ├── xgboost_property_type_classifier.pkl
│   └── xgboost_price_tier_classifier.pkl
├── property-price-model/
│   └── data/
│       ├── property_listings.csv
│       ├── amenities_data.csv
│       ├── location_data.csv
│       └── market_data.csv
├── train_model.ipynb        # Full ML pipeline notebook
└── README.md
```

---

## Datasets

The pipeline merges eight data sources:

| Dataset | Description |
|---|---|
| `property_listings` | Core property data (bedrooms, size, price, etc.) |
| `location_hotspots` | City location-quality score |
| `neighborhood_amenities` | Neighbourhood amenity index |
| `property_inspections` | Inspection grade index |
| `title_registrations` | Title/legal value index |
| `service_charges` | Service-charge level index |
| `construction_costs` | Construction cost index |
| `land_disputes` | Land dispute risk index |
| `electricity_access` | State electricity access percentage |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd property-price-model
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r api/requirements.txt
```

### 3. Train the models (optional — pre-trained models are included)

Open and run all cells in `train_model.ipynb`. This generates the `.pkl` files in the `model/` directory.

### 4. Start the API server

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

---

## API Endpoints

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |

### Metadata

| Method | Endpoint | Description |
|---|---|---|
| GET | `/meta/cities` | List of valid city values |
| GET | `/meta/neighborhoods` | List of valid neighborhood values |

### Predictions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict/rental-cost` | Predict annual + monthly rent |
| POST | `/predict/property-type` | Predict property type (5 classes) |
| POST | `/predict/price-tier` | Predict affordability tier (Low/Mid/High) |
| POST | `/predict/all` | All three predictions in one call |

---

## Example Request

**POST** `/predict/rental-cost`

```json
{
  "city": "Lagos",
  "neighborhood": "Victoria Island",
  "listing_year": 2026,
  "listing_month": 4,
  "amenity_count": 5,
  "amenity_median_value": 72.0,
  "amenity_mean_value": 70.5,
  "amenity_grade": "A",
  "elec_access_pct_mean": 93.8,
  "elec_access_pct_min": 84.8,
  "elec_population_total": 17230908,
  "elec_electrified_total": 16165395
}
```

**Response**

```json
{
  "annual_rent_ngn": 2640000.0,
  "monthly_rent_ngn": 220000.0
}
```

---

## Property Types

- `detached`
- `flat`
- `semi_detached`
- `terrace`
- `bungalow`

## Price Tiers

- `Low` — affordable / budget segment
- `Mid` — mid-market segment
- `High` — premium / luxury segment

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP
- **API**: FastAPI, Uvicorn, Pydantic v2
- **Data**: Pandas, NumPy
- **Serialization**: Joblib

---

## Deployment (DigitalOcean App Platform)

### 1. Connect your GitHub repo to DigitalOcean

1. Push this repository to GitHub (if not already done)
2. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps) → **Create App**
3. Select your GitHub repo and the `main` branch
4. DO will auto-detect the `Dockerfile` at the project root

### 2. Configure the app

The `.do/app.yaml` file pre-configures the service. Review and adjust `instance_size_slug` if your models need more RAM.

Set the following environment variable in the DO App Platform dashboard (Settings → Environment Variables):

| Variable | Value | Notes |
|---|---|---|
| `ALLOW_ORIGINS` | `https://your-vercel-app.vercel.app` | Your Vercel frontend URL |
| `PORT` | `8080` | Already set in app.yaml |

### 3. Deploy

Push to `main` — App Platform deploys automatically. Once live, your API URL will be:
```
https://<app-name>.ondigitalocean.app
```

Use this as `ML_API_URL` in your Vercel project's environment variables.

---

## Supabase Setup

Run the SQL in `supabase/schema.sql` in your Supabase project's **SQL Editor** to create the `predictions` table with indexes and Row Level Security policies.

---

## Next.js Integration (Vercel)

Add the following environment variables to your Vercel project:

| Variable | Value |
|---|---|
| `ML_API_URL` | `https://<app-name>.ondigitalocean.app` |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Your Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service role key (server-only) |

In your Next.js Server Action, call `POST ${ML_API_URL}/predict/all` with the user's inputs, then insert the result into the `predictions` Supabase table.
