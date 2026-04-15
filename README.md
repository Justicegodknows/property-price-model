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
