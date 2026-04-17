"""
Nigerian Real Estate — Prediction API
======================================
Endpoints
---------
POST /predict/rental-cost    → predict annual + monthly rent (regression)
POST /predict/property-type  → predict property type (classification, 5 classes)
POST /predict/price-tier     → predict affordability tier Low/Mid/High (classification)
POST /predict/all            → all three predictions in one call

GET  /meta/cities            → valid city values
GET  /meta/neighborhoods     → valid neighborhood values
GET  /health                 → liveness check
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # project root
MODEL_DIR = BASE_DIR / "model"

app = FastAPI()

# ── Feature order (must match training) ───────────────────────────────────────
REG_FEATURES = [
    "city_enc", "neighborhood_enc",
    "listing_year", "listing_month", "listing_quarter",
    "amenity_count",
    "amenity_median_value", "amenity_mean_value", "amenity_grade_enc",
    "elec_access_pct_mean", "elec_access_pct_min",
    "elec_population_total", "elec_electrified_total",
]

CLF_FEATURES = [
    "bedrooms", "bathrooms", "size_sqm", "age_years", "price_per_sqm",
    "amenity_count",
    "city_enc", "neighborhood_enc",
    "rental_annual_median", "rental_monthly_median", "rental_count",
    "hotspot_median_value",      "hotspot_mean_value",      "hotspot_grade_enc",
    "amenity_median_value",      "amenity_mean_value",      "amenity_grade_enc",
    "inspection_median_value",   "inspection_mean_value",   "inspection_grade_enc",
    "title_median_value",        "title_mean_value",        "title_grade_enc",
    "service_median_value",      "service_mean_value",      "service_grade_enc",
    "construction_median_value", "construction_mean_value", "construction_grade_enc",
    "land_dispute_median_value", "land_dispute_mean_value", "land_dispute_grade_enc",
    "elec_access_pct_mean", "elec_access_pct_min",
    "elec_population_total", "elec_electrified_total",
    "listing_year", "listing_month", "listing_quarter",
]


# ── Model loader (cached — loaded once at startup) ────────────────────────────
@lru_cache(maxsize=1)
def load_artifacts():
    encoders   = joblib.load(MODEL_DIR / "label_encoders.pkl")
    reg_model  = joblib.load(MODEL_DIR / "gradboost_rental_price_regressor.pkl")
    type_model = joblib.load(MODEL_DIR / "xgboost_property_type_classifier.pkl")
    tier_model = joblib.load(MODEL_DIR / "xgboost_price_tier_classifier.pkl")
    return encoders, reg_model, type_model, tier_model


# ── Helpers ───────────────────────────────────────────────────────────────────
def _encode(encoders, col: str, value: str) -> int:
    le = encoders[col]
    if value not in le.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown {col} '{value}'. Valid values: {le.classes_.tolist()}",
        )
    return int(le.transform([value])[0])


def _grade_enc(encoders, col: str, grade: Optional[str]) -> int:
    if grade is None:
        return 0
    return _encode(encoders, col, grade)


# ── Request / Response models ─────────────────────────────────────────────────
class RentalRequest(BaseModel):
    """Inputs for the rental-cost regression model."""
    city:              str   = Field(..., examples=["Lagos"])
    neighborhood:      str   = Field(..., examples=["Victoria Island"])
    listing_year:      int   = Field(..., examples=[2026])
    listing_month:     int   = Field(..., ge=1, le=12, examples=[4])
    amenity_count:     int   = Field(..., ge=0, examples=[5])
    amenity_median_value:  float = Field(..., examples=[72.0])
    amenity_mean_value:    float = Field(..., examples=[70.5])
    amenity_grade:         Optional[str] = Field(None, examples=["A"])
    elec_access_pct_mean:  float = Field(..., examples=[93.8])
    elec_access_pct_min:   float = Field(..., examples=[84.8])
    elec_population_total: int   = Field(..., examples=[17230908])
    elec_electrified_total: int  = Field(..., examples=[16165395])


class ClassificationRequest(BaseModel):
    """Inputs for property-type and price-tier classification (all features)."""
    city:              str   = Field(..., examples=["Lagos"])
    neighborhood:      str   = Field(..., examples=["Victoria Island"])
    listing_year:      int   = Field(..., examples=[2026])
    listing_month:     int   = Field(..., ge=1, le=12, examples=[4])
    bedrooms:          int   = Field(..., ge=0, examples=[3])
    bathrooms:         int   = Field(..., ge=0, examples=[2])
    size_sqm:          float = Field(..., examples=[120.0])
    age_years:         int   = Field(..., ge=0, examples=[5])
    price_per_sqm:     float = Field(..., examples=[18000.0])
    amenity_count:     int   = Field(..., ge=0, examples=[5])
    rental_annual_median:  float = Field(..., examples=[2200000.0])
    rental_monthly_median: float = Field(..., examples=[183000.0])
    rental_count:      int   = Field(..., ge=0, examples=[1500])
    # City index values
    hotspot_median_value:      float = Field(..., examples=[68.0])
    hotspot_mean_value:        float = Field(..., examples=[67.5])
    hotspot_grade:             Optional[str] = Field(None, examples=["B"])
    amenity_median_value:      float = Field(..., examples=[72.0])
    amenity_mean_value:        float = Field(..., examples=[70.5])
    amenity_grade:             Optional[str] = Field(None, examples=["A"])
    inspection_median_value:   float = Field(..., examples=[65.0])
    inspection_mean_value:     float = Field(..., examples=[64.0])
    inspection_grade:          Optional[str] = Field(None, examples=["B"])
    title_median_value:        float = Field(..., examples=[71.0])
    title_mean_value:          float = Field(..., examples=[70.0])
    title_grade:               Optional[str] = Field(None, examples=["A"])
    service_median_value:      float = Field(..., examples=[60.0])
    service_mean_value:        float = Field(..., examples=[59.5])
    service_grade:             Optional[str] = Field(None, examples=["C"])
    construction_median_value: float = Field(..., examples=[75.0])
    construction_mean_value:   float = Field(..., examples=[74.0])
    construction_grade:        Optional[str] = Field(None, examples=["A"])
    land_dispute_median_value: float = Field(..., examples=[30.0])
    land_dispute_mean_value:   float = Field(..., examples=[31.0])
    land_dispute_grade:        Optional[str] = Field(None, examples=["C"])
    elec_access_pct_mean:      float = Field(..., examples=[93.8])
    elec_access_pct_min:       float = Field(..., examples=[84.8])
    elec_population_total:     int   = Field(..., examples=[17230908])
    elec_electrified_total:    int   = Field(..., examples=[16165395])


class AllRequest(RentalRequest, ClassificationRequest):
    """Combined request for all three predictions at once."""
    pass


class RentalResponse(BaseModel):
    annual_rent_ngn:  float
    monthly_rent_ngn: float


class PropertyTypeResponse(BaseModel):
    property_type: str
    probabilities: dict[str, float]


class PriceTierResponse(BaseModel):
    price_tier:    str  # Low / Mid / High
    probabilities: dict[str, float]


class AllResponse(BaseModel):
    rental:        RentalResponse
    property_type: PropertyTypeResponse
    price_tier:    PriceTierResponse


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nigerian Real Estate Prediction API",
    description="Predict rental cost, property type, and price affordability tier.",
    version="1.0.0",
)

# Allow requests from your Vercel frontend; tighten ALLOW_ORIGINS in production
_raw_origins = os.getenv("ALLOW_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta/cities")
def get_cities():
    encoders, *_ = load_artifacts()
    return {"cities": encoders["city"].classes_.tolist()}


@app.get("/meta/neighborhoods")
def get_neighborhoods():
    encoders, *_ = load_artifacts()
    return {"neighborhoods": encoders["neighborhood"].classes_.tolist()}


# ── Regression ────────────────────────────────────────────────────────────────
@app.post("/predict/rental-cost", response_model=RentalResponse)
def predict_rental_cost(req: RentalRequest):
    encoders, reg_model, *_ = load_artifacts()

    quarter = (req.listing_month - 1) // 3 + 1
    row = pd.DataFrame([{
        "city_enc":             _encode(encoders, "city", req.city),
        "neighborhood_enc":     _encode(encoders, "neighborhood", req.neighborhood),
        "listing_year":         req.listing_year,
        "listing_month":        req.listing_month,
        "listing_quarter":      quarter,
        "amenity_count":        req.amenity_count,
        "amenity_median_value": req.amenity_median_value,
        "amenity_mean_value":   req.amenity_mean_value,
        "amenity_grade_enc":    _grade_enc(encoders, "amenity_grade", req.amenity_grade),
        "elec_access_pct_mean":  req.elec_access_pct_mean,
        "elec_access_pct_min":   req.elec_access_pct_min,
        "elec_population_total": req.elec_population_total,
        "elec_electrified_total": req.elec_electrified_total,
    }])[REG_FEATURES]

    annual = float(reg_model.predict(row)[0])
    return RentalResponse(annual_rent_ngn=round(annual), monthly_rent_ngn=round(annual / 12))


# ── Property Type Classification ──────────────────────────────────────────────
@app.post("/predict/property-type", response_model=PropertyTypeResponse)
def predict_property_type(req: ClassificationRequest):
    encoders, _, type_model, _ = load_artifacts()
    row = _build_clf_row(encoders, req)

    pred_enc = int(type_model.predict(row)[0])
    proba    = type_model.predict_proba(row)[0]
    classes  = encoders["property_type"].classes_.tolist()

    return PropertyTypeResponse(
        property_type=encoders["property_type"].inverse_transform([pred_enc])[0],
        probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
    )


# ── Price Tier Classification ─────────────────────────────────────────────────
@app.post("/predict/price-tier", response_model=PriceTierResponse)
def predict_price_tier(req: ClassificationRequest):
    encoders, _, _, tier_model = load_artifacts()
    row = _build_clf_row(encoders, req)

    pred_enc = int(tier_model.predict(row)[0])
    proba    = tier_model.predict_proba(row)[0]
    classes  = encoders["price_tier"].classes_.tolist()

    return PriceTierResponse(
        price_tier=encoders["price_tier"].inverse_transform([pred_enc])[0],
        probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
    )


# ── All predictions ───────────────────────────────────────────────────────────
@app.post("/predict/all", response_model=AllResponse)
def predict_all(req: AllRequest):
    rental_res = predict_rental_cost(req)
    type_res   = predict_property_type(req)
    tier_res   = predict_price_tier(req)
    return AllResponse(rental=rental_res, property_type=type_res, price_tier=tier_res)


# ── Internal helper ───────────────────────────────────────────────────────────
def _build_clf_row(encoders, req: ClassificationRequest) -> pd.DataFrame:
    quarter = (req.listing_month - 1) // 3 + 1
    return pd.DataFrame([{
        "bedrooms":    req.bedrooms,
        "bathrooms":   req.bathrooms,
        "size_sqm":    req.size_sqm,
        "age_years":   req.age_years,
        "price_per_sqm": req.price_per_sqm,
        "amenity_count": req.amenity_count,
        "city_enc":        _encode(encoders, "city", req.city),
        "neighborhood_enc": _encode(encoders, "neighborhood", req.neighborhood),
        "rental_annual_median":  req.rental_annual_median,
        "rental_monthly_median": req.rental_monthly_median,
        "rental_count":          req.rental_count,
        "hotspot_median_value":      req.hotspot_median_value,
        "hotspot_mean_value":        req.hotspot_mean_value,
        "hotspot_grade_enc":         _grade_enc(encoders, "hotspot_grade", req.hotspot_grade),
        "amenity_median_value":      req.amenity_median_value,
        "amenity_mean_value":        req.amenity_mean_value,
        "amenity_grade_enc":         _grade_enc(encoders, "amenity_grade", req.amenity_grade),
        "inspection_median_value":   req.inspection_median_value,
        "inspection_mean_value":     req.inspection_mean_value,
        "inspection_grade_enc":      _grade_enc(encoders, "inspection_grade", req.inspection_grade),
        "title_median_value":        req.title_median_value,
        "title_mean_value":          req.title_mean_value,
        "title_grade_enc":           _grade_enc(encoders, "title_grade", req.title_grade),
        "service_median_value":      req.service_median_value,
        "service_mean_value":        req.service_mean_value,
        "service_grade_enc":         _grade_enc(encoders, "service_grade", req.service_grade),
        "construction_median_value": req.construction_median_value,
        "construction_mean_value":   req.construction_mean_value,
        "construction_grade_enc":    _grade_enc(encoders, "construction_grade", req.construction_grade),
        "land_dispute_median_value": req.land_dispute_median_value,
        "land_dispute_mean_value":   req.land_dispute_mean_value,
        "land_dispute_grade_enc":    _grade_enc(encoders, "land_dispute_grade", req.land_dispute_grade),
        "elec_access_pct_mean":  req.elec_access_pct_mean,
        "elec_access_pct_min":   req.elec_access_pct_min,
        "elec_population_total": req.elec_population_total,
        "elec_electrified_total": req.elec_electrified_total,
        "listing_year":    req.listing_year,
        "listing_month":   req.listing_month,
        "listing_quarter": quarter,
    }])[CLF_FEATURES]


