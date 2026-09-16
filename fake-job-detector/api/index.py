"""
Fake job posting detector - inference API.

Rebuilds exactly the same feature pipeline used in fake_job_postings.ipynb:
    [ TF-IDF(full_text) | OneHot(categorical) | Scaled(numeric) ]
in that column order, then calls model.predict_proba.
"""

import json
import os

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.sparse import csr_matrix, hstack

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Column order must match the notebook exactly.
NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "has_salary_range",
    "missing_company_profile",
    "missing_description",
    "missing_requirements",
    "missing_benefits",
    "text_length",
]
CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "country",
]
TEXT_COLUMNS = ["company_profile", "description", "requirements", "benefits"]

# Probability at or above this is reported as fraudulent.
THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))


def _load(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


model = _load("model.pkl")
tfidf = _load("tfidf_vectorizer.pkl")
encoder = _load("onehot_encoder.pkl")
scaler = _load("standard_scaler.pkl")

# Optional metadata written by the export cell. Falls back to the encoder's
# own learned categories, which is where the country list ends up anyway.
try:
    with open(os.path.join(MODELS_DIR, "metadata.json"), encoding="utf-8") as f:
        METADATA = json.load(f)
except (OSError, ValueError):
    METADATA = {}

_country_idx = CATEGORICAL_COLUMNS.index("country")
KNOWN_COUNTRIES = set(METADATA.get("top_countries", [])) | set(
    str(c) for c in encoder.categories_[_country_idx]
)

app = FastAPI(title="Fake job posting detector")


class Posting(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    location: str = ""
    salary_range: str = ""
    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


def get_country(location: str) -> str:
    """Same rule as the notebook: the part before the first comma."""
    if not location or not location.strip():
        return "unknown"
    head = str(location).split(",")[0].strip()
    return head if head else "unknown"


def build_features(p: Posting):
    # A field left blank in the form is treated as a missing field, which is
    # itself one of the strongest signals in the training data.
    raw = {c: (getattr(p, c) or "").strip() for c in TEXT_COLUMNS}
    missing = {f"missing_{c}": int(raw[c] == "") for c in TEXT_COLUMNS}

    title = (p.title or "").strip()
    full_text = " ".join(
        [
            title,
            raw["company_profile"],
            raw["description"],
            raw["requirements"],
            raw["benefits"],
        ]
    )

    country = get_country(p.location)
    if country not in KNOWN_COUNTRIES:
        country = "other"

    cat_values = [
        (p.employment_type or "").strip() or "unknown",
        (p.required_experience or "").strip() or "unknown",
        (p.required_education or "").strip() or "unknown",
        country,
    ]

    numeric = {
        "telecommuting": int(bool(p.telecommuting)),
        "has_company_logo": int(bool(p.has_company_logo)),
        "has_questions": int(bool(p.has_questions)),
        "has_salary_range": int(bool((p.salary_range or "").strip())),
        "text_length": len(full_text),
        **missing,
    }

    x_text = tfidf.transform([full_text])
    x_cat = encoder.transform(np.array([cat_values], dtype=object))
    x_num = scaler.transform(
        np.array([[numeric[c] for c in NUMERIC_COLUMNS]], dtype=float)
    )

    features = hstack([x_text, x_cat, csr_matrix(x_num)]).tocsr()
    return features, numeric, country, full_text


def explain(numeric, country, full_text):
    """Plain-language notes drawn from the features, not from the model."""
    notes = []
    if numeric["missing_company_profile"]:
        notes.append("مفيش نبذة عن الشركة — ده أكتر حاجة بتتكرر في الإعلانات المزيفة")
    if not numeric["has_company_logo"]:
        notes.append("الإعلان من غير لوجو للشركة")
    if numeric["missing_requirements"]:
        notes.append("مفيش متطلبات وظيفة محددة")
    if numeric["missing_benefits"]:
        notes.append("مفيش تفاصيل عن المزايا")
    if len(full_text) < 400:
        notes.append("نص الإعلان قصير جدًا مقارنة بالإعلانات الحقيقية")
    if country == "unknown":
        notes.append("مكان الوظيفة مش محدد")
    if not numeric["has_questions"]:
        notes.append("مفيش أسئلة فرز للمتقدمين")
    return notes


@app.post("/api/predict")
def predict(posting: Posting):
    if not (posting.description or "").strip() and not (posting.title or "").strip():
        return JSONResponse(
            {"error": "اكتب على الأقل عنوان الوظيفة أو وصفها."}, status_code=400
        )

    features, numeric, country, full_text = build_features(posting)
    probability = float(model.predict_proba(features)[0, 1])

    if probability >= THRESHOLD:
        verdict = "fraudulent"
    elif probability >= THRESHOLD / 2:
        verdict = "suspicious"
    else:
        verdict = "real"

    return {
        "probability": probability,
        "verdict": verdict,
        "threshold": THRESHOLD,
        "notes": explain(numeric, country, full_text),
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": type(model).__name__,
        "n_features": int(getattr(tfidf, "idf_", np.array([])).shape[0])
        + int(sum(len(c) for c in encoder.categories_))
        + len(NUMERIC_COLUMNS),
        "threshold": THRESHOLD,
        "trained_at": METADATA.get("trained_at"),
        "sklearn_version": METADATA.get("sklearn_version"),
    }
