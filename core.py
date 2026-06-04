
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

STATIC_PATH = DATA_DIR / "static_colleges.csv"
SURVEY_PATH = DATA_DIR / "monthly_surveys.csv"

MODEL_PATH = MODELS_DIR / "rank_model.pkl"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.pkl"

ALLOWED_COLLEGE_TYPES = [
    "Engineering",
    "Degree",
    "Autonomous",
    "University",
    "Medical",
    "Arts",
    "Polytechnic",
    "Management",
    "Law",
]

NUMERIC_FEATURES = [
    "campus_area_acres",
    "greenery_score",
    "classroom_score",
    "labs_score",
    "library_score",
    "internet_score",
    "hostel_score",
    "sports_score",
    "faculty_score",
    "practical_score",
    "mentoring_score",
    "placements_score",
    "research_score",
    "values_score",
    "nirf_score",
    "naac_cgpa",
    "nba_score",
    "survey_learning_env",
    "survey_student_life",
    "survey_academic_quality",
    "survey_outcomes",
    "survey_trust",
    "genuine_ratio",
    "survey_count",
    "trusted_survey_count",
    "survey_months",
    "learning_env_score",
    "student_life_score",
    "academic_score",
    "outcome_score",
    "infrastructure_score",
    "official_score",
    "trust_score",
    "values_alignment_score",
    "title_fit_score",
]

CATEGORICAL_FEATURES = [
    "state",
    "college_type",
    "peer_group",
]

RESULT_COLUMNS = [
    "college_id",
    "college_name",
    "state",
    "college_type",
    "peer_group",
    "final_score",
    "category",
    "overall_rank",
    "state_rank",
    "nirf_like_score",
    "naac_like_grade",
    "nba_like_score",
    "confidence_score",
    "remarks",
    "suggestions",
]

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "verified"])


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def normalize_0_100(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi <= lo:
        return pd.Series([50.0] * len(series), index=series.index)
    s = safe_numeric(series)
    return ((s - lo) / (hi - lo)).clip(0, 1) * 100.0


def category_from_score(score: float) -> int:
    if score < 40:
        return 3
    if score < 50:
        return 4
    if score < 60:
        return 5
    if score < 70:
        return 6
    if score < 80:
        return 7
    if score < 90:
        return 8
    return 9


def naac_grade_from_score(score: float) -> str:
    if score >= 90:
        return "A++"
    if score >= 80:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B++"
    if score >= 50:
        return "B+"
    if score >= 40:
        return "B"
    return "C"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float(np.nan)
    v = values[mask]
    w = weights[mask]
    s = float(w.sum())
    if s <= 0:
        return float(v.mean())
    return float(np.average(v, weights=w))


def _default_if_missing(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df


def load_static_colleges(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or STATIC_PATH)
    df = pd.read_csv(path)
    df = clean_columns(df)

    defaults = {
        "college_id": "",
        "college_name": "",
        "state": "Unknown",
        "district": "Unknown",
        "college_type": "Degree",
        "peer_group": "General",
        "is_college": True,
        "campus_area_acres": 10.0,
        "greenery_score": 70.0,
        "classroom_score": 70.0,
        "labs_score": 70.0,
        "library_score": 70.0,
        "internet_score": 70.0,
        "hostel_score": 70.0,
        "sports_score": 70.0,
        "faculty_score": 70.0,
        "practical_score": 70.0,
        "mentoring_score": 70.0,
        "placements_score": 70.0,
        "research_score": 70.0,
        "values_score": 70.0,
        "nirf_score": 50.0,
        "naac_cgpa": 2.5,
        "nba_score": 50.0,
        "overall_base_score": 60.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["college_id"] = df["college_id"].astype(str)
    df["college_name"] = df["college_name"].astype(str)
    df["state"] = df["state"].astype(str)
    df["district"] = df["district"].astype(str)
    df["college_type"] = df["college_type"].astype(str)
    df["peer_group"] = df["peer_group"].astype(str)
    df["is_college"] = coerce_bool(df["is_college"])

    for c in [
        "campus_area_acres",
        "greenery_score",
        "classroom_score",
        "labs_score",
        "library_score",
        "internet_score",
        "hostel_score",
        "sports_score",
        "faculty_score",
        "practical_score",
        "mentoring_score",
        "placements_score",
        "research_score",
        "values_score",
        "nirf_score",
        "naac_cgpa",
        "nba_score",
        "overall_base_score",
    ]:
        df[c] = safe_numeric(df[c], default=60.0 if c != "naac_cgpa" else 2.5)

    return df


def load_surveys(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or SURVEY_PATH)
    df = pd.read_csv(path)
    df = clean_columns(df)

    defaults = {
        "survey_id": "",
        "college_id": "",
        "semester_id": 1,
        "month_no": 1,
        "student_id": "",
        "attendance_pct": 75.0,
        "marks_pct": 50.0,
        "verified_student": True,
        "learning_env": 70.0,
        "student_life": 70.0,
        "academic_quality": 70.0,
        "outcomes": 70.0,
        "trust": 70.0,
        "comment": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["survey_id"] = df["survey_id"].astype(str)
    df["college_id"] = df["college_id"].astype(str)
    df["student_id"] = df["student_id"].astype(str)
    df["comment"] = df["comment"].astype(str)
    df["semester_id"] = pd.to_numeric(df["semester_id"], errors="coerce").fillna(1).astype(int)
    df["month_no"] = pd.to_numeric(df["month_no"], errors="coerce").fillna(1).astype(int)
    df["verified_student"] = coerce_bool(df["verified_student"])

    for c in ["attendance_pct", "marks_pct", "learning_env", "student_life", "academic_quality", "outcomes", "trust"]:
        df[c] = safe_numeric(df[c], default=70.0)

    return df


def aggregate_surveys(surveys: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(surveys)
    df["verified_student"] = coerce_bool(df["verified_student"])
    genuine_mask = (df["attendance_pct"] >= 75.0) & (df["marks_pct"] >= 50.0) & (df["verified_student"])

    df["survey_weight"] = np.where(
        genuine_mask,
        1.0 + (df["attendance_pct"] / 100.0) * 0.45 + (df["marks_pct"] / 100.0) * 0.25,
        0.15 + (df["attendance_pct"] / 100.0) * 0.05,
    )
    df["genuine_flag"] = genuine_mask.astype(int)

    rows = []
    group_cols = ["college_id", "semester_id"]
    for (college_id, semester_id), g in df.groupby(group_cols, dropna=False):
        weighted = lambda col: weighted_mean(g[col], g["survey_weight"])
        row = {
            "college_id": str(college_id),
            "semester_id": int(semester_id),
            "survey_learning_env": weighted("learning_env"),
            "survey_student_life": weighted("student_life"),
            "survey_academic_quality": weighted("academic_quality"),
            "survey_outcomes": weighted("outcomes"),
            "survey_trust": weighted("trust"),
            "survey_count": int(len(g)),
            "trusted_survey_count": int(g["genuine_flag"].sum()),
            "genuine_ratio": float(g["genuine_flag"].mean()) if len(g) else 0.0,
            "survey_months": int(g["month_no"].nunique()),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        for c in ["survey_learning_env", "survey_student_life", "survey_academic_quality", "survey_outcomes", "survey_trust"]:
            out[c] = out[c].fillna(70.0)
    return out


def compute_derived_scores(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    # Core sub-scores
    df["infrastructure_score"] = (
        df[["classroom_score", "labs_score", "library_score", "internet_score", "hostel_score", "sports_score", "greenery_score"]]
        .mean(axis=1)
        .clip(0, 100)
    )
    df["learning_env_score"] = (
        0.60 * df["survey_learning_env"].fillna(70)
        + 0.40 * df[["faculty_score", "practical_score", "classroom_score"]].mean(axis=1)
    ).clip(0, 100)
    df["student_life_score"] = (
        0.70 * df["survey_student_life"].fillna(70)
        + 0.30 * df[["hostel_score", "greenery_score", "sports_score"]].mean(axis=1)
    ).clip(0, 100)
    df["academic_score"] = (
        0.60 * df["survey_academic_quality"].fillna(70)
        + 0.40 * df[["research_score", "mentoring_score", "faculty_score"]].mean(axis=1)
    ).clip(0, 100)
    df["outcome_score"] = (
        0.60 * df["survey_outcomes"].fillna(70)
        + 0.40 * df[["placements_score", "research_score", "practical_score"]].mean(axis=1)
    ).clip(0, 100)

    df["official_score"] = (
        0.55 * safe_numeric(df["nirf_score"], 50).clip(0, 100)
        + 15.0 * (safe_numeric(df["naac_cgpa"], 2.5).clip(0, 4) / 4.0)
        + 0.30 * safe_numeric(df["nba_score"], 50).clip(0, 100)
    ).clip(0, 100)

    df["trust_score"] = (
        0.65 * df["survey_trust"].fillna(70)
        + 0.35 * (df["genuine_ratio"].fillna(0.5) * 100.0)
    ).clip(0, 100)

    df["values_alignment_score"] = (
        0.60 * df["values_score"].fillna(70)
        + 0.40 * df[["greenery_score", "mentoring_score"]].mean(axis=1)
    ).clip(0, 100)

    df["title_fit_score"] = (
        0.35 * df["learning_env_score"]
        + 0.20 * df["academic_score"]
        + 0.15 * df["outcome_score"]
        + 0.10 * df["student_life_score"]
        + 0.10 * df["values_alignment_score"]
        + 0.10 * df["official_score"]
    ).clip(0, 100)

    df["final_score"] = (
        0.20 * df["learning_env_score"]
        + 0.12 * df["student_life_score"]
        + 0.18 * df["academic_score"]
        + 0.18 * df["outcome_score"]
        + 0.12 * df["infrastructure_score"]
        + 0.10 * df["official_score"]
        + 0.05 * df["values_alignment_score"]
        + 0.05 * df["trust_score"]
        + 0.05 * df["title_fit_score"]
    ).clip(0, 100)

    df["category"] = df["final_score"].apply(category_from_score)
    df["naac_like_grade"] = df["final_score"].apply(naac_grade_from_score)
    df["nirf_like_score"] = (
        0.40 * df["learning_env_score"]
        + 0.15 * df["academic_score"]
        + 0.20 * df["outcome_score"]
        + 0.15 * df["official_score"]
        + 0.10 * df["trust_score"]
    ).clip(0, 100)
    df["nba_like_score"] = (
        0.35 * df["academic_score"]
        + 0.25 * df["outcome_score"]
        + 0.20 * df["infrastructure_score"]
        + 0.10 * df["learning_env_score"]
        + 0.10 * df["official_score"]
    ).clip(0, 100)

    return df


def merge_static_and_surveys(static_df: pd.DataFrame, survey_df: pd.DataFrame) -> pd.DataFrame:
    if survey_df is None or survey_df.empty:
        survey_df = pd.DataFrame(columns=[
            "college_id", "survey_learning_env", "survey_student_life", "survey_academic_quality",
            "survey_outcomes", "survey_trust", "survey_count", "trusted_survey_count",
            "genuine_ratio", "survey_months"
        ])

    survey_df = clean_columns(survey_df)
    # If raw surveys are passed, aggregate them first.
    if "survey_learning_env" not in survey_df.columns:
        survey_df = aggregate_surveys(survey_df)

    merged = static_df.merge(survey_df, on="college_id", how="left", suffixes=("", "_survey"))
    for col in ["survey_learning_env", "survey_student_life", "survey_academic_quality", "survey_outcomes", "survey_trust"]:
        if col not in merged.columns:
            merged[col] = 70.0
        merged[col] = merged[col].fillna(70.0)
    for col in ["survey_count", "trusted_survey_count", "survey_months"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)
    if "genuine_ratio" not in merged.columns:
        merged["genuine_ratio"] = 0.0
    merged["genuine_ratio"] = merged["genuine_ratio"].fillna(0.0)
    return compute_derived_scores(merged)


def build_training_frame(static_path: Path | None = None, survey_path: Path | None = None) -> pd.DataFrame:
    static_df = load_static_colleges(static_path)
    surveys = load_surveys(survey_path)
    semester_df = aggregate_surveys(surveys)
    frame = merge_static_and_surveys(static_df, semester_df)

    # Training target: same composite score with mild noise for realism
    rng = np.random.default_rng(42)
    frame["target_score"] = (frame["final_score"] + rng.normal(0, 1.2, len(frame))).clip(0, 100)

    # Keep only colleges
    frame = frame[frame["is_college"]].copy()
    return frame


def prepare_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = frame.copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = safe_numeric(df[col], 0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).fillna("Unknown")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = safe_numeric(df["target_score"], 60.0)
    return X, y


def train_model_from_frame(frame: pd.DataFrame) -> Dict:
    X, y = prepare_features(frame)

    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
    }

    # Fit on full data for final model
    pipeline.fit(X, y)

    artifacts = {
        "metrics": metrics,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_frame": frame[[
            "college_id", "college_name", "state", "district", "college_type", "peer_group",
            "target_score", "final_score", "category", "nirf_like_score", "naac_like_grade",
            "nba_like_score", "learning_env_score", "student_life_score", "academic_score",
            "outcome_score", "infrastructure_score", "official_score", "trust_score",
            "values_alignment_score", "title_fit_score", "survey_count", "trusted_survey_count",
            "genuine_ratio", "is_college"
        ]].copy(),
        "feature_mean_vector": X[numeric_features].mean(numeric_only=True).to_dict(),
        "feature_std_vector": X[numeric_features].std(numeric_only=True).replace(0, 1).to_dict(),
        "state_benchmark": frame.groupby("state")["final_score"].max().to_dict(),
        "state_top_college": (
            frame.sort_values(["state", "final_score"], ascending=[True, False])
            .groupby("state")
            .first()[["college_id", "college_name", "final_score"]]
            .to_dict("index")
        ),
    }
    return {"pipeline": pipeline, "artifacts": artifacts, "metrics": metrics}


def save_model_bundle(bundle: Dict) -> None:
    ensure_dirs()
    joblib.dump(bundle["pipeline"], MODEL_PATH)
    joblib.dump(bundle["artifacts"], ARTIFACTS_PATH)


def load_model_bundle() -> Dict:
    if not MODEL_PATH.exists() or not ARTIFACTS_PATH.exists():
        raise FileNotFoundError("Model files not found. Run train_model.py first.")
    return {
        "pipeline": joblib.load(MODEL_PATH),
        "artifacts": joblib.load(ARTIFACTS_PATH),
    }


def _completed_feature_frame(input_df: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
    df = input_df.copy()
    # Fill all expected columns for new predictions
    defaults = {
        "campus_area_acres": 10.0,
        "greenery_score": 70.0,
        "classroom_score": 70.0,
        "labs_score": 70.0,
        "library_score": 70.0,
        "internet_score": 70.0,
        "hostel_score": 70.0,
        "sports_score": 70.0,
        "faculty_score": 70.0,
        "practical_score": 70.0,
        "mentoring_score": 70.0,
        "placements_score": 70.0,
        "research_score": 70.0,
        "values_score": 70.0,
        "nirf_score": 50.0,
        "naac_cgpa": 2.5,
        "nba_score": 50.0,
        "survey_learning_env": 70.0,
        "survey_student_life": 70.0,
        "survey_academic_quality": 70.0,
        "survey_outcomes": 70.0,
        "survey_trust": 70.0,
        "genuine_ratio": 0.5,
        "survey_count": 0,
        "trusted_survey_count": 0,
        "survey_months": 0,
        "college_type": "Degree",
        "state": "Unknown",
        "peer_group": "General",
    }
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = defaults.get(col, 0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = defaults.get(col, "Unknown")

    # ensure strings
    df["state"] = df["state"].astype(str)
    df["college_type"] = df["college_type"].astype(str)
    df["peer_group"] = df["peer_group"].astype(str)

    # Derived features
    temp = compute_derived_scores(df)
    return temp


def predict_college_row(input_row: Dict, bundle: Dict) -> Dict:
    pipeline = bundle["pipeline"]
    artifacts = bundle["artifacts"]

    df = pd.DataFrame([input_row])
    df = _completed_feature_frame(df, artifacts)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    predicted_score = float(pipeline.predict(X)[0])
    category = category_from_score(predicted_score)

    # confidence based on completeness and distance to training mean
    completeness = float(X.notna().mean().mean())
    numeric_vec = X[NUMERIC_FEATURES].iloc[0].astype(float).values
    train_mean = np.array([artifacts["feature_mean_vector"].get(c, 0.0) for c in NUMERIC_FEATURES], dtype=float)
    train_std = np.array([artifacts["feature_std_vector"].get(c, 1.0) for c in NUMERIC_FEATURES], dtype=float)
    z = (numeric_vec - train_mean) / np.where(train_std == 0, 1.0, train_std)
    dist = float(np.sqrt(np.mean(z ** 2)))
    similarity = math.exp(-dist / 2.0)
    confidence = max(0.0, min(100.0, 100.0 * (0.55 * completeness + 0.45 * similarity)))

    row = df.iloc[0].to_dict()
    row["final_score"] = predicted_score
    row["category"] = category
    row["naac_like_grade"] = naac_grade_from_score(predicted_score)
    row["nirf_like_score"] = float(
        (0.40 * row["learning_env_score"])
        + (0.15 * row["academic_score"])
        + (0.20 * row["outcome_score"])
        + (0.15 * row["official_score"])
        + (0.10 * row["trust_score"])
    )
    row["nba_like_score"] = float(
        (0.35 * row["academic_score"])
        + (0.25 * row["outcome_score"])
        + (0.20 * row["infrastructure_score"])
        + (0.10 * row["learning_env_score"])
        + (0.10 * row["official_score"])
    )
    row["confidence_score"] = confidence

    return row


def _gap_suggestions(row: Dict, benchmark: Dict | None = None) -> List[str]:
    tips = []
    if row["learning_env_score"] < 75:
        tips.append("Improve teaching interaction, practical sessions, and doubt-solving speed.")
    if row["student_life_score"] < 75:
        tips.append("Strengthen safety, hostel quality, mentoring, and grievance response.")
    if row["academic_score"] < 75:
        tips.append("Add more projects, research activity, and interdisciplinary learning.")
    if row["outcome_score"] < 75:
        tips.append("Raise internship support, placement quality, and alumni engagement.")
    if row["infrastructure_score"] < 75:
        tips.append("Upgrade labs, internet, library, classrooms, and greenery.")
    if row["trust_score"] < 75:
        tips.append("Increase verified student responses and audit the survey process.")
    if row["values_alignment_score"] < 75:
        tips.append("Add human values, constitution, and environment-based activities.")

    if benchmark is not None:
        gap = float(max(0.0, benchmark["final_score"] - row["final_score"]))
        if gap > 0:
            tips.append(f"To challenge rank 1 in {row['state']}, close a score gap of about {gap:.1f} points.")
    if not tips:
        tips.append("Keep the current balance and sustain verified student feedback every semester.")
    return tips


def _benchmark_for_state(row: Dict, training_frame: pd.DataFrame) -> Dict | None:
    st = str(row.get("state", "Unknown"))
    st_frame = training_frame[training_frame["state"].astype(str) == st]
    if st_frame.empty:
        return None
    top = st_frame.sort_values("final_score", ascending=False).iloc[0].to_dict()
    return top


def rank_all_colleges(bundle: Dict, static_df: pd.DataFrame, survey_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if survey_df is None or survey_df.empty:
        survey_df = load_surveys()
    frame = merge_static_and_surveys(static_df, survey_df)
    pipeline = bundle["pipeline"]
    artifacts = bundle["artifacts"]
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    frame["predicted_score"] = pipeline.predict(X)
    frame["category"] = frame["predicted_score"].apply(category_from_score)
    frame["overall_rank"] = frame["predicted_score"].rank(ascending=False, method="min").astype(int)
    frame["state_rank"] = frame.groupby("state")["predicted_score"].rank(ascending=False, method="min").astype(int)
    frame["confidence_score"] = frame.apply(
        lambda r: predict_college_row(r.to_dict(), bundle)["confidence_score"], axis=1
    )
    frame["remarks"] = frame.apply(lambda r: build_remarks(r.to_dict()), axis=1)
    frame["suggestions"] = frame.apply(lambda r: " | ".join(_gap_suggestions(r.to_dict(), _benchmark_for_state(r.to_dict(), artifacts["training_frame"]))), axis=1)
    return frame.sort_values("overall_rank").reset_index(drop=True)


def build_remarks(row: Dict) -> str:
    parts = []
    if row["final_score"] >= 85:
        parts.append("Excellent overall college environment.")
    elif row["final_score"] >= 70:
        parts.append("Very good education quality with a strong base.")
    elif row["final_score"] >= 55:
        parts.append("Average college; clear room for improvement.")
    else:
        parts.append("Weak score; major improvements needed.")

    if row["greenery_score"] >= 80:
        parts.append("Green and student-friendly campus.")
    if row["outcome_score"] >= 80:
        parts.append("Strong career and placement support.")
    if row["trust_score"] >= 80:
        parts.append("Survey trust looks solid.")
    return " ".join(parts)


def build_result_payload(row: Dict, bundle: Dict, existing_rank_df: Optional[pd.DataFrame] = None) -> Dict:
    if "final_score" not in row:
        row = predict_college_row(row, bundle)

    benchmark = _benchmark_for_state(row, bundle["artifacts"]["training_frame"])
    tips = _gap_suggestions(row, benchmark)
    remarks = build_remarks(row)

    overall_rank = None
    state_rank = None
    if existing_rank_df is not None and not existing_rank_df.empty:
        # include the input row into the ranking table for an estimated rank
        all_rows = existing_rank_df.copy()
        if "college_id" not in row or not row.get("college_id"):
            row["college_id"] = f"NEW-{abs(hash(row.get('college_name','new'))) % 10**8}"
        new_row = pd.DataFrame([row])
        new_row = compute_derived_scores(new_row)
        new_row["predicted_score"] = float(row["final_score"])
        new_row["category"] = category_from_score(float(row["final_score"]))
        new_row["overall_rank"] = np.nan
        new_row["state_rank"] = np.nan
        all_rows = pd.concat([all_rows, new_row[all_rows.columns.intersection(new_row.columns)]], ignore_index=True, sort=False)
        all_rows["predicted_score"] = all_rows.get("predicted_score", all_rows.get("final_score"))
        all_rows["predicted_score"] = pd.to_numeric(all_rows["predicted_score"], errors="coerce")
        if "state" in all_rows.columns:
            all_rows["overall_rank_calc"] = all_rows["predicted_score"].rank(ascending=False, method="min")
            all_rows["state_rank_calc"] = all_rows.groupby("state")["predicted_score"].rank(ascending=False, method="min")
            matched = all_rows[all_rows["college_name"].astype(str).str.lower() == str(row.get("college_name", "")).lower()]
            if not matched.empty:
                overall_rank = int(matched.iloc[0]["overall_rank_calc"])
                state_rank = int(matched.iloc[0]["state_rank_calc"])

    return {
        "college_id": row.get("college_id", ""),
        "college_name": row.get("college_name", ""),
        "state": row.get("state", "Unknown"),
        "district": row.get("district", ""),
        "college_type": row.get("college_type", "Degree"),
        "peer_group": row.get("peer_group", "General"),
        "final_score": round(float(row["final_score"]), 2),
        "category": int(row["category"]),
        "overall_rank": overall_rank,
        "state_rank": state_rank,
        "nirf_like_score": round(float(row["nirf_like_score"]), 2),
        "naac_like_grade": row["naac_like_grade"],
        "nba_like_score": round(float(row["nba_like_score"]), 2),
        "confidence_score": round(float(row.get("confidence_score", 0.0)), 2),
        "learning_env_score": round(float(row["learning_env_score"]), 2),
        "student_life_score": round(float(row["student_life_score"]), 2),
        "academic_score": round(float(row["academic_score"]), 2),
        "outcome_score": round(float(row["outcome_score"]), 2),
        "infrastructure_score": round(float(row["infrastructure_score"]), 2),
        "official_score": round(float(row["official_score"]), 2),
        "trust_score": round(float(row["trust_score"]), 2),
        "values_alignment_score": round(float(row["values_alignment_score"]), 2),
        "title_fit_score": round(float(row["title_fit_score"]), 2),
        "remarks": remarks,
        "suggestions": tips,
        "rank_1_benchmark": benchmark,
    }


def fuzzy_find_college(name: str, static_df: pd.DataFrame) -> Optional[pd.Series]:
    name = str(name).strip().lower()
    if not name:
        return None
    exact = static_df[static_df["college_name"].str.lower() == name]
    if not exact.empty:
        return exact.iloc[0]
    matches = get_close_matches(name, static_df["college_name"].astype(str).str.lower().tolist(), n=1, cutoff=0.55)
    if not matches:
        return None
    match_name = matches[0]
    found = static_df[static_df["college_name"].str.lower() == match_name]
    if found.empty:
        return None
    return found.iloc[0]


def load_existing_ranking(bundle: Dict) -> pd.DataFrame:
    static_df = load_static_colleges()
    surveys = load_surveys()
    ranked = rank_all_colleges(bundle, static_df, surveys)
    return ranked


def estimate_state_rank_for_new(row: Dict, ranked_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    if ranked_df is None or ranked_df.empty:
        return None, None
    st = str(row.get("state", "Unknown"))
    candidate = pd.DataFrame([row])
    candidate["predicted_score"] = float(row["final_score"])
    candidate["college_name"] = str(row.get("college_name", "New College"))
    candidate["state"] = st
    candidate["college_type"] = str(row.get("college_type", "Degree"))
    candidate["peer_group"] = str(row.get("peer_group", "General"))
    candidate["overall_rank"] = np.nan
    candidate["state_rank"] = np.nan
    pool = ranked_df.copy()
    needed_cols = list(pool.columns)
    for c in needed_cols:
        if c not in candidate.columns:
            candidate[c] = np.nan
    all_df = pd.concat([pool, candidate[needed_cols]], ignore_index=True, sort=False)
    all_df["predicted_score"] = pd.to_numeric(all_df["predicted_score"], errors="coerce")
    all_df["overall_rank_calc"] = all_df["predicted_score"].rank(ascending=False, method="min")
    all_df["state_rank_calc"] = all_df.groupby("state")["predicted_score"].rank(ascending=False, method="min")
    matched = all_df[(all_df["college_name"].astype(str).str.lower() == str(row.get("college_name", "")).lower()) & (all_df["state"].astype(str) == st)]
    if matched.empty:
        return None, None
    return int(matched.iloc[0]["overall_rank_calc"]), int(matched.iloc[0]["state_rank_calc"])


def ensure_schema_for_new_input(input_data: Dict) -> Dict:
    out = dict(input_data)
    defaults = {
        "college_id": f"NEW-{abs(hash(input_data.get('college_name','new'))) % 10**8}",
        "district": "Unknown",
        "peer_group": "General",
        "college_type": "Degree",
        "state": "Unknown",
        "campus_area_acres": 10.0,
        "greenery_score": 70.0,
        "classroom_score": 70.0,
        "labs_score": 70.0,
        "library_score": 70.0,
        "internet_score": 70.0,
        "hostel_score": 70.0,
        "sports_score": 70.0,
        "faculty_score": 70.0,
        "practical_score": 70.0,
        "mentoring_score": 70.0,
        "placements_score": 70.0,
        "research_score": 70.0,
        "values_score": 70.0,
        "nirf_score": 50.0,
        "naac_cgpa": 2.5,
        "nba_score": 50.0,
        "survey_learning_env": 70.0,
        "survey_student_life": 70.0,
        "survey_academic_quality": 70.0,
        "survey_outcomes": 70.0,
        "survey_trust": 70.0,
        "genuine_ratio": 0.5,
        "survey_count": 0,
        "trusted_survey_count": 0,
        "survey_months": 0,
    }
    for key, val in defaults.items():
        out.setdefault(key, val)
    return out
