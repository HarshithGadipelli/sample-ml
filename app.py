
from __future__ import annotations

from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from core import (
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_PATH,
    MODEL_PATH,
    ensure_dirs,
    fuzzy_find_college,
    load_existing_ranking,
    load_model_bundle,
    load_static_colleges,
    load_surveys,
    ensure_schema_for_new_input,
    build_result_payload,
    predict_college_row,
    rank_all_colleges,
)
from train_model import main as train_main
import pandas as pd

app = Flask(__name__)
app.secret_key = "college-ranking-secret-key"


def load_bundle_safely():
    ensure_dirs()
    if not MODEL_PATH.exists() or not ARTIFACTS_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run train_model.py first.")
    return load_model_bundle()


def _base_context():
    try:
        bundle = load_bundle_safely()
        static_df = load_static_colleges()
        survey_df = load_surveys()
        ranked_df = rank_all_colleges(bundle, static_df, survey_df)
        metrics = bundle["artifacts"]["metrics"]
    except Exception as exc:
        bundle = None
        ranked_df = pd.DataFrame()
        metrics = {}
    return bundle, ranked_df, metrics


@app.route("/", methods=["GET"])
def index():
    bundle, ranked_df, metrics = _base_context()
    colleges = []
    if not ranked_df.empty:
        colleges = ranked_df[["college_name", "state", "category", "predicted_score"]].head(12).to_dict("records")
    return render_template("index.html", colleges=colleges, metrics=metrics)


@app.route("/search", methods=["POST"])
def search():
    try:
        bundle = load_bundle_safely()
    except Exception as exc:
        flash(f"Model not ready: {exc}", "danger")
        return redirect(url_for("index"))
    name = request.form.get("college_name", "").strip()
    static_df = load_static_colleges()
    ranked_df = load_existing_ranking(bundle)

    found = fuzzy_find_college(name, static_df)
    if found is None:
        flash("College not found. Try a different spelling or use the new college form.", "warning")
        return redirect(url_for("index"))

    row = found.to_dict()
    ranked_row = ranked_df[ranked_df["college_id"].astype(str) == str(row["college_id"])]
    if not ranked_row.empty:
        row["final_score"] = float(ranked_row.iloc[0]["predicted_score"])
        row["category"] = int(ranked_row.iloc[0]["category"])
        row["confidence_score"] = float(ranked_row.iloc[0]["confidence_score"])
        row["naac_like_grade"] = str(ranked_row.iloc[0]["naac_like_grade"])
        row["nirf_like_score"] = float(ranked_row.iloc[0]["nirf_like_score"])
        row["nba_like_score"] = float(ranked_row.iloc[0]["nba_like_score"])
        row["overall_rank"] = int(ranked_row.iloc[0]["overall_rank"])
        row["state_rank"] = int(ranked_row.iloc[0]["state_rank"])
        row["suggestions"] = ranked_row.iloc[0]["suggestions"]
        row["remarks"] = ranked_row.iloc[0]["remarks"]
        row["learning_env_score"] = float(ranked_row.iloc[0]["learning_env_score"])
        row["student_life_score"] = float(ranked_row.iloc[0]["student_life_score"])
        row["academic_score"] = float(ranked_row.iloc[0]["academic_score"])
        row["outcome_score"] = float(ranked_row.iloc[0]["outcome_score"])
        row["infrastructure_score"] = float(ranked_row.iloc[0]["infrastructure_score"])
        row["official_score"] = float(ranked_row.iloc[0]["official_score"])
        row["trust_score"] = float(ranked_row.iloc[0]["trust_score"])
        row["values_alignment_score"] = float(ranked_row.iloc[0]["values_alignment_score"])
        row["title_fit_score"] = float(ranked_row.iloc[0]["title_fit_score"])
    else:
        row = predict_college_row(row, bundle)

    payload = build_result_payload(row, bundle, ranked_df)
    return render_template("result.html", result=payload, metrics=bundle["artifacts"]["metrics"], mode="existing")


@app.route("/predict-new", methods=["POST"])
def predict_new():
    try:
        bundle = load_bundle_safely()
    except Exception as exc:
        flash(f"Model not ready: {exc}", "danger")
        return redirect(url_for("index"))
    ranked_df = load_existing_ranking(bundle)

    form = request.form.to_dict()
    cleaned = {
        "college_id": form.get("college_id", "").strip() or None,
        "college_name": form.get("college_name", "").strip(),
        "state": form.get("state", "Unknown").strip(),
        "district": form.get("district", "Unknown").strip(),
        "college_type": form.get("college_type", "Degree").strip(),
        "peer_group": form.get("peer_group", form.get("college_type", "Degree")).strip(),
    }
    numeric_fields = [
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
    ]
    for f in numeric_fields:
        cleaned[f] = form.get(f, "")
    row = ensure_schema_for_new_input(cleaned)
    payload = build_result_payload(row, bundle, ranked_df)
    return render_template("result.html", result=payload, metrics=bundle["artifacts"]["metrics"], mode="new")


@app.route("/admin", methods=["GET"])
def admin():
    bundle, ranked_df, metrics = _base_context()
    preview = []
    if not ranked_df.empty:
        preview = ranked_df[["college_name", "state", "predicted_score", "category", "overall_rank", "state_rank"]].head(15).to_dict("records")
    return render_template("admin.html", preview=preview, metrics=metrics)


@app.route("/admin/upload-static", methods=["POST"])
def upload_static():
    file = request.files.get("static_file")
    if not file:
        flash("Please upload a static CSV file.", "danger")
        return redirect(url_for("admin"))
    save_path = DATA_DIR / "static_colleges.csv"
    file.save(save_path)
    flash("Static college dataset updated.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/upload-surveys", methods=["POST"])
def upload_surveys():
    file = request.files.get("survey_file")
    if not file:
        flash("Please upload a surveys CSV file.", "danger")
        return redirect(url_for("admin"))
    save_path = DATA_DIR / "monthly_surveys.csv"
    file.save(save_path)
    flash("Monthly survey dataset updated.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/retrain", methods=["POST"])
def retrain():
    try:
        train_main()
        flash("Model retrained successfully.", "success")
    except Exception as exc:
        flash(f"Retraining failed: {exc}", "danger")
    return redirect(url_for("admin"))


if __name__ == "__main__":
    ensure_dirs()
    app.run(debug=True, host="127.0.0.1", port=5000)
