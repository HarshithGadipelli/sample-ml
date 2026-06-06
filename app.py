
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
    add_new_college_to_csv,
    is_school_name,
    map_detailed_inputs_to_features,
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
    if is_school_name(name):
        flash(f"Validation Error: '{name}' is identified as a school. This system only accepts higher education institutions (colleges).", "danger")
        return redirect(url_for("index"))
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
    name = form.get("college_name", "").strip()
    
    # 1. School name validation
    if is_school_name(name):
        flash(f"Validation Error: '{name}' is identified as a school. This system only accepts higher education institutions (colleges).", "danger")
        return redirect(url_for("index"))

    # 2. Map Quick or Detailed inputs to model feature row
    row_mapped = map_detailed_inputs_to_features(form)
    row = predict_college_row(row_mapped, bundle)
    
    # 3. Save to database CSVs
    added = add_new_college_to_csv(row)
    if added:
        flash(f"College '{row['college_name']}' successfully predicted and added to the database.", "success")
    else:
        flash(f"College '{row['college_name']}' already exists in the database.", "info")

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


@app.route("/college-portal", methods=["GET"])
def college_portal():
    return render_template("college_portal.html")


@app.route("/submit-college-portal", methods=["POST"])
def submit_college_portal():
    form = request.form.to_dict()
    # Basic auth check (simulated)
    if form.get("auth_key") != "admin123":
        flash("Invalid authentication key.", "danger")
        return redirect(url_for("college_portal"))
    
    name = form.get("college_name", "").strip()
    static_df = load_static_colleges()
    found = fuzzy_find_college(name, static_df)
    
    if found is None:
        flash(f"College '{name}' not found in database. Please contact admin.", "warning")
        return redirect(url_for("college_portal"))
    
    # Update logic
    idx = static_df[static_df["college_id"] == found["college_id"]].index[0]
    for key in ["classroom_score", "labs_score", "internet_score", "greenery_score", 
                "faculty_score", "practical_score", "research_score", "mentoring_score", 
                "placements_score", "values_score"]:
        if key in form and form[key]:
            static_df.at[idx, key] = float(form[key])
            
    static_df.to_csv(DATA_DIR / "static_colleges.csv", index=False)
    flash(f"Successfully updated records for {found['college_name']}.", "success")
    return redirect(url_for("college_portal"))


@app.route("/student-review", methods=["GET"])
def student_review():
    return render_template("student_review.html")


@app.route("/submit-student-review", methods=["POST"])
def submit_student_review():
    form = request.form.to_dict()
    name = form.get("college_name", "").strip()
    static_df = load_static_colleges()
    found = fuzzy_find_college(name, static_df)
    
    if found is None:
        flash(f"College '{name}' not found. Review could not be submitted.", "warning")
        return redirect(url_for("student_review"))
    
    college_id = found["college_id"]
    sem_id = int(form.get("semester_id", 1))
    
    new_survey = {
        "survey_id": f"{college_id}-SEM{sem_id}-{pd.Timestamp.now().strftime('%H%M%S')}",
        "college_id": college_id,
        "semester_id": sem_id,
        "month_no": sem_id * 6, # 6 months per sem
        "student_id": form.get("student_id", "Anonymous"),
        "attendance_pct": float(form.get("attendance_pct", 75)),
        "marks_pct": float(form.get("marks_pct", 65)),
        "verified_student": True if float(form.get("attendance_pct", 75)) >= 75 and float(form.get("marks_pct", 65)) >= 50 else False,
        "learning_env": float(form.get("learning_env", 70)),
        "student_life": float(form.get("student_life", 70)),
        "academic_quality": float(form.get("academic_quality", 70)),
        "outcomes": float(form.get("outcomes", 70)),
        "trust": float(form.get("trust", 70)),
        "comment": form.get("comment", "")
    }
    
    survey_df = load_surveys()
    survey_df = pd.concat([survey_df, pd.DataFrame([new_survey])], ignore_index=True)
    survey_df.to_csv(DATA_DIR / "monthly_surveys.csv", index=False)
    
    flash(f"Thank you! Your semester {sem_id} review for {found['college_name']} was submitted.", "success")
    return redirect(url_for("student_review"))


if __name__ == "__main__":
    ensure_dirs()
    
    # Auto-generate new sample data and retrain model on startup
    print("Automatically regenerating sample data...")
    from generate_sample_data import main as generate_data
    generate_data()
    
    print("Automatically retraining model on new data...")
    from train_model import main as train_data
    train_data()
    
    app.run(debug=True, host="127.0.0.1", port=5000)
