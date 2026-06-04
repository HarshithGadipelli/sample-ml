
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from core import DATA_DIR, ensure_dirs


def clamp(v, lo=0, hi=100):
    return float(max(lo, min(hi, v)))


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(7)
    random.seed(7)

    states = [
        ("Telangana", ["Hyderabad", "Warangal", "Nizamabad", "Khammam", "Ranga Reddy"]),
        ("Andhra Pradesh", ["Visakhapatnam", "Guntur", "Krishna", "Chittoor", "Anantapur"]),
        ("Karnataka", ["Bengaluru", "Mysuru", "Mangaluru", "Belagavi", "Dharwad"]),
        ("Tamil Nadu", ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"]),
        ("Kerala", ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Kollam"]),
        ("Maharashtra", ["Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad"]),
    ]
    college_types = ["Engineering", "Degree", "Autonomous", "University", "Arts", "Management"]

    colleges = []
    surveys = []
    college_count = 72

    for i in range(college_count):
        state, districts = states[i % len(states)]
        district = districts[i % len(districts)]
        college_type = college_types[i % len(college_types)]
        peer_group = college_type

        latent = 50 + (i % 9) * 4 + rng.normal(0, 4)
        infra = clamp(latent + rng.normal(0, 7))
        teaching = clamp(latent + rng.normal(0, 8))
        outcomes = clamp(latent + rng.normal(0, 10))
        research = clamp(latent + rng.normal(0, 9))
        values = clamp(60 + rng.normal(0, 8))
        greenery = clamp(55 + (i % 7) * 5 + rng.normal(0, 6))

        college_id = f"C{i+1:04d}"
        college_name = f"{state.split()[0]} {college_type} College {i+1:02d}"

        classroom_score = clamp(infra + rng.normal(0, 4))
        labs_score = clamp(infra + rng.normal(2, 5))
        library_score = clamp(infra + rng.normal(0, 5))
        internet_score = clamp(infra + rng.normal(0, 6))
        hostel_score = clamp(infra + rng.normal(0, 5))
        sports_score = clamp(55 + rng.normal(0, 10))
        faculty_score = clamp(teaching + rng.normal(0, 5))
        practical_score = clamp(teaching + rng.normal(0, 6))
        mentoring_score = clamp(teaching + rng.normal(0, 5))
        placements_score = clamp(outcomes + rng.normal(0, 7))
        research_score = clamp(research + rng.normal(0, 7))
        values_score = clamp(values + rng.normal(0, 5))
        campus_area = round(max(2, rng.normal(25, 10)), 2)

        nirf_score = clamp(
            0.25 * teaching + 0.20 * research + 0.25 * placements_score + 0.15 * infra + 0.15 * 60
        )
        naac_cgpa = round(max(1.5, min(4.0, 2.0 + (latent / 100.0) * 2.0 + rng.normal(0, 0.15))), 2)
        nba_score = clamp(0.35 * practical_score + 0.35 * placements_score + 0.30 * research_score)

        colleges.append(
            {
                "college_id": college_id,
                "college_name": college_name,
                "state": state,
                "district": district,
                "college_type": college_type,
                "peer_group": peer_group,
                "is_college": True,
                "campus_area_acres": campus_area,
                "greenery_score": greenery,
                "classroom_score": classroom_score,
                "labs_score": labs_score,
                "library_score": library_score,
                "internet_score": internet_score,
                "hostel_score": hostel_score,
                "sports_score": sports_score,
                "faculty_score": faculty_score,
                "practical_score": practical_score,
                "mentoring_score": mentoring_score,
                "placements_score": placements_score,
                "research_score": research_score,
                "values_score": values_score,
                "nirf_score": nirf_score,
                "naac_cgpa": naac_cgpa,
                "nba_score": nba_score,
                "overall_base_score": clamp(0.2 * teaching + 0.2 * outcomes + 0.2 * infra + 0.2 * research + 0.2 * values),
            }
        )

        # Six-month semester survey rows
        for month_no in range(1, 7):
            for student_idx in range(1, 13):
                student_id = f"{college_id}-S{month_no}{student_idx:02d}"
                attendance = clamp(rng.normal(82 if latent > 55 else 72, 8), 30, 100)
                marks = clamp(rng.normal(72 if latent > 55 else 60, 12), 25, 100)
                verified = attendance >= 75 and marks >= 50 and rng.random() > 0.05
                genuine_boost = 8 if verified else -7

                learning_env = clamp(teaching + rng.normal(0, 8) + genuine_boost)
                student_life = clamp(58 + greenery * 0.25 + rng.normal(0, 10) + (5 if verified else -5))
                academic_quality = clamp((teaching + research) / 2 + rng.normal(0, 7) + genuine_boost / 2)
                outcome_rating = clamp((placements_score + practical_score) / 2 + rng.normal(0, 8) + genuine_boost / 2)
                trust = clamp(60 + (8 if verified else -12) + rng.normal(0, 7))

                surveys.append(
                    {
                        "survey_id": f"{college_id}-M{month_no}-{student_idx:02d}",
                        "college_id": college_id,
                        "semester_id": 1,
                        "month_no": month_no,
                        "student_id": student_id,
                        "attendance_pct": round(attendance, 2),
                        "marks_pct": round(marks, 2),
                        "verified_student": bool(verified),
                        "learning_env": round(learning_env, 2),
                        "student_life": round(student_life, 2),
                        "academic_quality": round(academic_quality, 2),
                        "outcomes": round(outcome_rating, 2),
                        "trust": round(trust, 2),
                        "comment": random.choice(
                            [
                                "Teaching is interactive and practical.",
                                "Needs better labs and more project work.",
                                "Good environment and friendly faculty.",
                                "Placements need more industry exposure.",
                                "Student support is improving.",
                                "Good greenery and a calm campus.",
                            ]
                        ),
                    }
                )

    static_df = pd.DataFrame(colleges)
    survey_df = pd.DataFrame(surveys)

    # Add a few clearly different colleges for variety
    extra_rows = []
    for i in range(6):
        college_id = f"TOP{i+1:03d}"
        state = "Telangana" if i < 3 else "Karnataka"
        college_type = "Engineering" if i % 2 == 0 else "Autonomous"
        college_name = f"Premier {state} {college_type} Institute {i+1}"
        base = 88 - i * 2
        extra_rows.append(
            {
                "college_id": college_id,
                "college_name": college_name,
                "state": state,
                "district": "Hyderabad" if state == "Telangana" else "Bengaluru",
                "college_type": college_type,
                "peer_group": college_type,
                "is_college": True,
                "campus_area_acres": round(30 + i * 2, 2),
                "greenery_score": clamp(base + rng.normal(0, 2)),
                "classroom_score": clamp(base + rng.normal(0, 2)),
                "labs_score": clamp(base + rng.normal(0, 2)),
                "library_score": clamp(base + rng.normal(0, 2)),
                "internet_score": clamp(base + rng.normal(0, 2)),
                "hostel_score": clamp(base + rng.normal(0, 2)),
                "sports_score": clamp(base - 2 + rng.normal(0, 2)),
                "faculty_score": clamp(base + rng.normal(0, 2)),
                "practical_score": clamp(base + rng.normal(0, 2)),
                "mentoring_score": clamp(base + rng.normal(0, 2)),
                "placements_score": clamp(base + rng.normal(0, 2)),
                "research_score": clamp(base + rng.normal(0, 2)),
                "values_score": clamp(base + rng.normal(0, 2)),
                "nirf_score": clamp(base + 2),
                "naac_cgpa": round(min(4.0, 3.55 + i * 0.05), 2),
                "nba_score": clamp(base + 1),
                "overall_base_score": clamp(base),
            }
        )

        for month_no in range(1, 7):
            for student_idx in range(1, 15):
                attendance = clamp(rng.normal(92, 4), 60, 100)
                marks = clamp(rng.normal(86, 6), 40, 100)
                verified = attendance >= 78 and marks >= 55
                surveys.append(
                    {
                        "survey_id": f"{college_id}-M{month_no}-{student_idx:02d}",
                        "college_id": college_id,
                        "semester_id": 1,
                        "month_no": month_no,
                        "student_id": f"{college_id}-S{month_no}{student_idx:02d}",
                        "attendance_pct": round(attendance, 2),
                        "marks_pct": round(marks, 2),
                        "verified_student": bool(verified),
                        "learning_env": round(clamp(base + rng.normal(0, 4)), 2),
                        "student_life": round(clamp(base - 1 + rng.normal(0, 4)), 2),
                        "academic_quality": round(clamp(base + rng.normal(0, 4)), 2),
                        "outcomes": round(clamp(base + rng.normal(0, 4)), 2),
                        "trust": round(clamp(base + rng.normal(0, 4)), 2),
                        "comment": random.choice(
                            [
                                "Great college with strong academic culture.",
                                "Excellent mentoring and placement support.",
                                "Campus is well maintained and green.",
                            ]
                        ),
                    }
                )

    static_df = pd.concat([static_df, pd.DataFrame(extra_rows)], ignore_index=True)
    survey_df = pd.DataFrame(surveys)

    static_path = DATA_DIR / "static_colleges.csv"
    survey_path = DATA_DIR / "monthly_surveys.csv"

    static_df.to_csv(static_path, index=False)
    survey_df.to_csv(survey_path, index=False)

    # Template for new college input
    template = static_df.head(12).copy()
    template["college_id"] = [f"NEW{i+1:03d}" for i in range(len(template))]
    template["college_name"] = [f"New College Input {i+1}" for i in range(len(template))]
    template.to_csv(DATA_DIR / "new_colleges_template.csv", index=False)

    print(f"Generated static colleges: {len(static_df)} rows -> {static_path}")
    print(f"Generated surveys: {len(survey_df)} rows -> {survey_path}")
    print(f"Generated new college template -> {DATA_DIR / 'new_colleges_template.csv'}")


if __name__ == "__main__":
    main()
