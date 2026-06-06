from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths matching core.py
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def clamp(v, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, v)))

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(42)
    random.seed(42)

    # Real Telangana Colleges
    telangana_colleges = [
        {
            "college_id": "TS001",
            "college_name": "Jawaharlal Nehru Technological University (JNTUH)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 89.0,
            "greenery_score": 85.0,
            "nirf_score": 75.0,
            "naac_cgpa": 3.52,
            "nba_score": 82.0,
            "base_latent": 82.0,
        },
        {
            "college_id": "TS002",
            "college_name": "Osmania University College of Engineering (OUCE)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 120.0,
            "greenery_score": 88.0,
            "nirf_score": 78.0,
            "naac_cgpa": 3.58,
            "nba_score": 85.0,
            "base_latent": 84.0,
        },
        {
            "college_id": "TS003",
            "college_name": "Chaitanya Bharathi Institute of Technology (CBIT)",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 50.0,
            "greenery_score": 80.0,
            "nirf_score": 68.0,
            "naac_cgpa": 3.25,
            "nba_score": 78.0,
            "base_latent": 79.0,
        },
        {
            "college_id": "TS004",
            "college_name": "Vasavi College of Engineering (VCE)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 32.0,
            "greenery_score": 76.0,
            "nirf_score": 65.0,
            "naac_cgpa": 3.21,
            "nba_score": 76.0,
            "base_latent": 78.0,
        },
        {
            "college_id": "TS005",
            "college_name": "VNR Vignana Jyothi Institute of Engineering and Tech (VNRVJIET)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 22.0,
            "greenery_score": 78.0,
            "nirf_score": 70.0,
            "naac_cgpa": 3.40,
            "nba_score": 80.0,
            "base_latent": 81.0,
        },
        {
            "college_id": "TS006",
            "college_name": "Gokaraju Rangaraju Institute of Engineering and Tech (GRIET)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 24.0,
            "greenery_score": 75.0,
            "nirf_score": 60.0,
            "naac_cgpa": 3.12,
            "nba_score": 72.0,
            "base_latent": 73.0,
        },
        {
            "college_id": "TS007",
            "college_name": "Mahatma Gandhi Institute of Technology (MGIT)",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 30.0,
            "greenery_score": 74.0,
            "nirf_score": 58.0,
            "naac_cgpa": 3.08,
            "nba_score": 70.0,
            "base_latent": 71.0,
        },
        {
            "college_id": "TS008",
            "college_name": "Anurag University",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 55.0,
            "greenery_score": 82.0,
            "nirf_score": 62.0,
            "naac_cgpa": 3.15,
            "nba_score": 74.0,
            "base_latent": 74.0,
        },
        {
            "college_id": "TS009",
            "college_name": "Nizam College",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Degree",
            "peer_group": "Arts",
            "campus_area_acres": 25.0,
            "greenery_score": 84.0,
            "nirf_score": 55.0,
            "naac_cgpa": 3.10,
            "nba_score": 50.0,
            "base_latent": 70.0,
        },
        {
            "college_id": "TS010",
            "college_name": "St. Francis College for Women",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Degree",
            "peer_group": "Arts",
            "campus_area_acres": 8.0,
            "greenery_score": 72.0,
            "nirf_score": 60.0,
            "naac_cgpa": 3.25,
            "nba_score": 52.0,
            "base_latent": 76.0,
        },
        {
            "college_id": "TS011",
            "college_name": "Badruka College of Commerce and Arts",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Degree",
            "peer_group": "Management",
            "campus_area_acres": 5.0,
            "greenery_score": 55.0,
            "nirf_score": 52.0,
            "naac_cgpa": 3.01,
            "nba_score": 48.0,
            "base_latent": 68.0,
        },
        {
            "college_id": "TS012",
            "college_name": "Loyola Academy Degree and PG College",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "Degree",
            "peer_group": "Arts",
            "campus_area_acres": 97.0,
            "greenery_score": 90.0,
            "nirf_score": 66.0,
            "naac_cgpa": 3.32,
            "nba_score": 60.0,
            "base_latent": 77.0,
        },
        {
            "college_id": "TS013",
            "college_name": "International Institute of Information Technology (IIIT Hyderabad)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 66.0,
            "greenery_score": 85.0,
            "nirf_score": 90.0,
            "naac_cgpa": 3.65,
            "nba_score": 95.0,
            "base_latent": 92.0,
        },
        {
            "college_id": "TS014",
            "college_name": "Indian Institute of Technology (IIT Hyderabad)",
            "state": "Telangana",
            "district": "Sangareddy",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 576.0,
            "greenery_score": 95.0,
            "nirf_score": 92.0,
            "naac_cgpa": 3.80,
            "nba_score": 96.0,
            "base_latent": 94.0,
        },
        {
            "college_id": "TS015",
            "college_name": "National Institute of Technology (NIT Warangal)",
            "state": "Telangana",
            "district": "Warangal",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 248.0,
            "greenery_score": 90.0,
            "nirf_score": 85.0,
            "naac_cgpa": 3.70,
            "nba_score": 88.0,
            "base_latent": 88.0,
        },
        {
            "college_id": "TS016",
            "college_name": "Sreenidhi Institute of Science and Technology (SNIST)",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 33.0,
            "greenery_score": 78.0,
            "nirf_score": 62.0,
            "naac_cgpa": 3.10,
            "nba_score": 75.0,
            "base_latent": 75.0,
        },
        {
            "college_id": "TS017",
            "college_name": "Institute of Aeronautical Engineering (IARE)",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 17.0,
            "greenery_score": 72.0,
            "nirf_score": 60.0,
            "naac_cgpa": 3.20,
            "nba_score": 72.0,
            "base_latent": 73.0,
        },
        {
            "college_id": "TS018",
            "college_name": "CVR College of Engineering",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 33.0,
            "greenery_score": 80.0,
            "nirf_score": 63.0,
            "naac_cgpa": 3.18,
            "nba_score": 78.0,
            "base_latent": 76.0,
        },
        {
            "college_id": "TS019",
            "college_name": "Vardhaman College of Engineering",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 14.0,
            "greenery_score": 75.0,
            "nirf_score": 61.0,
            "naac_cgpa": 3.24,
            "nba_score": 75.0,
            "base_latent": 74.0,
        },
        {
            "college_id": "TS020",
            "college_name": "G Narayanamma Institute of Technology and Science (GNITS)",
            "state": "Telangana",
            "district": "Hyderabad",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 12.0,
            "greenery_score": 76.0,
            "nirf_score": 59.0,
            "naac_cgpa": 3.10,
            "nba_score": 74.0,
            "base_latent": 75.0,
        },
        {
            "college_id": "TS021",
            "college_name": "MVSR Engineering College",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 18.0,
            "greenery_score": 74.0,
            "nirf_score": 58.0,
            "naac_cgpa": 3.05,
            "nba_score": 70.0,
            "base_latent": 72.0,
        },
        {
            "college_id": "TS022",
            "college_name": "Kakatiya Institute of Technology and Science (KITS)",
            "state": "Telangana",
            "district": "Warangal",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 65.0,
            "greenery_score": 82.0,
            "nirf_score": 55.0,
            "naac_cgpa": 3.12,
            "nba_score": 68.0,
            "base_latent": 71.0,
        },
        {
            "college_id": "TS023",
            "college_name": "BITS Pilani, Hyderabad Campus",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 200.0,
            "greenery_score": 92.0,
            "nirf_score": 88.0,
            "naac_cgpa": 3.75,
            "nba_score": 90.0,
            "base_latent": 89.0,
        },
        {
            "college_id": "TS024",
            "college_name": "Mahindra University",
            "state": "Telangana",
            "district": "Medchal",
            "college_type": "University",
            "peer_group": "Engineering",
            "campus_area_acres": 130.0,
            "greenery_score": 85.0,
            "nirf_score": 65.0,
            "naac_cgpa": 2.90,
            "nba_score": 65.0,
            "base_latent": 75.0,
        },
        {
            "college_id": "TS025",
            "college_name": "Maturi Venkata Subba Rao (MVSR) Engineering College",
            "state": "Telangana",
            "district": "Ranga Reddy",
            "college_type": "Autonomous",
            "peer_group": "Engineering",
            "campus_area_acres": 18.0,
            "greenery_score": 75.0,
            "nirf_score": 55.0,
            "naac_cgpa": 3.10,
            "nba_score": 70.0,
            "base_latent": 72.0,
        }
    ]

    # Generate remaining colleges up to 500
    districts = ["Hyderabad", "Ranga Reddy", "Medchal", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Mahabubnagar", "Nalgonda", "Adilabad", "Sangareddy"]
    suffixes = ["Institute of Technology", "Engineering College", "College of Engineering", "Institute of Science and Technology", "University", "Degree College"]
    
    current_count = len(telangana_colleges)
    target_count = 500
    
    for i in range(current_count + 1, target_count + 1):
        dist = rng.choice(districts)
        suffix = rng.choice(suffixes)
        name_prefix = f"Telangana {dist} {rng.integers(1, 200)}"
        
        c_type = "University" if "University" in suffix else "Autonomous" if "Institute" in suffix else "Degree"
        peer_group = "Engineering" if "Engineering" in suffix or "Technology" in suffix else "Arts"
        
        latent = float(np.clip(rng.normal(65, 10), 40, 95))
        
        telangana_colleges.append({
            "college_id": f"TS{i:03d}",
            "college_name": f"{name_prefix} {suffix}",
            "state": "Telangana",
            "district": dist,
            "college_type": c_type,
            "peer_group": peer_group,
            "campus_area_acres": float(np.clip(rng.normal(20, 15), 2, 300)),
            "greenery_score": float(np.clip(rng.normal(latent + 5, 10), 30, 100)),
            "nirf_score": float(np.clip(rng.normal(latent - 10, 15), 20, 95)),
            "naac_cgpa": float(np.clip(rng.normal(latent / 25, 0.5), 1.5, 4.0)),
            "nba_score": float(np.clip(rng.normal(latent, 12), 30, 95)),
            "base_latent": latent,
        })
    colleges = []
    surveys = []

    for c_info in telangana_colleges:
        college_id = c_info["college_id"]
        latent = c_info["base_latent"]

        # Core scores (high-level features)
        greenery = c_info["greenery_score"]
        classroom_score = clamp(latent + rng.normal(0, 4))
        labs_score = clamp(latent + rng.normal(1, 4))
        library_score = clamp(latent + rng.normal(0, 3))
        internet_score = clamp(latent + rng.normal(0, 5))
        hostel_score = clamp(latent + rng.normal(-2, 6))
        sports_score = clamp(latent + rng.normal(2, 6))
        faculty_score = clamp(latent + rng.normal(0, 3))
        practical_score = clamp(latent + rng.normal(2, 4))
        mentoring_score = clamp(latent + rng.normal(1, 4))
        placements_score = clamp(latent + rng.normal(3, 5))
        research_score = clamp(latent + rng.normal(-2, 6))
        values_score = clamp(70 + rng.normal(0, 6))

        # Detailed Attributes (Learning Env, Student Life, Academic Quality, Outcomes, Integrity)
        class_size = clamp(60 + rng.normal(0, 10), 30, 120)
        attention_score = clamp(latent + rng.normal(0, 5))
        teaching_pace = clamp(70 + rng.normal(0, 8))
        doubt_solving_rate = clamp(latent + rng.normal(1, 4))
        teacher_responsiveness = clamp(latent + rng.normal(0, 4))
        lecture_prac_ratio = clamp(60 + rng.normal(0, 10))  # e.g., 60% practicals
        freedom_vs_pressure = clamp(70 + rng.normal(0, 8))

        mental_health = clamp(latent + rng.normal(-5, 8))
        anti_ragging = clamp(90 + rng.normal(0, 4), 80, 100)
        complaint_time = clamp(latent + rng.normal(0, 5))
        hostel_quality = hostel_score
        canteen_hygiene = clamp(latent + rng.normal(-2, 5))
        disabled_accessibility = clamp(latent + rng.normal(2, 8))
        first_gen_inclusion = clamp(80 + rng.normal(0, 5))

        syllabus_update = clamp(latent + rng.normal(0, 3))
        industry_projects = clamp(latent + rng.normal(1, 5))
        interdisciplinary = clamp(latent + rng.normal(-2, 6))
        peer_learning = clamp(latent + rng.normal(2, 4))
        research_papers = clamp(latent + rng.normal(-5, 10))
        innovation_hackathons = clamp(latent + rng.normal(1, 5))

        placement_rate = placements_score
        internship_rate = clamp(placements_score - 5)
        startup_creation = clamp(latent / 3 + rng.normal(0, 3))
        higher_studies = clamp(15 + rng.normal(0, 5))
        alumni_success = clamp(latent + rng.normal(0, 4))
        salary_dist = clamp(latent + rng.normal(1, 6))
        dropout_rate = clamp(2.5 + rng.normal(0, 1), 0.5, 10.0)
        pass_rate_trend = clamp(75 + rng.normal(0, 5), 60, 100)

        data_completeness = clamp(95 + rng.normal(0, 2), 90, 100)
        audit_confidence = clamp(latent + rng.normal(3, 4))
        response_consistency = clamp(88 + rng.normal(0, 3))
        anomaly_flags = clamp(rng.uniform(0, 5))  # lower is better
        evidence_verification = clamp(latent + rng.normal(2, 4))
        bias_detection = clamp(90 + rng.normal(0, 3))

        overall_base_score = clamp(0.25 * faculty_score + 0.25 * practical_score + 0.20 * classroom_score + 0.15 * research_score + 0.15 * values_score)

        college_record = {
            "college_id": college_id,
            "college_name": c_info["college_name"],
            "state": c_info["state"],
            "district": c_info["district"],
            "college_type": c_info["college_type"],
            "peer_group": c_info["peer_group"],
            "is_college": True,
            "campus_area_acres": c_info["campus_area_acres"],
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
            "nirf_score": c_info["nirf_score"],
            "naac_cgpa": c_info["naac_cgpa"],
            "nba_score": c_info["nba_score"],
            "overall_base_score": overall_base_score,
            # Learning Environment
            "class_size": class_size,
            "attention_score": attention_score,
            "teaching_pace": teaching_pace,
            "doubt_solving_rate": doubt_solving_rate,
            "teacher_responsiveness": teacher_responsiveness,
            "lecture_prac_ratio": lecture_prac_ratio,
            "freedom_vs_pressure": freedom_vs_pressure,
            # Student Life
            "mental_health": mental_health,
            "anti_ragging": anti_ragging,
            "complaint_time": complaint_time,
            "hostel_quality": hostel_quality,
            "canteen_hygiene": canteen_hygiene,
            "disabled_accessibility": disabled_accessibility,
            "first_gen_inclusion": first_gen_inclusion,
            # Academic Quality
            "syllabus_update": syllabus_update,
            "industry_projects": industry_projects,
            "interdisciplinary": interdisciplinary,
            "peer_learning": peer_learning,
            "research_papers": research_papers,
            "innovation_hackathons": innovation_hackathons,
            # Outcomes
            "placement_rate": placement_rate,
            "internship_rate": internship_rate,
            "startup_creation": startup_creation,
            "higher_studies": higher_studies,
            "alumni_success": alumni_success,
            "salary_dist": salary_dist,
            "dropout_rate": dropout_rate,
            "pass_rate_trend": pass_rate_trend,
            # Integrity and Trust
            "data_completeness": data_completeness,
            "audit_confidence": audit_confidence,
            "response_consistency": response_consistency,
            "anomaly_flags": anomaly_flags,
            "evidence_verification": evidence_verification,
            "bias_detection": bias_detection,
        }
        colleges.append(college_record)

        # Generate semester survey rows (6 months per semester log)
        # 12 students per college, answering semester surveys
        for student_idx in range(1, 15):
            student_id = f"{college_id}-S{student_idx:02d}"
            attendance = clamp(rng.normal(82 if latent > 75 else 72, 6), 35, 100)
            marks = clamp(rng.normal(75 if latent > 75 else 62, 10), 30, 100)
            verified = attendance >= 75 and marks >= 50 and rng.random() > 0.04
            genuine_boost = 6 if verified else -8

            learning_env = clamp(latent + rng.normal(0, 6) + genuine_boost)
            student_life = clamp(greenery * 0.4 + latent * 0.6 + rng.normal(0, 8) + (4 if verified else -6))
            academic_quality = clamp(latent + rng.normal(0, 5) + genuine_boost / 2)
            outcome_rating = clamp(placements_score + rng.normal(0, 6) + genuine_boost / 2)
            trust = clamp(latent + (8 if verified else -10) + rng.normal(0, 5))

            surveys.append(
                {
                    "survey_id": f"{college_id}-SEM1-{student_idx:02d}",
                    "college_id": college_id,
                    "semester_id": 1,
                    "month_no": 6,  # aggregated at month 6 (end of semester)
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
                            "Teaching incorporates both fundamentals and modern practical tools.",
                            "Good focus on practical sessions and mentoring.",
                            "Campus has trees and nature, very peaceful.",
                            "Career guidance is helpful from the beginning.",
                            "Excellent labs and digital board infrastructure.",
                            "Strictness is moderate, allowing good learning freedom.",
                        ]
                    ),
                }
            )

    static_df = pd.DataFrame(colleges)
    survey_df = pd.DataFrame(surveys)

    static_path = DATA_DIR / "static_colleges.csv"
    survey_path = DATA_DIR / "monthly_surveys.csv"

    static_df.to_csv(static_path, index=False)
    survey_df.to_csv(survey_path, index=False)

    print(f"Generated {len(static_df)} static colleges -> {static_path}")
    print(f"Generated {len(survey_df)} survey logs -> {survey_path}")

if __name__ == "__main__":
    main()
