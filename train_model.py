
from __future__ import annotations

import argparse
from pathlib import Path

from core import (
    BASE_DIR,
    DATA_DIR,
    MODEL_PATH,
    ARTIFACTS_PATH,
    build_training_frame,
    ensure_dirs,
    save_model_bundle,
    train_model_from_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the college ranking model.")
    parser.add_argument("--static", type=str, default=str(DATA_DIR / "static_colleges.csv"))
    parser.add_argument("--surveys", type=str, default=str(DATA_DIR / "monthly_surveys.csv"))
    args = parser.parse_args()

    ensure_dirs()
    frame = build_training_frame(Path(args.static), Path(args.surveys))
    bundle = train_model_from_frame(frame)
    save_model_bundle(bundle)
    print("Model trained successfully.")
    print(f"Rows: {len(frame)}")
    print(f"MAE: {bundle['metrics']['mae']:.3f}")
    print(f"R2: {bundle['metrics']['r2']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Artifacts saved to: {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
