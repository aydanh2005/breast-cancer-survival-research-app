"""Train compact models for the public Streamlit demonstration.

The repeated-validation workflow in the thesis project remains the source of
all scientific conclusions. The smaller Random Survival Forest created here is
used only to keep the online demonstration artifact deployable.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest

from run_repeated_validation import (
    ALL_FEATURES,
    EVALUATION_TIME,
    create_npi_frame,
    make_cox_pipeline,
    make_multivariable_preprocessor,
    make_npi_pipeline,
    make_survival_target,
    survival_probability_at_time,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "metabric_analysis_cohort.csv"
OOF_FILE = PROJECT_ROOT / "outputs" / "repeated_validation_predictions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "streamlit_deployment_models.joblib"
METADATA_FILE = MODEL_DIR / "streamlit_deployment_metadata.json"

RANDOM_SEED = 2026
DEPLOYMENT_TREES = 60
DEPLOYMENT_MAX_DEPTH = 10


def make_compact_rsf_pipeline() -> Pipeline:
    """Create a bounded-size RSF for software demonstration only."""

    return Pipeline(
        steps=[
            ("preprocessor", make_multivariable_preprocessor()),
            (
                "model",
                RandomSurvivalForest(
                    n_estimators=DEPLOYMENT_TREES,
                    max_depth=DEPLOYMENT_MAX_DEPTH,
                    min_samples_split=20,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def oof_thresholds() -> dict[str, float] | None:
    """Derive descriptive quartiles from scientific OOF predictions."""

    if not OOF_FILE.exists():
        return None
    predictions = pd.read_csv(OOF_FILE)
    required = {"patient_id", "model_key", "event_risk_59_9m"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")
    rsf = predictions.loc[
        predictions["model_key"].eq("random_survival_forest")
    ]
    patient_risk = rsf.groupby("patient_id")["event_risk_59_9m"].mean()
    return {
        "lower_quartile": float(patient_risk.quantile(0.25)),
        "upper_quartile": float(patient_risk.quantile(0.75)),
    }


def main() -> None:
    """Fit, compress, and document the public demonstration models."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = validate_dataset(pd.read_csv(DATA_FILE))
    target = make_survival_target(data)

    npi_model = make_npi_pipeline().fit(create_npi_frame(data), target)
    cox_model = make_cox_pipeline().fit(data[ALL_FEATURES], target)
    rsf_model = make_compact_rsf_pipeline().fit(data[ALL_FEATURES], target)

    models = {
        "npi_cox": npi_model,
        "multivariable_cox": cox_model,
        "random_survival_forest": rsf_model,
    }
    joblib.dump(models, MODEL_FILE, compress=3)

    rsf_risk = 1.0 - survival_probability_at_time(
        rsf_model, data[ALL_FEATURES], EVALUATION_TIME
    )
    metadata = {
        "artifact_purpose": "Public Streamlit academic research demonstration",
        "clinical_use": False,
        "deployment_model_is_scientific_evidence": False,
        "scientific_results_source": "Repeated 5-fold cross-validation with 3 repetitions",
        "evaluation_time_months": EVALUATION_TIME,
        "training_patients": int(len(data)),
        "training_events": int(data["event"].sum()),
        "features": ALL_FEATURES,
        "random_seed": RANDOM_SEED,
        "deployment_rsf_trees": DEPLOYMENT_TREES,
        "deployment_rsf_max_depth": DEPLOYMENT_MAX_DEPTH,
        "oof_rsf_descriptive_thresholds": oof_thresholds(),
        "full_cohort_rsf_mean_predicted_risk": float(np.mean(rsf_risk)),
        "important_limitation": (
            "This compact full-cohort fit supports the online software demonstration "
            "only. Thesis conclusions use the saved repeated out-of-fold results."
        ),
    }
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
    print("PUBLIC STREAMLIT DEPLOYMENT MODELS")
    print("=" * 70)
    print(f"Patients: {len(data)} | Events: {int(data['event'].sum())}")
    print(f"Compact RSF: {DEPLOYMENT_TREES} trees, max depth {DEPLOYMENT_MAX_DEPTH}")
    print(f"Compressed model size: {size_mb:.1f} MB")
    print(f"Saved: {MODEL_FILE}")
    print(f"Saved: {METADATA_FILE}")
    if size_mb >= 95:
        raise RuntimeError("Deployment model is too large for a standard GitHub repository.")


if __name__ == "__main__":
    main()
