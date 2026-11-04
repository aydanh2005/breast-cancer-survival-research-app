"""Pure input validation used by the Streamlit research demonstrator."""

from __future__ import annotations

import pandas as pd


NUMERIC_FEATURES = [
    "age_at_diagnosis",
    "tumour_size_mm",
    "histological_grade",
    "positive_lymph_nodes",
]
CATEGORICAL_FEATURES = [
    "er_status",
    "pr_status",
    "her2_status",
    "histological_subtype",
]
ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_patient_input(patient: pd.DataFrame, cohort: pd.DataFrame) -> None:
    """Raise ValueError when an input row cannot be used safely by the demo."""
    if len(patient) != 1:
        raise ValueError("Exactly one patient record is required.")

    missing_columns = sorted(set(ALL_INPUT_FEATURES) - set(patient.columns))
    if missing_columns:
        raise ValueError(f"Required input fields are missing: {missing_columns}")

    if patient[ALL_INPUT_FEATURES].isna().any().any():
        raise ValueError("All eight baseline input fields must contain a value.")

    for feature in NUMERIC_FEATURES:
        value = pd.to_numeric(patient.loc[patient.index[0], feature], errors="coerce")
        reference = pd.to_numeric(cohort[feature], errors="coerce").dropna()
        if pd.isna(value):
            raise ValueError(f"{feature} must be numeric.")
        if value < reference.min() or value > reference.max():
            raise ValueError(
                f"{feature}={value:g} is outside the development-cohort range "
                f"[{reference.min():g}, {reference.max():g}]."
            )

    for feature in CATEGORICAL_FEATURES:
        value = str(patient.loc[patient.index[0], feature]).strip()
        allowed = set(cohort[feature].dropna().astype(str).str.strip())
        if value not in allowed:
            raise ValueError(f"{feature} contains an unsupported category: {value!r}.")

