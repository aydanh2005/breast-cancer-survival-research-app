"""Professional Streamlit demonstration for the METABRIC thesis project.

The application presents three full-cohort demonstration models, validation
results produced by the scientific workflow, input-range checks, and global
SHAP findings. It is an academic artifact and is not intended for clinical use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_repeated_validation import (  # noqa: E402
    ALL_FEATURES,
    EVALUATION_TIME,
    create_npi_frame,
    survival_probability_at_time,
)
from app_validation import validate_patient_input  # noqa: E402


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metabric_analysis_cohort.csv"
)
MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "streamlit_deployment_models.joblib"
)
METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "streamlit_deployment_metadata.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SHAP_FILE = OUTPUT_DIR / "rsf_shap_global_importance.csv"
BOOTSTRAP_FILE = (
    OUTPUT_DIR
    / "repeated_validation_cindex_bootstrap_ci.csv"
)
DIFFERENCE_FILE = (
    OUTPUT_DIR
    / "repeated_validation_paired_cindex_differences.csv"
)
CALIBRATION_FILE = (
    OUTPUT_DIR
    / "repeated_validation_calibration_summary.csv"
)


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

DISPLAY_NAMES = {
    "age_at_diagnosis": "Age at diagnosis",
    "tumour_size_mm": "Tumour size",
    "histological_grade": "Histological grade",
    "positive_lymph_nodes": "Positive lymph nodes",
    "er_status": "ER status",
    "pr_status": "PR status",
    "her2_status": "HER2 status",
    "histological_subtype": "Histological subtype",
}

DISPLAY_UNITS = {
    "age_at_diagnosis": "years",
    "tumour_size_mm": "mm",
    "histological_grade": "",
    "positive_lymph_nodes": "",
}

MODEL_NAMES = {
    "npi_cox": "NPI-based Cox",
    "multivariable_cox": "Multivariable Cox",
    "random_survival_forest": "Random Survival Forest",
}


st.set_page_config(
    page_title="Breast Cancer Survival Research Demonstrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        padding: 1rem;
        border-radius: 0.8rem;
    }

    [data-testid="stMetricValue"] > div {
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.15;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, #17324d, #365f7d);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        line-height: 1.2;
    }

    .hero p {
        margin: 0;
        opacity: 0.9;
    }

    .small-note {
        color: #5e6b78;
        font-size: 0.88rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.1rem 1.2rem;
        }

        .hero h1 {
            font-size: 1.55rem;
            line-height: 1.2;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> dict[str, object]:
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_supporting_data() -> tuple[pd.DataFrame, dict[str, object]]:
    cohort = pd.read_csv(DATA_FILE)

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return cohort, metadata


@st.cache_data
def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


def numeric_summary(
    cohort: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    values = pd.to_numeric(
        cohort[feature],
        errors="coerce",
    ).dropna()

    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
    }


def categorical_options(
    cohort: pd.DataFrame,
    feature: str,
) -> list[str]:
    values = (
        cohort[feature]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        value
        for value in values.unique()
        if value
    )


def predict_event_risk(
    models: dict[str, object],
    patient: pd.DataFrame,
) -> dict[str, float]:
    npi_patient = create_npi_frame(patient)

    survival = {
        "npi_cox": survival_probability_at_time(
            models["npi_cox"],
            npi_patient,
            EVALUATION_TIME,
        )[0],
        "multivariable_cox": survival_probability_at_time(
            models["multivariable_cox"],
            patient[ALL_FEATURES],
            EVALUATION_TIME,
        )[0],
        "random_survival_forest": survival_probability_at_time(
            models["random_survival_forest"],
            patient[ALL_FEATURES],
            EVALUATION_TIME,
        )[0],
    }

    return {
        key: float(np.clip(1.0 - value, 0.0, 1.0))
        for key, value in survival.items()
    }


def descriptive_risk_group(
    risk: float,
    metadata: dict[str, object],
) -> str:
    thresholds = metadata.get(
        "oof_rsf_descriptive_thresholds"
    )

    if not isinstance(thresholds, dict):
        return "Not available"

    lower = float(thresholds["lower_quartile"])
    upper = float(thresholds["upper_quartile"])

    if risk <= lower:
        return "Lower range"

    if risk >= upper:
        return "Higher range"

    return "Intermediate range"


def range_warnings(
    patient: pd.DataFrame,
    cohort: pd.DataFrame,
) -> list[str]:
    warnings = []

    for feature in NUMERIC_FEATURES:
        summary = numeric_summary(cohort, feature)
        value = float(patient.loc[0, feature])

        if value < summary["p05"] or value > summary["p95"]:
            unit = DISPLAY_UNITS[feature]
            interval = (
                f"{summary['p05']:.1f}–"
                f"{summary['p95']:.1f} {unit}"
            ).strip()

            warnings.append(
                f"{DISPLAY_NAMES[feature]} ({value:g}) lies "
                f"outside the central 90% of the development "
                f"cohort ({interval})."
            )

    return warnings


def patient_summary(
    patient: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        value = patient.loc[0, feature]
        unit = DISPLAY_UNITS.get(feature, "")

        if isinstance(value, (int, float, np.number)):
            shown = f"{value:g} {unit}".strip()
        else:
            shown = str(value)

        rows.append(
            {
                "Baseline variable": DISPLAY_NAMES[feature],
                "Entered value": shown,
            }
        )

    return pd.DataFrame(rows)


def report_text(
    patient: pd.DataFrame,
    risks: dict[str, float],
    group: str,
) -> str:
    lines = [
        "METABRIC FIVE-YEAR OUTCOME RESEARCH DEMONSTRATION",
        "Academic demonstration only — not for clinical use.",
        "",
        "BASELINE INPUTS",
    ]

    for _, row in patient_summary(patient).iterrows():
        lines.append(
            f"{row['Baseline variable']}: "
            f"{row['Entered value']}"
        )

    lines.extend(
        [
            "",
            "MODEL OUTPUTS",
        ]
    )

    for key, value in risks.items():
        lines.append(
            f"{MODEL_NAMES[key]}: {value:.1%}"
        )

    lines.extend(
        [
            f"Descriptive RSF range: {group}",
            "",
            (
                "The category uses validation-derived quartiles "
                "and is not a clinical threshold."
            ),
            (
                "The estimates are not externally or "
                "prospectively validated."
            ),
        ]
    )

    return "\n".join(lines)


def render_prediction_tab(
    patient: pd.DataFrame | None,
    risks: dict[str, float] | None,
    cohort: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    if patient is None or risks is None:
        st.info(
            "Enter baseline information in the sidebar and "
            "select **Generate prediction**."
        )
        return

    rsf_risk = risks["random_survival_forest"]
    risk_group = descriptive_risk_group(
        rsf_risk,
        metadata,
    )

    left, middle, right = st.columns(3)

    left.metric(
        "RSF five-year event risk",
        f"{rsf_risk:.1%}",
    )

    middle.metric(
        "Predicted event-free probability",
        f"{1.0 - rsf_risk:.1%}",
    )

    right.metric(
        "RSF risk position",
        risk_group,
    )

    st.caption(
        "Risk position is descriptive only and is defined by "
        "quartiles of repeated out-of-fold RSF predictions."
    )

    st.progress(
        int(round(rsf_risk * 100)),
        text=f"RSF predicted risk: {rsf_risk:.1%}",
    )

    st.caption(
        "The bar is a model output, not a clinical decision "
        "threshold. The descriptive range is based on quartiles "
        "of repeated out-of-fold RSF predictions."
    )

    warnings = range_warnings(
        patient,
        cohort,
    )

    if warnings:
        st.warning(
            "Input-range check:\n\n"
            + "\n\n".join(
                f"• {item}"
                for item in warnings
            )
        )
    else:
        st.success(
            "All numeric inputs fall within the central 90% "
            "of the development cohort."
        )

    st.subheader("Comparison of model estimates")

    comparison = pd.DataFrame(
        {
            "Model": [
                MODEL_NAMES[key]
                for key in risks
            ],
            "Predicted risk": [
                risks[key]
                for key in risks
            ],
        }
    )

    st.bar_chart(
        comparison.set_index("Model"),
        y="Predicted risk",
        horizontal=True,
    )

    st.dataframe(
        comparison.style.format(
            {
                "Predicted risk": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.info(
        "The three estimates use different modelling assumptions. "
        "Their differences do not show that any estimate is "
        "clinically correct for this individual."
    )

    with st.expander(
        "Entered baseline profile",
        expanded=False,
    ):
        st.dataframe(
            patient_summary(patient),
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Download demonstration summary",
        data=report_text(
            patient,
            risks,
            risk_group,
        ),
        file_name="research_prediction_summary.txt",
        mime="text/plain",
    )


def render_explanation_tab(
    patient: pd.DataFrame | None,
) -> None:
    st.subheader(
        "Validation-based global explanation"
    )

    shap_results = read_optional_csv(SHAP_FILE)

    required_columns = {
        "feature",
        "mean_absolute_shap",
    }

    if (
        shap_results is None
        or not required_columns.issubset(
            shap_results.columns
        )
    ):
        st.info(
            "Run `explain_rsf_with_shap.py` to generate "
            "the global explanation file."
        )
        return

    plot_data = (
        shap_results[
            [
                "feature",
                "mean_absolute_shap",
            ]
        ]
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
        .set_index("feature")
    )

    st.bar_chart(
        plot_data,
        horizontal=True,
    )

    st.caption(
        "Mean absolute SHAP values describe average feature "
        "influence across validation refits. They are model "
        "attributions, not causal effects."
    )

    if patient is not None:
        st.subheader("Current input profile")

        st.dataframe(
            patient_summary(patient),
            hide_index=True,
            width="stretch",
        )

        st.info(
            "This profile is shown beside the global explanation "
            "for context. It is not presented as a patient-specific "
            "SHAP explanation."
        )


def render_evaluation_tab() -> None:
    st.subheader("Scientific evaluation outputs")

    bootstrap = read_optional_csv(BOOTSTRAP_FILE)
    differences = read_optional_csv(DIFFERENCE_FILE)
    calibration = read_optional_csv(CALIBRATION_FILE)

    if bootstrap is not None:
        st.markdown(
            "#### C-index with patient-level bootstrap intervals"
        )

        formatted = bootstrap.copy()

        numeric = [
            "c_index",
            "ci_95_lower",
            "ci_95_upper",
        ]

        available_numeric = [
            column
            for column in numeric
            if column in formatted.columns
        ]

        st.dataframe(
            formatted.style.format(
                {
                    column: "{:.3f}"
                    for column in available_numeric
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption(
            "C-index bootstrap summary file is not available "
            "in `outputs`."
        )

    if differences is not None:
        st.markdown(
            "#### Paired model differences"
        )

        st.dataframe(
            differences,
            hide_index=True,
            width="stretch",
        )

    if calibration is not None:
        st.markdown(
            "#### Five-year calibration from repeated validation"
        )

        st.dataframe(
            calibration.style.format(
                precision=4
            ),
            hide_index=True,
            width="stretch",
        )

    if (
        bootstrap is None
        and differences is None
        and calibration is None
    ):
        st.info(
            "Run the evaluation scripts to populate this tab "
            "with saved results."
        )

    st.caption(
        "Evaluation metrics are loaded from the reproducible "
        "analysis outputs; they are not calculated from the "
        "full-cohort demonstration fit displayed in the app."
    )


def render_method_tab(
    metadata: dict[str, object],
) -> None:
    st.subheader(
        "Purpose and analytical design"
    )

    st.write(
        "This artifact demonstrates five-year relapse-free "
        "outcome prediction from eight baseline "
        "clinicopathological variables. It compares an "
        "NPI-based Cox model, a multivariable Cox model, "
        "and a Random Survival Forest."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Demonstration training cohort",
        f"{metadata.get('training_patients', 'N/A')}",
    )

    col2.metric(
        "Prediction horizon",
        f"{EVALUATION_TIME:g} months",
    )

    col3.metric(
        "Models compared",
        "3",
    )

    st.markdown("#### Safeguards")

    st.markdown(
        """
        - Treatment, follow-up, outcome, and molecular-cluster
          variables are excluded from inputs.
        - Scientific performance is estimated from out-of-fold
          validation, not the full-cohort fit.
        - Global SHAP results are treated as model attribution
          rather than causal evidence.
        - Input-range checks highlight values with relatively
          limited cohort support.
        """
    )

    st.markdown("#### Limitations")

    st.markdown(
        """
        - Retrospective METABRIC data may not represent current
          or external populations.
        - The models have not undergone independent external or
          prospective validation.
        - The prototype does not estimate treatment effects and
          does not support clinical decisions.
        - Predictions may be uncertain for uncommon input
          combinations even when individual values are valid.
        """
    )

    st.markdown("#### Online deployment model")

    st.info(
        "To keep the public application lightweight, its Random "
        "Survival Forest is a compact full-cohort demonstration "
        "fit. It is not used to calculate or support the thesis "
        "results. All scientific conclusions shown under Model "
        "evaluation come from the pre-specified repeated "
        "out-of-fold validation workflow."
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>
            Five-Year Relapse-Free Outcome Research Demonstrator
          </h1>
          <p>
            Explainable survival modelling using retrospective
            METABRIC clinical data
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Academic demonstration only. Not clinically validated "
        "and not for diagnosis, treatment, referral, or "
        "follow-up decisions."
    )

    st.caption(
        "Online deployment uses a compact demonstration fit; "
        "scientific results are loaded separately from the "
        "repeated validation analysis."
    )

    required = [
        DATA_FILE,
        MODEL_FILE,
        METADATA_FILE,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        st.error(
            "Required files are missing: "
            + ", ".join(
                str(
                    path.relative_to(PROJECT_ROOT)
                )
                for path in missing
            )
        )
        st.stop()

    models = load_models()
    cohort, metadata = load_supporting_data()

    st.sidebar.header(
        "Baseline clinical inputs"
    )

    st.sidebar.caption(
        "Treatment and outcome information are "
        "intentionally excluded."
    )

    age_stats = numeric_summary(
        cohort,
        "age_at_diagnosis",
    )

    size_stats = numeric_summary(
        cohort,
        "tumour_size_mm",
    )

    node_stats = numeric_summary(
        cohort,
        "positive_lymph_nodes",
    )

    grade_options = sorted(
        int(value)
        for value in pd.to_numeric(
            cohort["histological_grade"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    with st.sidebar.form("prediction_form"):
        age = st.number_input(
            "Age at diagnosis (years)",
            min_value=age_stats["minimum"],
            max_value=age_stats["maximum"],
            value=age_stats["median"],
            step=1.0,
        )

        tumour_size = st.number_input(
            "Tumour size (mm)",
            min_value=size_stats["minimum"],
            max_value=size_stats["maximum"],
            value=size_stats["median"],
            step=1.0,
        )

        grade = st.selectbox(
            "Histological grade",
            grade_options,
        )

        positive_nodes = st.number_input(
            "Positive lymph nodes",
            min_value=int(node_stats["minimum"]),
            max_value=int(node_stats["maximum"]),
            value=int(node_stats["median"]),
            step=1,
        )

        er_status = st.selectbox(
            "ER status",
            categorical_options(
                cohort,
                "er_status",
            ),
        )

        pr_status = st.selectbox(
            "PR status",
            categorical_options(
                cohort,
                "pr_status",
            ),
        )

        her2_status = st.selectbox(
            "HER2 status",
            categorical_options(
                cohort,
                "her2_status",
            ),
        )

        subtype = st.selectbox(
            "Histological subtype",
            categorical_options(
                cohort,
                "histological_subtype",
            ),
        )

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            width="stretch",
        )

    if submitted:
        candidate_patient = pd.DataFrame(
            [
                {
                    "age_at_diagnosis": age,
                    "tumour_size_mm": tumour_size,
                    "histological_grade": grade,
                    "positive_lymph_nodes": positive_nodes,
                    "er_status": er_status,
                    "pr_status": pr_status,
                    "her2_status": her2_status,
                    "histological_subtype": subtype,
                }
            ]
        )

        try:
            validate_patient_input(
                candidate_patient,
                cohort,
            )

        except ValueError as error:
            st.error(
                "The entered profile cannot be evaluated: "
                f"{error}"
            )

            st.session_state.pop(
                "patient",
                None,
            )
            st.session_state.pop(
                "risks",
                None,
            )

        else:
            st.session_state["patient"] = (
                candidate_patient
            )

            st.session_state["risks"] = (
                predict_event_risk(
                    models,
                    candidate_patient,
                )
            )

    patient = st.session_state.get("patient")
    risks = st.session_state.get("risks")

    (
        prediction_tab,
        explanation_tab,
        evaluation_tab,
        method_tab,
    ) = st.tabs(
        [
            "Prediction",
            "Explanation",
            "Model evaluation",
            "Method & limitations",
        ]
    )

    with prediction_tab:
        render_prediction_tab(
            patient,
            risks,
            cohort,
            metadata,
        )

    with explanation_tab:
        render_explanation_tab(patient)

    with evaluation_tab:
        render_evaluation_tab()

    with method_tab:
        render_method_tab(metadata)


if __name__ == "__main__":
    main()