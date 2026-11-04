"""Repeated five-fold cross-validation for the METABRIC survival models.

Three models are evaluated on identical validation folds:

1. NPI-based Cox model
2. Multivariable Cox regression
3. Random Survival Forest

Five-fold cross-validation is repeated three times using pre-specified random
seeds. All imputation, scaling, and categorical encoding are learned only from
the corresponding training fold.

The script saves:
- patient-level out-of-fold predictions;
- fold-level performance measures;
- an overall repeated-validation summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metabric_analysis_cohort.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

N_SPLITS = 5
REPETITION_SEEDS = [2026, 2027, 2028]
EVALUATION_TIME = 59.9


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

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


MODEL_LABELS = {
    "npi_cox": "NPI-based Cox",
    "multivariable_cox": "Multivariable Cox",
    "random_survival_forest": "Random Survival Forest",
}


def make_survival_target(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Create the structured outcome required by scikit-survival."""

    return Surv.from_arrays(
        event=frame["event"].astype(bool).to_numpy(),
        time=frame["time_months"].astype(float).to_numpy(),
    )


def lymph_node_stage(
    nodes: pd.Series,
) -> pd.Series:
    """Create the NPI lymph-node stage."""

    values = pd.to_numeric(
        nodes,
        errors="coerce",
    )

    return pd.Series(
        np.select(
            [
                values.eq(0),
                values.between(
                    1,
                    3,
                    inclusive="both",
                ),
                values.ge(4),
            ],
            [
                1.0,
                2.0,
                3.0,
            ],
            default=np.nan,
        ),
        index=nodes.index,
        dtype="float64",
    )


def create_npi_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate NPI using tumour size in centimetres."""

    tumour_mm = pd.to_numeric(
        frame["tumour_size_mm"],
        errors="coerce",
    )

    grade = pd.to_numeric(
        frame["histological_grade"],
        errors="coerce",
    )

    node_group = lymph_node_stage(
        frame["positive_lymph_nodes"]
    )

    calculated_npi = (
        0.2 * (tumour_mm / 10.0)
        + grade
        + node_group
    )

    return pd.DataFrame(
        {
            "calculated_npi": calculated_npi,
        },
        index=frame.index,
    )


def make_npi_pipeline() -> Pipeline:
    """Create the NPI-based Cox reference model."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                CoxPHSurvivalAnalysis(
                    alpha=0.01,
                ),
            ),
        ]
    )


def make_multivariable_preprocessor() -> ColumnTransformer:
    """Create fold-specific preprocessing."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_cox_pipeline() -> Pipeline:
    """Create the multivariable Cox model."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_multivariable_preprocessor(),
            ),
            (
                "model",
                CoxPHSurvivalAnalysis(
                    alpha=0.01,
                ),
            ),
        ]
    )


def make_rsf_pipeline(
    random_seed: int,
) -> Pipeline:
    """Create the pre-specified Random Survival Forest."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_multivariable_preprocessor(),
            ),
            (
                "model",
                RandomSurvivalForest(
                    n_estimators=300,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def survival_probability_at_time(
    model: Pipeline,
    features: pd.DataFrame,
    evaluation_time: float,
) -> np.ndarray:
    """Predict survival probability at the selected time."""

    survival_functions = (
        model.predict_survival_function(
            features
        )
    )

    probabilities = np.array(
        [
            function(evaluation_time)
            for function in survival_functions
        ],
        dtype=float,
    )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


def evaluate_fold(
    model_key: str,
    repetition: int,
    fold: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    survival_probabilities: np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate censoring-aware measures for one validation fold."""

    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores,
    )[0]

    auc_values, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        risk_scores,
        np.array(
            [EVALUATION_TIME]
        ),
    )

    _, brier_values = brier_score(
        y_train,
        y_test,
        survival_probabilities.reshape(
            -1,
            1,
        ),
        np.array(
            [EVALUATION_TIME]
        ),
    )

    return {
        "model_key": model_key,
        "model": MODEL_LABELS[model_key],
        "repetition": repetition,
        "fold": fold,
        "training_n": len(y_train),
        "validation_n": len(y_test),
        "validation_events": int(
            y_test["event"].sum()
        ),
        "evaluation_time_months": (
            EVALUATION_TIME
        ),
        "c_index": float(c_index),
        "dynamic_auc": float(
            auc_values[0]
        ),
        "mean_dynamic_auc": float(
            mean_auc
        ),
        "ipcw_brier_score": float(
            brier_values[0]
        ),
    }


def fit_and_predict_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    y_train: np.ndarray,
    rsf_seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit all three models and predict on one validation fold."""

    results: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    # Model 1: NPI-based Cox reference
    train_npi = create_npi_frame(
        train_data
    )
    test_npi = create_npi_frame(
        test_data
    )

    npi_model = make_npi_pipeline()
    npi_model.fit(
        train_npi,
        y_train,
    )

    results["npi_cox"] = {
        "risk": npi_model.predict(
            test_npi
        ),
        "survival": (
            survival_probability_at_time(
                npi_model,
                test_npi,
                EVALUATION_TIME,
            )
        ),
    }

    # Model 2: Multivariable Cox regression
    cox_model = make_cox_pipeline()

    cox_model.fit(
        train_data[ALL_FEATURES],
        y_train,
    )

    results["multivariable_cox"] = {
        "risk": cox_model.predict(
            test_data[ALL_FEATURES]
        ),
        "survival": (
            survival_probability_at_time(
                cox_model,
                test_data[ALL_FEATURES],
                EVALUATION_TIME,
            )
        ),
    }

    # Model 3: Random Survival Forest
    rsf_model = make_rsf_pipeline(
        random_seed=rsf_seed
    )

    rsf_model.fit(
        train_data[ALL_FEATURES],
        y_train,
    )

    results[
        "random_survival_forest"
    ] = {
        "risk": rsf_model.predict(
            test_data[ALL_FEATURES]
        ),
        "survival": (
            survival_probability_at_time(
                rsf_model,
                test_data[ALL_FEATURES],
                EVALUATION_TIME,
            )
        ),
    }

    return results


def validate_dataset(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the prepared analysis cohort."""

    required_columns = {
        "patient_id",
        "time_months",
        "event",
        *ALL_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    result = data.copy()

    result["time_months"] = pd.to_numeric(
        result["time_months"],
        errors="raise",
    )

    result["event"] = pd.to_numeric(
        result["event"],
        errors="raise",
    ).astype(int)

    if result[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate patient identifiers "
            "were found."
        )

    if not result["event"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "The event column must contain "
            "only 0 and 1."
        )

    if result[
        "time_months"
    ].isna().any():
        raise ValueError(
            "Missing survival times were found."
        )

    if (
        result["time_months"] <= 0
    ).any():
        raise ValueError(
            "Survival times must be greater "
            "than zero."
        )

    return result


def create_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise performance across the 15 validation folds."""

    measure_columns = [
        "c_index",
        "dynamic_auc",
        "ipcw_brier_score",
    ]

    summary = (
        fold_metrics
        .groupby(
            [
                "model_key",
                "model",
            ],
            as_index=False,
        )[measure_columns]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    summary.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        ).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    return summary


def main() -> None:
    """Run repeated cross-validation and save all results."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        DATA_FILE
    )

    data = validate_dataset(
        data
    )

    y = make_survival_target(
        data
    )

    prediction_rows: list[
        dict[str, object]
    ] = []

    metric_rows: list[
        dict[str, object]
    ] = []

    total_model_fits = (
        len(REPETITION_SEEDS)
        * N_SPLITS
        * len(MODEL_LABELS)
    )

    completed_model_fits = 0

    print(
        "REPEATED FIVE-FOLD SURVIVAL VALIDATION"
    )
    print("=" * 70)
    print(
        f"Patients: {len(data)}"
    )
    print(
        f"Events: {int(data['event'].sum())}"
    )
    print(
        f"Repetitions: "
        f"{len(REPETITION_SEEDS)}"
    )
    print(
        f"Folds per repetition: {N_SPLITS}"
    )
    print(
        f"Total model fits: "
        f"{total_model_fits}"
    )
    print(
        f"Evaluation time: "
        f"{EVALUATION_TIME} months"
    )
    print()

    for repetition, split_seed in enumerate(
        REPETITION_SEEDS,
        start=1,
    ):
        print(
            f"Starting repetition "
            f"{repetition}/"
            f"{len(REPETITION_SEEDS)} "
            f"with split seed "
            f"{split_seed}."
        )

        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )

        for fold, (
            train_index,
            test_index,
        ) in enumerate(
            splitter.split(
                data,
                data["event"],
            ),
            start=1,
        ):
            print(
                f"  Repetition {repetition}, "
                f"fold {fold}/{N_SPLITS}"
            )

            train_data = (
                data
                .iloc[train_index]
                .copy()
            )

            test_data = (
                data
                .iloc[test_index]
                .copy()
            )

            y_train = y[train_index]
            y_test = y[test_index]

            rsf_seed = (
                split_seed * 100
                + fold
            )

            fold_predictions = (
                fit_and_predict_models(
                    train_data=train_data,
                    test_data=test_data,
                    y_train=y_train,
                    rsf_seed=rsf_seed,
                )
            )

            for (
                model_key,
                model_predictions,
            ) in fold_predictions.items():
                risk_scores = (
                    model_predictions["risk"]
                )

                survival_probabilities = (
                    model_predictions[
                        "survival"
                    ]
                )

                if (
                    np.isnan(
                        risk_scores
                    ).any()
                    or np.isnan(
                        survival_probabilities
                    ).any()
                ):
                    raise RuntimeError(
                        "Missing predictions found "
                        f"for {model_key}, "
                        f"repetition {repetition}, "
                        f"fold {fold}."
                    )

                metric_rows.append(
                    evaluate_fold(
                        model_key=model_key,
                        repetition=repetition,
                        fold=fold,
                        y_train=y_train,
                        y_test=y_test,
                        risk_scores=risk_scores,
                        survival_probabilities=(
                            survival_probabilities
                        ),
                    )
                )

                for (
                    position,
                    patient_index,
                ) in enumerate(
                    test_index
                ):
                    prediction_rows.append(
                        {
                            "patient_id": (
                                data.iloc[
                                    patient_index
                                ]["patient_id"]
                            ),
                            "repetition": (
                                repetition
                            ),
                            "split_seed": (
                                split_seed
                            ),
                            "fold": fold,
                            "model_key": (
                                model_key
                            ),
                            "model": (
                                MODEL_LABELS[
                                    model_key
                                ]
                            ),
                            "time_months": (
                                float(
                                    data.iloc[
                                        patient_index
                                    ][
                                        "time_months"
                                    ]
                                )
                            ),
                            "event": (
                                int(
                                    data.iloc[
                                        patient_index
                                    ]["event"]
                                )
                            ),
                            "risk_score": (
                                float(
                                    risk_scores[
                                        position
                                    ]
                                )
                            ),
                            (
                                "survival_"
                                "probability_59_9m"
                            ): (
                                float(
                                    survival_probabilities[
                                        position
                                    ]
                                )
                            ),
                            (
                                "event_risk_59_9m"
                            ): (
                                float(
                                    1.0
                                    - survival_probabilities[
                                        position
                                    ]
                                )
                            ),
                        }
                    )

                completed_model_fits += 1

                print(
                    f"    Completed "
                    f"{MODEL_LABELS[model_key]} "
                    f"({completed_model_fits}/"
                    f"{total_model_fits})"
                )

        print(
            f"Repetition {repetition} "
            "completed.\n"
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    expected_prediction_rows = (
        len(data)
        * len(REPETITION_SEEDS)
        * len(MODEL_LABELS)
    )

    if len(predictions) != (
        expected_prediction_rows
    ):
        raise RuntimeError(
            "Unexpected number of prediction "
            "rows. Expected "
            f"{expected_prediction_rows}, "
            f"found {len(predictions)}."
        )

    summary = create_summary(
        fold_metrics
    )

    prediction_file = (
        OUTPUT_DIR
        / "repeated_validation_predictions.csv"
    )

    fold_metrics_file = (
        OUTPUT_DIR
        / "repeated_validation_fold_metrics.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "repeated_validation_summary.csv"
    )

    predictions.to_csv(
        prediction_file,
        index=False,
    )

    fold_metrics.to_csv(
        fold_metrics_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(
        "\nREPEATED-VALIDATION SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("\nSaved:")
    print(prediction_file)
    print(fold_metrics_file)
    print(summary_file)

    print(
        "\nRepeated validation completed "
        "successfully."
    )


if __name__ == "__main__":
    main()