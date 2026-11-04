# Five-Year Relapse-Free Outcome Research Demonstrator

This repository contains the public Streamlit demonstration artifact developed
for Aydan Huseynli's Bachelor thesis using retrospective METABRIC clinical data.

## Important scope statement

The application is an academic demonstration only. It is not clinically
validated and must not be used for diagnosis, treatment, referral, or follow-up
decisions. The compact Random Survival Forest used by the online interface is a
deployment-only fit. Scientific conclusions are based on the separate repeated
out-of-fold validation results included in `outputs/`.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/train_deployment_models.py
python -m streamlit run app.py
```

The application will normally open at `http://localhost:8501`.

## Public deployment

After `models/streamlit_deployment_models.joblib` has been generated and checked
to be below GitHub's file-size limit, upload the complete folder to a GitHub
repository. In Streamlit Community Cloud, select that repository and set the
entry point to `app.py`.

## Reproducibility

- Python: 3.12
- Evaluation horizon: 59.9 months
- Deployment random seed: 2026
- Scientific results: repeated stratified 5-fold validation, three repetitions
- Performance uncertainty: paired patient-level bootstrap
