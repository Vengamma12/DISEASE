# Disease Prediction from Medical Data Using Machine Learning

## About
A beginner-friendly educational ML project with a Streamlit interface. It uses a Random Forest classifier to map entered sample symptoms and measurements to one of four **synthetic demo labels**.

**Medical safety:** The included dataset is generated for demonstration, not collected from patients. The labels are illustrative symptom-pattern labels, not reliable diagnoses. This project is not clinically validated and must not be used to diagnose, treat, triage, or rule out disease.

## Files
- `app.py` — interactive web app
- `train_model.py` — retrain the model
- `data/synthetic_medical_data.csv` — generated sample dataset (1,800 rows)
- `models/disease_demo_model.joblib` — saved model bundle
- `requirements.txt` — dependencies

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open Command Prompt in the extracted `disease_prediction_ml_project` folder.
4. Install dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```
5. Run:
   ```bash
   py -m streamlit run app.py
   ```

## Retrain
```bash
py train_model.py
```

## Inputs
Age, fever, cough, fatigue, headache, body pain, nausea, sore throat, breathlessness, temperature, and oxygen saturation.

## Model and evaluation
Random Forest classifier using scikit-learn. Held-out accuracy on the generated dataset is 71.4%. This is not evidence of clinical accuracy or suitability for real medical use.
