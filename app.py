import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Prediction ML Demo", page_icon="🩺", layout="centered")
bundle = joblib.load("models/disease_demo_model.joblib")
model = bundle["model"]
features = bundle["features"]

st.title("🩺 Disease Prediction from Medical Data")
st.caption("Educational ML demonstration using synthetic data — not a medical diagnosis.")

st.warning(
    "This app is not clinically validated. Predictions may be wrong and must not be used "
    "to diagnose, treat, or rule out a disease. Consult a qualified healthcare professional."
)

with st.form("symptoms"):
    st.subheader("Enter sample information")
    age = st.slider("Age", 1, 100, 25)
    temperature_c = st.slider("Body temperature (°C)", 35.0, 41.5, 37.0, 0.1)
    oxygen_saturation_pct = st.slider("Oxygen saturation (%)", 85, 100, 98)
    c1, c2 = st.columns(2)
    with c1:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
        fatigue = st.checkbox("Fatigue")
        headache = st.checkbox("Headache")
        body_pain = st.checkbox("Body pain")
    with c2:
        nausea = st.checkbox("Nausea")
        sore_throat = st.checkbox("Sore throat")
        breathlessness = st.checkbox("Breathlessness")
    submitted = st.form_submit_button("Run demo prediction")

if submitted:
    row = pd.DataFrame([{
        "age": age, "fever": int(fever), "cough": int(cough), "fatigue": int(fatigue),
        "headache": int(headache), "body_pain": int(body_pain), "nausea": int(nausea),
        "sore_throat": int(sore_throat), "breathlessness": int(breathlessness),
        "temperature_c": temperature_c, "oxygen_saturation_pct": oxygen_saturation_pct
    }])[features]
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    st.subheader("Model output (educational only)")
    st.write(f"Most likely synthetic training label: **{prediction}**")
    st.caption("These labels describe patterns in generated sample data, not confirmed diseases.")
    st.write("Model scores (not clinical probabilities):")
    for label, score in sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True):
        st.write(f"- {label}: {score * 100:.1f}%")
    st.error("Do not rely on this output for medical decisions. Seek professional care for health concerns.")

with st.expander("Project details"):
    st.write("Model: Random Forest classifier in a scikit-learn pipeline.")
    st.write(f"Test accuracy on held-out synthetic data: {bundle['accuracy'] * 100:.1f}%.")
    st.write("The accuracy is only a synthetic-data benchmark and does not indicate clinical performance.")
