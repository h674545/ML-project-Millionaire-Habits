import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Millionær-estimator")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

pipeline = load_model()

st.title("Millionær-estimator")
st.write("Fyll inn opplysningene under for å se om profilen din ligner en millionær.")

age = st.number_input("Alder", min_value=20, max_value=85, value=35, step=1)
annual_income = st.number_input("Årsinntekt (USD)", min_value=1, value=60_000, step=1_000)
monthly_savings = st.number_input("Månedlig sparing (USD)", min_value=0, value=1_000, step=100)

monthly_income = annual_income / 12

if monthly_savings > monthly_income:
    st.error(
        f"Månedlig sparing ({monthly_savings:,.0f} USD) er høyere enn månedlig inntekt "
        f"({monthly_income:,.0f} USD). Sjekk at tallene stemmer."
    )
    st.stop()

savings_rate = monthly_savings / monthly_income

if savings_rate > 0.74:
    st.warning(
        f"Spareraten ({savings_rate:.1%}) er utenfor det modellen er trent på (maks ~74 %). "
        "Resultatet kan være unøyaktig."
    )

X_input = pd.DataFrame(
    [[age, annual_income, savings_rate]],
    columns=["age", "total_annual_income", "savings_rate"],
)
prob = pipeline.predict_proba(X_input)[0][1]

st.subheader("Resultat")
st.write(f"Sannsynlighet for millionær-profil: **{prob:.1%}**")
st.progress(float(prob))

st.subheader("Hva om du sparte mer?")
max_slider = max(int(monthly_income * 0.90), monthly_savings + 500)
what_if = st.slider("Månedlig sparing (USD)", 0, max_slider, monthly_savings, step=100)

what_if_rate = what_if / monthly_income
X_whatif = pd.DataFrame(
    [[age, annual_income, what_if_rate]],
    columns=["age", "total_annual_income", "savings_rate"],
)
prob_whatif = pipeline.predict_proba(X_whatif)[0][1]
st.write(f"Med {what_if:,} USD i månedlig sparing: **{prob_whatif:.1%}**")

st.caption("Modellen er trent på syntetiske data og er ikke finansiell rådgivning.")
