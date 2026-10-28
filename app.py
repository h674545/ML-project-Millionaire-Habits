import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Millionær-estimator", page_icon="💰")

# --- Last modell ---
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

pipeline = load_model()

# --- Tittel ---
st.title("💰 Millionær-profil Estimator")
st.caption("Maskinlæringsmodell trent på syntetiske data fra Kaggle.")

# --- Inputfelter ---
st.header("Fyll inn dine opplysninger")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Alder", min_value=20, max_value=85, value=35, step=1)
with col2:
    annual_income = st.number_input(
        "Årsinntekt (USD)", min_value=1, value=60_000, step=1_000
    )

monthly_savings = st.number_input(
    "Månedlig sparing (USD)", min_value=0, value=1_000, step=100
)

# --- Validering ---
errors = []
if age < 20 or age > 85:
    errors.append("Alder må være mellom 20 og 85.")
if annual_income <= 0:
    errors.append("Inntekt må være større enn 0.")

if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

# --- Beregn sparerate og prediker ---
monthly_income = annual_income / 12
savings_rate = monthly_savings / monthly_income

if monthly_savings > monthly_income:
    st.error(
        f"⚠️ Månedlig sparing ({monthly_savings:,.0f} USD) overstiger månedlig inntekt "
        f"({monthly_income:,.0f} USD). Sjekk at du bruker USD og at tallene stemmer."
    )
    st.stop()

if savings_rate > 0.74:
    st.warning(
        f"⚠️ Spareraten din ({savings_rate:.1%}) er høyere enn det modellen ble trent på "
        f"(maks ~74 %). Estimatet kan være unøyaktig."
    )

X_input = pd.DataFrame(
    [[age, annual_income, savings_rate]],
    columns=["age", "total_annual_income", "savings_rate"],
)
prob = pipeline.predict_proba(X_input)[0][1]

# --- Vis resultat ---
st.header("Resultat")

if prob >= 0.6:
    st.success(f"Sannsynlighet: **{prob:.1%}** – profilen din ligner en millionær")
elif prob >= 0.4:
    st.warning(f"Sannsynlighet: **{prob:.1%}** – profilen din er nær millionær-nivå")
else:
    st.error(f"Sannsynlighet: **{prob:.1%}** – profilen din ligner ikke en millionær ennå")

st.progress(float(prob))
st.caption(f"Din nåværende sparerate: {savings_rate:.1%}")

# --- Hva om-glidebryter ---
st.header("Hva om du endret sparingen?")
st.write("Dra glideren for å utforske hvordan spareraten påvirker estimatet:")

max_slider = max(int(monthly_income * 0.90), monthly_savings + 500)
what_if = st.slider(
    "Månedlig sparing (USD)",
    min_value=0,
    max_value=max_slider,
    value=monthly_savings,
    step=100,
)

what_if_rate = what_if / monthly_income
X_whatif = pd.DataFrame(
    [[age, annual_income, what_if_rate]],
    columns=["age", "total_annual_income", "savings_rate"],
)
prob_whatif = pipeline.predict_proba(X_whatif)[0][1]
delta = prob_whatif - prob

st.metric(
    label="Estimert sannsynlighet",
    value=f"{prob_whatif:.1%}",
    delta=f"{delta:+.1%} sammenlignet med ditt opprinnelige estimat",
)

# --- Graf: sannsynlighet over hele sparerate-spekteret ---
st.subheader("Sannsynlighet for ulike sparerater")
rates = np.linspace(0, 0.74, 100)
X_range = pd.DataFrame({
    "age": age,
    "total_annual_income": annual_income,
    "savings_rate": rates,
})
probs = pipeline.predict_proba(X_range)[:, 1]

chart_df = pd.DataFrame({
    "Sparerate (%)": (rates * 100).round(1),
    "Sannsynlighet (%)": (probs * 100).round(1),
}).set_index("Sparerate (%)")

st.line_chart(chart_df)
st.caption(
    "Grafen viser hvordan sannsynligheten endrer seg med spareraten, "
    "gitt din alder og inntekt."
)

# --- Disclaimer ---
st.divider()
st.info(
    "⚠️ Denne modellen er trent på et **syntetisk** datasett og reflekterer "
    "ikke nødvendigvis faktiske mønstre. Resultatet er **ikke finansiell rådgivning**."
)
