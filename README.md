# Millionær-profil Estimator

Streamlit-app som predikerer om en person har en millionær-profil, basert på maskinlæring.  
Laget som universitetsprosjekt. Modellen er trent på et syntetisk datasett fra Kaggle.

## Forutsetninger

- Python 3.9+
- Datasettet `Millionaire_Habits_Dataset.csv` fra Kaggle, lagt i mappen `archive/`

## Installasjon

```bash
pip install -r requirements.txt
```

## Bruk

**Steg 1 – tren modellen** (gjøres én gang, lagrer `model.joblib`):
```bash
python3 train.py
```

Forventet output: ROC-AUC rundt 0,88 på testsettet.

**Steg 2 – start appen:**
```bash
python3 -m streamlit run app.py
```

Åpne nettleseren på `http://localhost:8501`.

## Features modellen bruker

| Feature | Beskrivelse |
|---|---|
| `age` | Alder (20–85) |
| `total_annual_income` | Årsinntekt i USD |
| `savings_rate` | Månedlig sparing / (årsinntekt / 12) |

## Viktig

Modellen er trent på **syntetiske data** og er ikke finansiell rådgivning.
