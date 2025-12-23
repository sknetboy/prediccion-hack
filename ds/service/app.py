from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd

class ClientInput(BaseModel):
    plan_type: str = Field(...)
    tenure_months: int = Field(...)
    monthly_charges: float = Field(...)
    late_payments: int = Field(...)
    app_logins: int = Field(...)

app = FastAPI()
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model.joblib')
if not os.path.exists(model_path):
    raise RuntimeError('Model not found. Run training.')

pipe = joblib.load(model_path)
stats = {'total_evaluados': 0, 'tasa_churn': 0.0, 'churn_count': 0}

@app.post('/predict')
def predict(payload: ClientInput):
    df = pd.DataFrame([payload.dict()])
    proba = float(pipe.predict_proba(df)[:, 1][0])
    pred = 'Va a cancelar' if proba >= 0.5 else 'Va a continuar'
    stats['total_evaluados'] += 1
    if pred == 'Va a cancelar':
        stats['churn_count'] += 1
    stats['tasa_churn'] = stats['churn_count'] / stats['total_evaluados'] if stats['total_evaluados'] else 0.0
    return {'prevision': pred, 'probabilidad': proba}

@app.get('/stats')
def get_stats():
    return {'total_evaluados': stats['total_evaluados'], 'tasa_churn': round(stats['tasa_churn'], 4)}