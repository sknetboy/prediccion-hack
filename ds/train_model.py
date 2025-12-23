import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def load_or_generate():
    path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        n = 500
        rng = np.random.default_rng(42)
        plan = rng.choice(['basic', 'standard', 'premium'], n)
        tenure = rng.integers(1, 36, n)
        monthly = rng.normal(30, 10, n).clip(5, 200)
        late = rng.poisson(0.5, n)
        logins = rng.poisson(20, n)
        churn_prob = 0.2 + 0.4 * (late > 2) + 0.3 * (tenure < 6) + 0.2 * (monthly > 80)
        churn_prob = churn_prob.clip(0, 1)
        churn = rng.binomial(1, churn_prob)
        df = pd.DataFrame({
            'plan_type': plan,
            'tenure_months': tenure,
            'monthly_charges': monthly.round(2),
            'late_payments': late,
            'app_logins': logins,
            'churn': churn
        })
        os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
        df.to_csv(path, index=False)
    return df

df = load_or_generate()
X = df[['plan_type', 'tenure_months', 'monthly_charges', 'late_payments', 'app_logins']]
y = df['churn']

categorical = ['plan_type']
numeric = ['tenure_months', 'monthly_charges', 'late_payments', 'app_logins']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

model = LogisticRegression(max_iter=1000)
pipe = Pipeline([('pre', preprocess), ('clf', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

out_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(out_dir, exist_ok=True)
joblib.dump(pipe, os.path.join(out_dir, 'model.joblib'))

metrics = {
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1': float(f1),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test))
}
pd.Series(metrics).to_json(os.path.join(out_dir, 'metrics.json'))
print(metrics)