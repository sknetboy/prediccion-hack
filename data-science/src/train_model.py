import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.isotonic import IsotonicRegression
import json
import os

def load_or_generate():
    env_path = os.getenv('TRAIN_DATA_PATH') or os.getenv('DATASET_PATH')
    if env_path and os.path.exists(env_path):
        df = pd.read_csv(env_path)
        return df
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'shared', 'data')
    base = os.path.normpath(base)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, 'dataset.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        n = 600
        rng = np.random.default_rng(42)
        plan = rng.choice(['Basic', 'Standard', 'Premium'], n)
        tiempo = rng.integers(1, 36, n)
        retrasos = rng.poisson(0.7, n)
        uso = rng.normal(18, 6, n).clip(0.5, 100)
        churn_prob = 0.15 + 0.35 * (retrasos > 2) + 0.25 * (tiempo < 6) + 0.2 * (uso < 8)
        churn_prob = churn_prob.clip(0, 1)
        churn = rng.binomial(1, churn_prob)
        df = pd.DataFrame({
            'plan': plan,
            'tiempo_contrato_meses': tiempo,
            'retrasos_pago': retrasos,
            'uso_mensual': uso.round(2),
            'churn': churn
        })
        df.to_csv(path, index=False)
    return df

df = load_or_generate()
if not set(['nps','quejas','canal_contacto','interacciones_soporte','tipo_pago']).issubset(df.columns):
    rng = np.random.default_rng(7)
    df['nps'] = rng.integers(0, 11, len(df))
    df['quejas'] = rng.poisson(0.5, len(df))
    df['canal_contacto'] = rng.choice(['web','app','telefono','email','chat'], len(df))
    df['interacciones_soporte'] = rng.poisson(1.0, len(df))
    df['tipo_pago'] = rng.choice(['tarjeta','transferencia','efectivo','debito_automatico'], len(df))
if not set(['region','tipo_cliente']).issubset(df.columns):
    rng = np.random.default_rng(9)
    df['region'] = rng.choice(['norte','sur','este','oeste'], len(df))
    df['tipo_cliente'] = rng.choice(['nuevo','antiguo','corporativo'], len(df))

X = df[['plan', 'tiempo_contrato_meses', 'retrasos_pago', 'uso_mensual',
        'nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']]
y = df['churn']

categorical = ['plan','canal_contacto','tipo_pago','region','tipo_cliente']
numeric = ['tiempo_contrato_meses', 'retrasos_pago', 'uso_mensual', 'nps', 'quejas', 'interacciones_soporte']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

model = LogisticRegression(max_iter=1000)
pipe = Pipeline([('pre', preprocess), ('clf', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

calib_method = os.getenv('CALIBRATION_METHOD', 'isotonic')
calibrator = None
try:
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    pipe.fit(X_tr, y_tr)
    base_proba_cal = pipe.predict_proba(X_cal)[:, 1]
    if calib_method == 'sigmoid':
        from sklearn.linear_model import LogisticRegression as LR
        lr_cal = LR(max_iter=1000)
        lr_cal.fit(base_proba_cal.reshape(-1, 1), y_cal)
        calibrator = ('sigmoid', lr_cal)
    else:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(base_proba_cal, y_cal)
        calibrator = ('isotonic', iso)
    pipe.fit(X_train, y_train)
except Exception:
    calibrator = None

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data-science', 'models')
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
joblib.dump(pipe, os.path.join(out_dir, 'model.joblib'))

if calibrator is not None:
    method, cal = calibrator
    joblib.dump(cal, os.path.join(out_dir, f'calibrator__{method}.joblib'))
    proba_test = pipe.predict_proba(X_test)[:, 1]
    if method == 'sigmoid':
        cal_proba = cal.predict_proba(proba_test.reshape(-1, 1))[:, 1]
    else:
        cal_proba = cal.predict(proba_test)
    from sklearn.metrics import brier_score_loss
    brier = float(brier_score_loss(y_test, cal_proba))
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(cal_proba, bins) - 1
    ece = 0.0
    for b in range(10):
        m = (idx == b)
        if np.any(m):
            conf = float(np.mean(cal_proba[m]))
            freq = float(np.mean(y_test.values[m]))
            ece += (np.sum(m) / len(y_test)) * abs(conf - freq)
    calib_metrics = {'method': method, 'brier': brier, 'ece': float(ece)}
    pd.Series(calib_metrics).to_json(os.path.join(out_dir, 'calibration.json'))

metrics = {
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1': float(f1)
}
pd.Series(metrics).to_json(os.path.join(out_dir, 'metrics.json'))
print(metrics)

# Guardar baseline de entrenamiento para detección de drift
baseline = {'numeric': {}, 'categorical': {}}
for feat in numeric:
    s = pd.to_numeric(df[feat], errors='coerce')
    s = s.dropna()
    if len(s) == 0:
        baseline['numeric'][feat] = {'mean': 0.0, 'std': 0.0, 'bins': [], 'counts': []}
    else:
        b = np.linspace(float(s.min()), float(s.max()), 11)
        cnt, _ = np.histogram(s, bins=b)
        baseline['numeric'][feat] = {
            'mean': float(s.mean()),
            'std': float(s.std() or 0.0),
            'bins': [float(x) for x in b.tolist()],
            'counts': [int(x) for x in cnt.tolist()]
        }
for feat in categorical:
    vc = df[feat].astype(str).str.strip().value_counts(dropna=False)
    total = int(vc.sum()) or 1
    freq = {str(k): float(v)/total for k, v in vc.to_dict().items()}
    baseline['categorical'][feat] = {'freq': freq}
with open(os.path.join(out_dir, 'baseline.json'), 'w', encoding='utf-8') as f:
    json.dump(baseline, f)

seg_key = os.getenv('SEGMENTATION_KEY')
if seg_key in {'plan','region','tipo_cliente'}:
    segments = sorted(df[seg_key].unique().tolist())
    seg_metrics = {}
    for s in segments:
        dfi = df[df[seg_key] == s].copy()
        Xi = dfi[['plan', 'tiempo_contrato_meses', 'retrasos_pago', 'uso_mensual',
                  'nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']]
        yi = dfi['churn']
        Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.2, random_state=42, stratify=yi)
        pi = Pipeline([('pre', preprocess), ('clf', LogisticRegression(max_iter=1000))])
        pi.fit(Xtr, ytr)
        pred_i = pi.predict(Xte)
        mi = {
            'accuracy': float(accuracy_score(yte, pred_i)),
            'precision': float(precision_score(yte, pred_i)),
            'recall': float(recall_score(yte, pred_i)),
            'f1': float(f1_score(yte, pred_i))
        }
        seg_metrics[s] = mi
        fname = f"model__seg__{seg_key}__{s}.joblib"
        joblib.dump(pi, os.path.join(out_dir, fname))
        try:
            Xtr2, Xcal2, ytr2, ycal2 = train_test_split(Xtr, ytr, test_size=0.25, random_state=42, stratify=ytr)
            pi.fit(Xtr2, ytr2)
            base_p = pi.predict_proba(Xcal2)[:, 1]
            if calib_method == 'sigmoid':
                from sklearn.linear_model import LogisticRegression as LR
                lr2 = LR(max_iter=1000)
                lr2.fit(base_p.reshape(-1, 1), ycal2)
                joblib.dump(lr2, os.path.join(out_dir, f"calibrator__seg__{seg_key}__{s}__sigmoid.joblib"))
            else:
                iso2 = IsotonicRegression(out_of_bounds='clip')
                iso2.fit(base_p, ycal2)
                joblib.dump(iso2, os.path.join(out_dir, f"calibrator__seg__{seg_key}__{s}__isotonic.joblib"))
            pi.fit(Xtr, ytr)
        except Exception:
            pass
    pd.Series({'key': seg_key, 'segments': segments}).to_json(os.path.join(out_dir, 'segments.json'))
    # Guardamos métricas por segmento
    with open(os.path.join(out_dir, 'seg_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'key': seg_key, 'metrics': seg_metrics}, f)