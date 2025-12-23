from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
import subprocess
import sys
import json
import time
import unicodedata
import re
import warnings

class ClientInput(BaseModel):
    tiempo_contrato_meses: int = Field(...)
    retrasos_pago: int = Field(...)
    uso_mensual: float = Field(...)
    plan: str = Field(...)
    nps: int = Field(...)
    quejas: int = Field(...)
    canal_contacto: str = Field(...)
    interacciones_soporte: int = Field(...)
    tipo_pago: str = Field(...)
    region: str = Field(...)
    tipo_cliente: str = Field(...)

app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'model.joblib')
model_path = os.path.normpath(model_path)
if not os.path.exists(model_path):
    raise RuntimeError('Modelo no encontrado. Ejecute el entrenamiento.')

pipe = joblib.load(model_path)
segments_manifest_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'segments.json')
segments_manifest_path = os.path.normpath(segments_manifest_path)
segmentation_key = os.getenv('SEGMENTATION_KEY')
segment_models = {}
segment_calibrators = {}
global_calibrator = None
calibration_method = None
calibration_metrics = {}
segment_metrics = {}
history_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'dashboard_history.json')
history_path = os.path.normpath(history_path)
ab_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'ab_experiments.json')
ab_path = os.path.normpath(ab_path)
ab_data = {}
baseline_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'baseline.json')
baseline_path = os.path.normpath(baseline_path)
baseline_stats = {}
drift_state = {'last': None, 'events': []}
def _discover_segment_models():
    base = os.path.dirname(model_path)
    segment_models.clear()
    if segmentation_key:
        prefix = f"model__seg__{segmentation_key}__"
        for name in os.listdir(base):
            if name.startswith(prefix) and name.endswith('.joblib'):
                seg_val = name[len(prefix):-7]
                try:
                    segment_models[seg_val] = joblib.load(os.path.join(base, name))
                except Exception:
                    pass
def _discover_calibrators():
    base = os.path.dirname(model_path)
    global global_calibrator, calibration_method, calibration_metrics
    global segment_calibrators
    global_calibrator = None
    calibration_method = None
    calibration_metrics = {}
    segment_calibrators.clear()
    for name in os.listdir(base):
        if name.startswith('calibrator__') and name.endswith('.joblib') and not name.startswith('calibrator__seg__'):
            p = os.path.join(base, name)
            try:
                global_calibrator = joblib.load(p)
                if '__isotonic' in name or 'calibrator__isotonic.joblib' in name:
                    calibration_method = 'isotonic'
                elif '__sigmoid' in name or 'calibrator__sigmoid.joblib' in name:
                    calibration_method = 'sigmoid'
                else:
                    calibration_method = 'unknown'
                break
            except Exception:
                global_calibrator = None
    try:
        cal_json = os.path.join(base, 'calibration.json')
        if os.path.exists(cal_json):
            with open(cal_json, 'r', encoding='utf-8') as f:
                calibration_metrics = json.load(f)
    except Exception:
        calibration_metrics = {}
    if segmentation_key:
        seg_prefix = f"calibrator__seg__{segmentation_key}__"
        for name in os.listdir(base):
            if name.startswith(seg_prefix) and name.endswith('.joblib'):
                parts = name.split('__')
                if len(parts) >= 5:
                    seg_val = parts[3]
                    method = parts[4][:-7]
                    try:
                        cal = joblib.load(os.path.join(base, name))
                        segment_calibrators[seg_val] = (method, cal)
                    except Exception:
                        pass
    try:
        segm_json = os.path.join(base, 'seg_metrics.json')
        if os.path.exists(segm_json):
            with open(segm_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'metrics' in data:
                    segment_metrics.clear()
                    segment_metrics.update(data['metrics'])
    except Exception:
        segment_metrics = {}
    try:
        bp = os.path.join(base, 'baseline.json')
        if os.path.exists(bp):
            with open(bp, 'r', encoding='utf-8') as f:
                global baseline_stats
                baseline_stats = json.load(f)
    except Exception:
        baseline_stats = {}
def _load_models():
    global pipe
    pipe = joblib.load(model_path)
    _discover_segment_models()
    _discover_calibrators()
_load_models()
stats = {'total_evaluados': 0, 'tasa_churn': 0.0, 'churn_count': 0}
dashboard_history = []
default_threshold = float(os.getenv('CHURN_THRESHOLD', '0.5'))
auto_mode = os.getenv('AUTO_THRESHOLD')
beneficio_env = float(os.getenv('RETENTION_BENEFIT', '40'))
costo_env = float(os.getenv('RETENTION_COST', '10'))
drift_threshold = float(os.getenv('DRIFT_THRESHOLD', '0.25'))
auto_retrain = os.getenv('AUTO_RETRAIN', '0') in {'1','true','True'}

def _load_history():
    global dashboard_history
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dashboard_history = data[-1000:]
    except Exception:
        dashboard_history = []

def _save_history():
    try:
        data = dashboard_history[-1000:]
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

_load_history()

def _load_validation_df():
    path = os.getenv('VALIDATION_PATH')
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        cols = ['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','churn']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            rng = np.random.default_rng(42)
            for c in missing:
                if c == 'churn':
                    df[c] = rng.integers(0, 2, len(df))
                elif c in ['plan']:
                    df[c] = rng.choice(['Basico','Estandar','Premium'], len(df))
                elif c in ['canal_contacto']:
                    df[c] = rng.choice(['web','app','telefono','email','chat'], len(df))
                elif c in ['tipo_pago']:
                    df[c] = rng.choice(['tarjeta','transferencia','efectivo','debito_automatico'], len(df))
                elif c in ['tiempo_contrato_meses','retrasos_pago','nps','quejas','interacciones_soporte']:
                    df[c] = rng.integers(0, 10, len(df))
                elif c in ['uso_mensual']:
                    df[c] = rng.normal(15, 5, len(df)).clip(0.1)
        return df
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        'tiempo_contrato_meses': rng.integers(1, 60, n),
        'retrasos_pago': rng.poisson(1.0, n),
        'uso_mensual': np.clip(rng.normal(15, 5, n), 0.1, None),
        'plan': rng.choice(['Basico','Estandar','Premium'], n),
        'nps': rng.integers(0, 11, n),
        'quejas': rng.poisson(0.5, n),
        'canal_contacto': rng.choice(['web','app','telefono','email','chat'], n),
        'interacciones_soporte': rng.poisson(1.0, n),
        'tipo_pago': rng.choice(['tarjeta','transferencia','efectivo','debito_automatico'], n),
        'region': rng.choice(['norte','sur','este','oeste'], n),
        'tipo_cliente': rng.choice(['nuevo','antiguo','corporativo'], n)
    })
    probas = pipe.predict_proba(df)[:, 1]
    df['churn'] = (rng.random(n) < probas).astype(int)
    return df

def _calibrate_threshold(mode: str = 'f1', beneficio: float = 40.0, costo: float = 10.0):
    df = _load_validation_df()
    y = df['churn'].astype(int).values
    X = df[['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']]
    probas = pipe.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_m = -1e18
    for t in thresholds:
        preds = (probas >= t).astype(int)
        if mode == 'f1':
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            denom = 2 * tp + fp + fn
            m = 0.0 if denom == 0 else (2 * tp) / denom
        else:
            actions = preds
            gains = probas * beneficio - costo
            m = float((actions * gains).sum())
        if m > best_m:
            best_m = m
            best_t = float(t)
    return best_t, best_m

if auto_mode in {'f1', 'expected_gain'}:
    t, _m = _calibrate_threshold('f1' if auto_mode == 'f1' else 'expected_gain', beneficio_env, costo_env)
    default_threshold = t
    os.environ['CHURN_THRESHOLD'] = str(t)

@app.post('/predict')
def predict(payload: ClientInput, umbral: float | None = None):
    df = pd.DataFrame([payload.model_dump()])
    used_segment = None
    selected_pipe = pipe
    if segmentation_key == 'plan':
        pv = df.iloc[0]['plan']
        if pv in segment_models:
            selected_pipe = segment_models[pv]
            used_segment = pv
    proba = float(selected_pipe.predict_proba(df)[:, 1][0])
    try:
        if used_segment and used_segment in segment_calibrators:
            m, cal = segment_calibrators[used_segment]
            if m == 'sigmoid':
                proba = float(cal.predict_proba(np.array([[proba]]))[:, 1][0])
            else:
                proba = float(cal.predict(np.array([proba]))[0])
        elif global_calibrator is not None and calibration_method:
            if calibration_method == 'sigmoid':
                proba = float(global_calibrator.predict_proba(np.array([[proba]]))[:, 1][0])
            else:
                proba = float(global_calibrator.predict(np.array([proba]))[0])
    except Exception:
        pass
    threshold = umbral if umbral is not None else default_threshold
    pred = 'Va a cancelar' if proba >= threshold else 'Va a continuar'

    pre = selected_pipe.named_steps['pre']
    clf = selected_pipe.named_steps['clf']
    names = pre.get_feature_names_out()
    xt = pre.transform(df)
    try:
        import numpy as np
        if hasattr(xt, 'toarray'):
            xt_row = xt.toarray()[0]
        else:
            xt_row = xt[0]
        contrib_raw = clf.coef_[0] * xt_row
        contrib_by_feat = {
            'tiempo_contrato_meses': 0.0,
            'retrasos_pago': 0.0,
            'uso_mensual': 0.0,
            'nps': 0.0,
            'quejas': 0.0,
            'interacciones_soporte': 0.0,
            'plan': 0.0,
            'canal_contacto': 0.0,
            'tipo_pago': 0.0,
            'region': 0.0,
            'tipo_cliente': 0.0
        }
        for i, n in enumerate(names):
            if n.startswith('num__'):
                feat = n[len('num__'):]
                if feat in contrib_by_feat:
                    contrib_by_feat[feat] += float(contrib_raw[i])
            elif n.startswith('cat__'):
                for cat_feat in ['plan','canal_contacto','tipo_pago','region','tipo_cliente']:
                    prefix = 'cat__' + cat_feat + '_'
                    if n.startswith(prefix):
                        contrib_by_feat[cat_feat] += float(contrib_raw[i])
                        break
        top3 = sorted(contrib_by_feat.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        explicacion = [k for k, v in top3]
    except Exception:
        explicacion = ['plan','retrasos_pago','tiempo_contrato_meses']
    stats['total_evaluados'] += 1
    if pred == 'Va a cancelar':
        stats['churn_count'] += 1
    stats['tasa_churn'] = stats['churn_count'] / stats['total_evaluados'] if stats['total_evaluados'] else 0.0
    resp = {'prevision': pred, 'probabilidad': proba, 'explicacion': explicacion, 'umbral': threshold}
    if used_segment:
        resp['segmento'] = {'key': 'plan', 'value': used_segment}
    return resp

@app.get('/stats')
def get_stats():
    out = {'total_evaluados': stats['total_evaluados'], 'tasa_churn': round(stats['tasa_churn'], 4)}
    if calibration_metrics:
        out['calibracion'] = calibration_metrics
    if segment_metrics:
        out['segmentos'] = segment_metrics
    return out

@app.get('/segment_metrics')
def get_segment_metrics():
    return segment_metrics

@app.get('/calibration_report')
def calibration_report(bins: int = 10, segment: str | None = None):
    import numpy as np
    df = _load_validation_df()
    y = df['churn'].astype(int).values
    X = df[['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']]
    sp = pipe
    if segmentation_key and segment and segment in segment_models:
        sp = segment_models.get(segment, pipe)
    probas = sp.predict_proba(X)[:, 1]
    prob_cal = probas.copy()
    try:
        if segmentation_key and segment and segment in segment_calibrators:
            m, cal = segment_calibrators[segment]
            if m == 'sigmoid':
                prob_cal = cal.predict_proba(probas.reshape(-1, 1))[:, 1]
            else:
                prob_cal = cal.predict(probas)
        elif global_calibrator is not None and calibration_method:
            if calibration_method == 'sigmoid':
                prob_cal = global_calibrator.predict_proba(probas.reshape(-1, 1))[:, 1]
            else:
                prob_cal = global_calibrator.predict(probas)
    except Exception:
        prob_cal = probas
    bins = int(max(2, min(50, bins)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx_u = np.digitize(probas, edges) - 1
    idx_c = np.digitize(prob_cal, edges) - 1
    data = []
    ece_u = 0.0
    ece_c = 0.0
    n = len(y)
    for b in range(bins):
        mu = idx_u == b
        mc = idx_c == b
        cu = float(np.mean(probas[mu])) if np.any(mu) else float((edges[b] + edges[b+1]) / 2)
        cc = float(np.mean(prob_cal[mc])) if np.any(mc) else float((edges[b] + edges[b+1]) / 2)
        fu = float(np.mean(y[mu])) if np.any(mu) else 0.0
        fc = float(np.mean(y[mc])) if np.any(mc) else 0.0
        cnt = int(np.sum(mu))
        ece_u += (cnt / n) * abs(cu - fu)
        ece_c += (cnt / n) * abs(cc - fc)
        data.append({'bin_low': float(edges[b]), 'bin_high': float(edges[b+1]), 'conf_uncal': cu, 'freq_uncal': fu, 'conf_cal': cc, 'freq_cal': fc, 'count': cnt})
    return {'bins': bins, 'ece_uncal': float(ece_u), 'ece_cal': float(ece_c), 'data': data}

@app.post('/retrain')
def retrain(data_path: str | None = None):
    script = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'src', 'train_model.py')
    script = os.path.normpath(script)
    env = os.environ.copy()
    if data_path:
        env['TRAIN_DATA_PATH'] = data_path
    subprocess.run([sys.executable, script], check=True, env=env)
    _load_models()
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'models', 'metrics.json')
    metrics_path = os.path.normpath(metrics_path)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    return {'status': 'ok', 'metrics': metrics}

def _explain_for(selected_pipe, df_row):
    pre = selected_pipe.named_steps['pre']
    clf = selected_pipe.named_steps['clf']
    names = pre.get_feature_names_out()
    xt = pre.transform(df_row)
    import numpy as np
    if hasattr(xt, 'toarray'):
        xt_row = xt.toarray()[0]
    else:
        xt_row = xt[0]
    contrib_raw = clf.coef_[0] * xt_row
    contrib_by_feat = {
        'tiempo_contrato_meses': 0.0,
        'retrasos_pago': 0.0,
        'uso_mensual': 0.0,
        'nps': 0.0,
        'quejas': 0.0,
        'interacciones_soporte': 0.0,
        'plan': 0.0,
        'canal_contacto': 0.0,
        'tipo_pago': 0.0,
        'region': 0.0,
        'tipo_cliente': 0.0
    }
    for i, n in enumerate(names):
        if n.startswith('num__'):
            feat = n[len('num__'):]
            if feat in contrib_by_feat:
                contrib_by_feat[feat] += float(contrib_raw[i])
        elif n.startswith('cat__'):
            for cat_feat in ['plan','canal_contacto','tipo_pago','region','tipo_cliente']:
                prefix = 'cat__' + cat_feat + '_'
                if n.startswith(prefix):
                    contrib_by_feat[cat_feat] += float(contrib_raw[i])
                    break
    top3 = sorted(contrib_by_feat.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return [k for k, v in top3]

def _explain_batch(selected_pipe, df, top_k: int = 3):
    pre = selected_pipe.named_steps['pre']
    clf = selected_pipe.named_steps['clf']
    names = pre.get_feature_names_out()
    xt = pre.transform(df)
    import numpy as np
    if hasattr(xt, 'toarray'):
        X = xt.toarray()
    else:
        X = xt
    coef = clf.coef_[0]
    contrib = X * coef
    idx_map = {
        'tiempo_contrato_meses': [],
        'retrasos_pago': [],
        'uso_mensual': [],
        'nps': [],
        'quejas': [],
        'interacciones_soporte': [],
        'plan': [],
        'canal_contacto': [],
        'tipo_pago': [],
        'region': [],
        'tipo_cliente': []
    }
    for i, n in enumerate(names):
        if n.startswith('num__'):
            feat = n[len('num__'):]
            if feat in idx_map:
                idx_map[feat].append(i)
        elif n.startswith('cat__'):
            for cat_feat in ['plan','canal_contacto','tipo_pago','region','tipo_cliente']:
                prefix = 'cat__' + cat_feat + '_'
                if n.startswith(prefix):
                    idx_map[cat_feat].append(i)
                    break
    out = []
    m = X.shape[0]
    for r in range(m):
        d = {}
        for feat, idxs in idx_map.items():
            if len(idxs) == 0:
                d[feat] = 0.0
            else:
                s = float(np.sum(contrib[r, idxs]))
                d[feat] = s
        topn = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max(1, int(top_k))]
        out.append([k for k, v in topn])
    return out

@app.post('/predict_bulk')
def predict_bulk(file: UploadFile = File(...), umbral: float | None = None, explain_top: int | None = None, strict_telco: int | None = None):
    content = file.file.read()
    import io
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail='CSV inválido: no se pudo leer el archivo')
    def _norm(s: str):
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.strip().lower().replace(' ', '_').replace('-', '_')
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s
    # Modo estricto: solo aceptar telecom_cleaned.csv
    if strict_telco:
        telco_required = {
            'churn','customer_gender','customer_seniorcitizen','customer_partner','customer_dependents','customer_tenure',
            'phone_phoneservice','phone_multiplelines','internet_internetservice','internet_onlinesecurity','internet_onlinebackup',
            'internet_deviceprotection','internet_techsupport','internet_streamingtv','internet_streamingmovies',
            'account_contract','account_paperlessbilling','account_paymentmethod','account_charges_monthly','account_charges_total'
        }
        cols_norm = {_norm(c) for c in df.columns}
        if not telco_required.issubset(cols_norm):
            raise HTTPException(status_code=400, detail='Solo se acepta telecom_cleaned.csv')
    synonyms = {
        'plan':'plan','tipo_plan':'plan','plan_tipo':'plan','plan_tarifa':'plan','plan_name':'plan','subscription_plan':'plan','tarifa':'plan','tipo_de_contrato':'plan','tipo_contrato':'plan','contract':'plan','contracttype':'plan','contract_type':'plan',
        'tiempo_contrato_meses':'tiempo_contrato_meses','meses_contrato':'tiempo_contrato_meses','meses_de_contrato':'tiempo_contrato_meses','meses_servicio':'tiempo_contrato_meses',
        'tenure':'tiempo_contrato_meses','tenure_months':'tiempo_contrato_meses','tenuremonths':'tiempo_contrato_meses','customer_tenure':'tiempo_contrato_meses','customer_tenure_months':'tiempo_contrato_meses',
        'months_on_service':'tiempo_contrato_meses','months_of_service':'tiempo_contrato_meses','contract_months':'tiempo_contrato_meses','months_of_contract':'tiempo_contrato_meses',
        'retrasos_pago':'retrasos_pago','atrasos_pago':'retrasos_pago','late_payments':'retrasos_pago','latepaymentcount':'retrasos_pago','payments_late':'retrasos_pago','payment_late_count':'retrasos_pago',
        'uso_mensual':'uso_mensual','monthly_usage':'uso_mensual','monthly_charges':'uso_mensual','monthlycharges':'uso_mensual','monthly_charge':'uso_mensual','charges_monthly':'uso_mensual','daily_watch_time_hours':'uso_mensual',
        'monthly_fees':'uso_mensual','monthly_fee':'uso_mensual','amount_monthly':'uso_mensual','avg_monthly_usage':'uso_mensual','average_monthly_usage':'uso_mensual','usage_monthly':'uso_mensual','usage_per_month':'uso_mensual','average_usage_per_month':'uso_mensual','total_usage_monthly':'uso_mensual','total_monthly_usage':'uso_mensual','monthlyfee':'uso_mensual',
        'nps':'nps','nps_score':'nps','net_promoter_score':'nps','netpromoterscore':'nps','customer_satisfaction_score_1_10':'nps',
        'quejas':'quejas','reclamos':'quejas','complaints':'quejas','complaints_count':'quejas','num_complaints':'quejas',
        'canal_contacto':'canal_contacto','canal':'canal_contacto','contact_channel':'canal_contacto','customer_service_channel':'canal_contacto','support_channel':'canal_contacto','channel':'canal_contacto','service_channel':'canal_contacto','device_used_most_often':'canal_contacto',
        'interacciones_soporte':'interacciones_soporte','soporte_interacciones':'interacciones_soporte','support_interactions':'interacciones_soporte','cs_interactions':'interacciones_soporte','helpdesk_interactions':'interacciones_soporte','tickets_count':'interacciones_soporte',
        'tipo_pago':'tipo_pago','payment_type':'tipo_pago','tipo_de_pago':'tipo_pago','paymentmethod':'tipo_pago','payment_method':'tipo_pago','method_payment':'tipo_pago','pay_method':'tipo_pago','payment':'tipo_pago','paytype':'tipo_pago',
        'region':'region','zona':'region','area':'region','geo_region':'region','region_name':'region',
        'tipo_cliente':'tipo_cliente','customer_type':'tipo_cliente','tipo_de_cliente':'tipo_cliente','client_type':'tipo_cliente','segment':'tipo_cliente','customer_segment':'tipo_cliente'
    }
    # Mapeos específicos de datasets populares
    synonyms.update({
        'contract':'plan','contracttype':'plan','contract_type':'plan'
    })
    rename_map = {}
    for c in list(df.columns):
        nc = _norm(str(c))
        if nc in synonyms:
            rename_map[c] = synonyms[nc]
    if rename_map:
        df = df.rename(columns=rename_map)
    # Ajustes específicos para columnas mapeadas desde datasets externos
    if 'payment_history_on_time_delayed' in df.columns and 'retrasos_pago' not in df.columns:
        ph = df['payment_history_on_time_delayed'].astype(str).str.strip().str.lower()
        df['retrasos_pago'] = ph.map({'on-time': 0, 'ontime': 0, 'on_time': 0, 'delayed': 1}).fillna(0).astype(int)
    if 'support_queries_logged' in df.columns and 'interacciones_soporte' not in df.columns:
        df['interacciones_soporte'] = pd.to_numeric(df['support_queries_logged'], errors='coerce').fillna(0).astype(int)
    if 'device_used_most_often' in df.columns:
        dv = df['device_used_most_often'].astype(str).str.strip().str.lower()
        df['canal_contacto'] = dv.map({
            'smart tv':'app','smart_tv':'app','tv':'app','televisor':'app',
            'tablet':'app','phone':'app','mobile':'app','smartphone':'app',
            'laptop':'web','pc':'web','desktop':'web'
        }).fillna(df.get('canal_contacto','app'))
    if 'subscription_length_months' in df.columns and 'tiempo_contrato_meses' not in df.columns:
        df['tiempo_contrato_meses'] = pd.to_numeric(df['subscription_length_months'], errors='coerce')
    if 'daily_watch_time_hours' in df.columns and 'uso_mensual' not in df.columns:
        df['uso_mensual'] = pd.to_numeric(df['daily_watch_time_hours'], errors='coerce')
    required = ['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if 'customer_tenure' in df.columns:
            df['tiempo_contrato_meses'] = df['customer_tenure']
        if 'account_Charges_Monthly' in df.columns:
            df['uso_mensual'] = df['account_Charges_Monthly']
        if 'account_PaymentMethod' in df.columns:
            pm = df['account_PaymentMethod'].astype(str).str.strip().str.lower()
            df['tipo_pago'] = pm.map({
                'electronic check':'transferencia',
                'mailed check':'efectivo',
                'bank transfer (automatic)':'debito_automatico',
                'credit card (automatic)':'tarjeta'
            }).fillna('transferencia')
        if 'account_Contract' in df.columns:
            ct = df['account_Contract'].astype(str).str.strip().str.lower()
            df['plan'] = ct.map({
                'month-to-month':'Basico',
                'one year':'Estandar',
                'two year':'Premium'
            }).fillna('Basico')
        if 'region' not in df.columns:
            df['region'] = 'norte'
        if 'tipo_cliente' not in df.columns:
            if 'customer_tenure' in df.columns:
                df['tipo_cliente'] = np.where(df['customer_tenure'].astype(float) >= 12, 'antiguo', 'nuevo')
            else:
                df['tipo_cliente'] = 'nuevo'
        for col, default in [
            ('retrasos_pago', 0),
            ('nps', 7),
            ('quejas', 0),
            ('canal_contacto', 'app'),
            ('interacciones_soporte', 1),
            ('tipo_pago', 'transferencia'),
            ('plan', 'Basico'),
            ('tiempo_contrato_meses', 0),
            ('uso_mensual', 0.0)
        ]:
            if col not in df.columns:
                df[col] = default
    core = ['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual']
    present_core = [c for c in core if c in df.columns]
    if len(present_core) < 3:
        raise HTTPException(status_code=400, detail='CSV inválido: requiere al menos 3 de plan,tiempo_contrato_meses,retrasos_pago,uso_mensual')
    def _harmonize_values(df):
        def _norm_val(s):
            s = str(s).strip().lower()
            s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            return s
        if 'plan' in df:
            m = {
                'basico':'Basico','basic':'Basico','month-to-month':'Basico','m2m':'Basico','mensual':'Basico',
                'estandar':'Estandar','standard':'Estandar','one year':'Estandar','1 year':'Estandar','anual':'Estandar',
                'premium':'Premium','two year':'Premium','2 year':'Premium','avanzado':'Premium'
            }
            df['plan'] = df['plan'].map(lambda x: m.get(_norm_val(x), 'Basico'))
        if 'tipo_pago' in df:
            m = {
                'electronic check':'transferencia','transferencia':'transferencia','bank transfer':'transferencia','wire':'transferencia','bank transfer (automatic)':'debito_automatico',
                'debito_automatico':'debito_automatico','debit':'debito_automatico','direct debit':'debito_automatico',
                'credit card (automatic)':'tarjeta','tarjeta':'tarjeta','credit card':'tarjeta',
                'cash':'efectivo','efectivo':'efectivo','cheque':'efectivo','mailed check':'efectivo'
            }
            df['tipo_pago'] = df['tipo_pago'].map(lambda x: m.get(_norm_val(x), 'transferencia'))
        if 'canal_contacto' in df:
            m = {
                'web':'web','website':'web','portal':'web','site':'web',
                'app':'app','mobile':'app','android':'app','ios':'app','movil':'app',
                'telefono':'telefono','phone':'telefono','call':'telefono','hotline':'telefono',
                'email':'email','mail':'email','correo':'email',
                'chat':'chat','whatsapp':'chat'
            }
            df['canal_contacto'] = df['canal_contacto'].map(lambda x: m.get(_norm_val(x), 'web'))
        if 'region' in df:
            m = {
                'norte':'norte','north':'norte','n':'norte','north_america':'norte',
                'sur':'sur','south':'sur','s':'sur','south_america':'sur','africa':'sur',
                'este':'este','east':'este','e':'este','europe':'este','asia':'este',
                'oeste':'oeste','west':'oeste','w':'oeste','oceania':'oeste'
            }
            df['region'] = df['region'].map(lambda x: m.get(_norm_val(x), 'norte'))
        if 'tipo_cliente' in df:
            m = {
                'nuevo':'nuevo','new':'nuevo',
                'antiguo':'antiguo','old':'antiguo','existing':'antiguo','existing customer':'antiguo',
                'corporativo':'corporativo','corporate':'corporativo','enterprise':'corporativo'
            }
            df['tipo_cliente'] = df['tipo_cliente'].map(lambda x: m.get(_norm_val(x), 'nuevo'))
        num_cols = ['tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','interacciones_soporte']
        for c in num_cols:
            if c in df:
                if c == 'uso_mensual':
                    s = df[c].astype(str).str.replace(r'[^0-9,.-]', '', regex=True).str.replace(',', '.', regex=False)
                    df[c] = pd.to_numeric(s, errors='coerce')
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
        fill_defaults = {
            'tiempo_contrato_meses': 0,
            'retrasos_pago': 0,
            'uso_mensual': 0.0,
            'nps': 7,
            'quejas': 0,
            'interacciones_soporte': 1
        }
        for k, v in fill_defaults.items():
            if k in df:
                df[k] = df[k].fillna(v)
        return df
    df = _harmonize_values(df)
    # Detección de drift y auto-reentrenamiento
    drift = _compute_drift(df)
    drift_state['last'] = {'ts': int(time.time()*1000), 'score': drift.get('score', 0.0), 'detail': drift}
    drift_state['events'].append(drift_state['last'])
    if len(drift_state['events']) > 100:
        drift_state['events'] = drift_state['events'][-100:]
    if auto_retrain and drift.get('score', 0.0) >= drift_threshold:
        try:
            import threading
            def _bg_retrain():
                try:
                    script = os.path.join(os.path.dirname(__file__), '..', 'data-science', 'src', 'train_model.py')
                    script = os.path.normpath(script)
                    env = os.environ.copy()
                    subprocess.run([sys.executable, script], check=True, env=env)
                    _load_models()
                except Exception:
                    pass
            threading.Thread(target=_bg_retrain, daemon=True).start()
        except Exception:
            pass
    threshold = umbral if umbral is not None else default_threshold
    top_k = 1 if explain_top == 1 else 3
    results = []
    if segmentation_key:
        key = segmentation_key
        for val, grp in df.groupby(key):
            sp = segment_models.get(val, pipe)
            probas = sp.predict_proba(grp)[:, 1]
            try:
                expls = _explain_batch(sp, grp, top_k)
            except Exception:
                base = ['plan','retrasos_pago','tiempo_contrato_meses']
                expls = [base[:top_k] for _ in range(len(grp))]
            for j, (_, row) in enumerate(grp.iterrows()):
                proba = float(probas[j])
                try:
                    if segmentation_key and val in segment_calibrators:
                        m, cal = segment_calibrators[val]
                        if m == 'sigmoid':
                            proba = float(cal.predict_proba(np.array([[proba]]))[:, 1][0])
                        else:
                            proba = float(cal.predict(np.array([proba]))[0])
                    elif global_calibrator is not None and calibration_method:
                        if calibration_method == 'sigmoid':
                            proba = float(global_calibrator.predict_proba(np.array([[proba]]))[:, 1][0])
                        else:
                            proba = float(global_calibrator.predict(np.array([proba]))[0])
                except Exception:
                    pass
                pred = 'Va a cancelar' if proba >= threshold else 'Va a continuar'
                item = {'prevision': pred, 'probabilidad': proba, 'explicacion': expls[j], 'umbral': threshold, 'id': int(len(results))}
                item['plan'] = row['plan']
                item['region'] = row['region']
                item['tipo_cliente'] = row['tipo_cliente']
                item['segmento'] = {'key': key, 'value': val}
                results.append(item)
    else:
        probas = pipe.predict_proba(df)[:, 1]
        try:
            expls = _explain_batch(pipe, df, top_k)
        except Exception:
            base = ['plan','retrasos_pago','tiempo_contrato_meses']
            expls = [base[:top_k] for _ in range(len(df))]
        for idx in range(len(df)):
            proba = float(probas[idx])
            try:
                if global_calibrator is not None and calibration_method:
                    if calibration_method == 'sigmoid':
                        proba = float(global_calibrator.predict_proba(np.array([[proba]]))[:, 1][0])
                    else:
                        proba = float(global_calibrator.predict(np.array([proba]))[0])
            except Exception:
                pass
            pred = 'Va a cancelar' if proba >= threshold else 'Va a continuar'
            item = {'prevision': pred, 'probabilidad': proba, 'explicacion': expls[idx], 'umbral': threshold, 'id': int(idx)}
            row = df.iloc[idx]
            item['plan'] = row['plan']
            item['region'] = row['region']
            item['tipo_cliente'] = row['tipo_cliente']
            results.append(item)
    avg = float(np.mean([it['probabilidad'] for it in results])) if results else 0.0
    dashboard_history.append({'ts': int(time.time()*1000), 'avg': avg})
    _save_history()
    return results

@app.get('/drift_status')
def drift_status():
    return drift_state

@app.post('/normalize_csv')
def normalize_csv(file: UploadFile = File(...), save_path: str | None = None, strict_telco: int | None = 0):
    content = file.file.read()
    import io
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail='CSV inválido: no se pudo leer el archivo')
    def _norm(s: str):
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.strip().lower().replace(' ', '_').replace('-', '_')
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s
    if strict_telco:
        telco_required = {
            'churn','customer_gender','customer_seniorcitizen','customer_partner','customer_dependents','customer_tenure',
            'phone_phoneservice','phone_multiplelines','internet_internetservice','internet_onlinesecurity','internet_onlinebackup',
            'internet_deviceprotection','internet_techsupport','internet_streamingtv','internet_streamingmovies',
            'account_contract','account_paperlessbilling','account_paymentmethod','account_charges_monthly','account_charges_total'
        }
        cols_norm = {_norm(c) for c in df.columns}
        if not telco_required.issubset(cols_norm):
            raise HTTPException(status_code=400, detail='Solo se acepta telecom_cleaned.csv')
    synonyms = {
        'plan':'plan','tipo_plan':'plan','plan_tipo':'plan','plan_tarifa':'plan','plan_name':'plan','subscription_plan':'plan','tarifa':'plan','tipo_de_contrato':'plan','tipo_contrato':'plan','contract':'plan','contracttype':'plan','contract_type':'plan',
        'tiempo_contrato_meses':'tiempo_contrato_meses','meses_contrato':'tiempo_contrato_meses','meses_de_contrato':'tiempo_contrato_meses','meses_servicio':'tiempo_contrato_meses',
        'tenure':'tiempo_contrato_meses','tenure_months':'tiempo_contrato_meses','tenuremonths':'tiempo_contrato_meses','customer_tenure':'tiempo_contrato_meses','customer_tenure_months':'tiempo_contrato_meses',
        'months_on_service':'tiempo_contrato_meses','months_of_service':'tiempo_contrato_meses','contract_months':'tiempo_contrato_meses','months_of_contract':'tiempo_contrato_meses',
        'retrasos_pago':'retrasos_pago','atrasos_pago':'retrasos_pago','late_payments':'retrasos_pago','latepaymentcount':'retrasos_pago','payments_late':'retrasos_pago','payment_late_count':'retrasos_pago',
        'uso_mensual':'uso_mensual','monthly_usage':'uso_mensual','monthly_charges':'uso_mensual','monthlycharges':'uso_mensual','monthly_charge':'uso_mensual','charges_monthly':'uso_mensual','daily_watch_time_hours':'uso_mensual',
        'monthly_fees':'uso_mensual','monthly_fee':'uso_mensual','amount_monthly':'uso_mensual','avg_monthly_usage':'uso_mensual','average_monthly_usage':'uso_mensual','usage_monthly':'uso_mensual','usage_per_month':'uso_mensual','average_usage_per_month':'uso_mensual','total_usage_monthly':'uso_mensual','total_monthly_usage':'uso_mensual','monthlyfee':'uso_mensual',
        'nps':'nps','nps_score':'nps','net_promoter_score':'nps','netpromoterscore':'nps','customer_satisfaction_score_1_10':'nps',
        'quejas':'quejas','reclamos':'quejas','complaints':'quejas','complaints_count':'quejas','num_complaints':'quejas',
        'canal_contacto':'canal_contacto','canal':'canal_contacto','contact_channel':'canal_contacto','customer_service_channel':'canal_contacto','support_channel':'canal_contacto','channel':'canal_contacto','service_channel':'canal_contacto','device_used_most_often':'canal_contacto',
        'interacciones_soporte':'interacciones_soporte','soporte_interacciones':'interacciones_soporte','support_interactions':'interacciones_soporte','cs_interactions':'interacciones_soporte','helpdesk_interactions':'interacciones_soporte','tickets_count':'interacciones_soporte','support_queries_logged':'interacciones_soporte',
        'tipo_pago':'tipo_pago','payment_type':'tipo_pago','tipo_de_pago':'tipo_pago','paymentmethod':'tipo_pago','payment_method':'tipo_pago','method_payment':'tipo_pago','pay_method':'tipo_pago','payment':'tipo_pago','paytype':'tipo_pago',
        'region':'region','zona':'region','area':'region','geo_region':'region','region_name':'region',
        'tipo_cliente':'tipo_cliente','customer_type':'tipo_cliente','tipo_de_cliente':'tipo_cliente','client_type':'tipo_cliente','segment':'tipo_cliente','customer_segment':'tipo_cliente'
    }
    synonyms.update({
        'contract':'plan','contracttype':'plan','contract_type':'plan'
    })
    rename_map = {}
    for c in list(df.columns):
        nc = _norm(str(c))
        if nc in synonyms:
            rename_map[c] = synonyms[nc]
    if rename_map:
        df = df.rename(columns=rename_map)
    if 'payment_history_on_time_delayed' in df.columns and 'retrasos_pago' not in df.columns:
        ph = df['payment_history_on_time_delayed'].astype(str).str.strip().str.lower()
        df['retrasos_pago'] = ph.map({'on-time': 0, 'ontime': 0, 'on_time': 0, 'delayed': 1}).fillna(0).astype(int)
    if 'device_used_most_often' in df.columns:
        dv = df['device_used_most_often'].astype(str).str.strip().str.lower()
        df['canal_contacto'] = dv.map({'smart tv':'app','smart_tv':'app','tv':'app','televisor':'app','tablet':'app','phone':'app','mobile':'app','smartphone':'app','laptop':'web','pc':'web','desktop':'web'}).fillna(df.get('canal_contacto','app'))
    if 'subscription_length_months' in df.columns and 'tiempo_contrato_meses' not in df.columns:
        df['tiempo_contrato_meses'] = pd.to_numeric(df['subscription_length_months'], errors='coerce')
    if 'daily_watch_time_hours' in df.columns and 'uso_mensual' not in df.columns:
        df['uso_mensual'] = pd.to_numeric(df['daily_watch_time_hours'], errors='coerce')
    required = ['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','canal_contacto','interacciones_soporte','tipo_pago','region','tipo_cliente']
    if 'region' not in df.columns:
        df['region'] = 'norte'
    if 'tipo_cliente' not in df.columns:
        if 'customer_tenure' in df.columns:
            df['tipo_cliente'] = np.where(pd.to_numeric(df['customer_tenure'], errors='coerce').fillna(0) >= 12, 'antiguo', 'nuevo')
        else:
            df['tipo_cliente'] = 'nuevo'
    for col, default in [('retrasos_pago', 0), ('nps', 7), ('quejas', 0), ('canal_contacto', 'app'), ('interacciones_soporte', 1), ('tipo_pago', 'transferencia'), ('plan','Basico'), ('tiempo_contrato_meses',0), ('uso_mensual',0.0)]:
        if col not in df.columns:
            df[col] = default
    df = df[[c for c in df.columns if c in set(required) or c not in set(required)]]
    def _norm_val(s):
        s = str(s).strip().lower()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s
    if 'plan' in df:
        m = {'basico':'Basico','basic':'Basico','month-to-month':'Basico','m2m':'Basico','mensual':'Basico','estandar':'Estandar','standard':'Estandar','one year':'Estandar','1 year':'Estandar','anual':'Estandar','premium':'Premium','two year':'Premium','2 year':'Premium','avanzado':'Premium'}
        df['plan'] = df['plan'].map(lambda x: m.get(_norm_val(x), 'Basico'))
    if 'tipo_pago' in df:
        m = {'electronic check':'transferencia','transferencia':'transferencia','bank transfer':'transferencia','wire':'transferencia','bank transfer (automatic)':'debito_automatico','debito_automatico':'debito_automatico','debit':'debito_automatico','direct debit':'debito_automatico','credit card (automatic)':'tarjeta','tarjeta':'tarjeta','credit card':'tarjeta','cash':'efectivo','efectivo':'efectivo','cheque':'efectivo','mailed check':'efectivo'}
        df['tipo_pago'] = df['tipo_pago'].map(lambda x: m.get(_norm_val(x), 'transferencia'))
    if 'canal_contacto' in df:
        m = {'web':'web','website':'web','portal':'web','site':'web','app':'app','mobile':'app','android':'app','ios':'app','movil':'app','telefono':'telefono','phone':'telefono','call':'telefono','hotline':'telefono','email':'email','mail':'email','correo':'email','chat':'chat','whatsapp':'chat'}
        df['canal_contacto'] = df['canal_contacto'].map(lambda x: m.get(_norm_val(x), 'web'))
    if 'region' in df:
        m = {'norte':'norte','north':'norte','n':'norte','north_america':'norte','sur':'sur','south':'sur','s':'sur','south_america':'sur','africa':'sur','este':'este','east':'este','e':'este','europe':'este','asia':'este','oeste':'oeste','west':'oeste','w':'oeste','oceania':'oeste'}
        df['region'] = df['region'].map(lambda x: m.get(_norm_val(x), 'norte'))
    if 'tipo_cliente' in df:
        m = {'nuevo':'nuevo','new':'nuevo','antiguo':'antiguo','old':'antiguo','existing':'antiguo','existing customer':'antiguo','corporativo':'corporativo','corporate':'corporativo','enterprise':'corporativo'}
        df['tipo_cliente'] = df['tipo_cliente'].map(lambda x: m.get(_norm_val(x), 'nuevo'))
    num_cols = ['tiempo_contrato_meses','retrasos_pago','uso_mensual','nps','quejas','interacciones_soporte']
    for c in num_cols:
        if c in df:
            if c == 'uso_mensual':
                s = df[c].astype(str).str.replace(r'[^0-9,.-]', '', regex=True).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(s, errors='coerce')
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce')
    fill_defaults = {'tiempo_contrato_meses': 0, 'retrasos_pago': 0, 'uso_mensual': 0.0, 'nps': 7, 'quejas': 0, 'interacciones_soporte': 1}
    for k, v in fill_defaults.items():
        if k in df:
            df[k] = df[k].fillna(v)
    core = ['plan','tiempo_contrato_meses','retrasos_pago','uso_mensual']
    present_core = [c for c in core if c in df.columns]
    if len(present_core) < 3:
        raise HTTPException(status_code=400, detail='CSV inválido: requiere al menos 3 de plan,tiempo_contrato_meses,retrasos_pago,uso_mensual')
    df = df[required]
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    if save_path:
        try:
            sp = os.path.normpath(save_path)
            with open(sp, 'wb') as f:
                f.write(csv_bytes)
        except Exception:
            pass
    return StreamingResponse(io.BytesIO(csv_bytes), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="dataset_normalizado.csv"'})

@app.post('/calibrate_threshold')
def calibrate_threshold(modo: str = 'f1', beneficio: float = 40.0, costo: float = 10.0):
    global default_threshold
    m = 'f1' if modo not in {'f1', 'expected_gain'} else modo
    t, metric = _calibrate_threshold(m, beneficio, costo)
    default_threshold = t
    os.environ['CHURN_THRESHOLD'] = str(t)
    return {'umbral': t, 'modo': m, 'metric': metric}

@app.get('/dashboard_history')
def dashboard_data():
    return dashboard_history[-200:]

@app.post('/clear_history')
def clear_history():
    global dashboard_history
    dashboard_history = []
    try:
        if os.path.exists(history_path):
            os.remove(history_path)
    except Exception:
        pass
    return {'status': 'ok'}

@app.post('/ab/create')
def ab_create(nombre: str, ratio_tratamiento: float = 0.5):
    global ab_data
    r = max(0.0, min(1.0, float(ratio_tratamiento)))
    exp_id = str(int(time.time()*1000))
    ab_data[exp_id] = {
        'nombre': nombre,
        'ratio': r,
        'created_ts': int(time.time()*1000),
        'assignments': {},
        'outcomes': {},
        'report': {}
    }
    _save_ab()
    return {'exp_id': exp_id, 'ratio': r}

@app.post('/ab/assign')
def ab_assign(exp_id: str, file: UploadFile = File(...)):
    global ab_data
    if exp_id not in ab_data:
        raise HTTPException(status_code=404, detail='Experimento no encontrado')
    content = file.file.read()
    import io
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail='CSV inválido: no se pudo leer el archivo')
    def _norm(s: str):
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.strip().lower().replace(' ', '_').replace('-', '_')
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s
    id_cols = {'id','user_id','customer_id','cliente_id'}
    col_id = None
    for c in df.columns:
        if _norm(str(c)) in id_cols:
            col_id = c
            break
    if not col_id:
        raise HTTPException(status_code=400, detail='CSV inválido: falta columna id')
    ids = [str(x) for x in df[col_id].tolist()]
    rng = np.random.default_rng()
    r = float(ab_data[exp_id]['ratio'])
    assign = ab_data[exp_id]['assignments']
    for uid in ids:
        if uid in assign:
            continue
        g = 'treatment' if rng.random() < r else 'control'
        assign[uid] = g
    _save_ab()
    return {'exp_id': exp_id, 'assigned': len(ids)}

@app.post('/ab/outcomes')
def ab_outcomes(exp_id: str, file: UploadFile = File(...)):
    global ab_data
    if exp_id not in ab_data:
        raise HTTPException(status_code=404, detail='Experimento no encontrado')
    content = file.file.read()
    import io
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail='CSV inválido: no se pudo leer el archivo')
    def _norm(s: str):
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.strip().lower().replace(' ', '_').replace('-', '_')
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s
    id_cols = {'id','user_id','customer_id','cliente_id'}
    churn_cols = {'churn','churn_status','churn_status_yes_no'}
    retained_cols = {'retained','retencion'}
    revenue_cols = {'revenue','ingreso','income','monthly_income'}
    col_id = col_churn = col_ret = col_rev = None
    for c in df.columns:
        nc = _norm(str(c))
        if nc in id_cols:
            col_id = c
        elif nc in churn_cols:
            col_churn = c
        elif nc in retained_cols:
            col_ret = c
        elif nc in revenue_cols:
            col_rev = c
    if not col_id:
        raise HTTPException(status_code=400, detail='CSV inválido: falta columna id')
    out = ab_data[exp_id]['outcomes']
    for _, row in df.iterrows():
        uid = str(row[col_id])
        d = out.get(uid, {})
        if col_churn is not None:
            v = str(row[col_churn]).strip().lower()
            d['churn'] = 1 if v in {'1','yes','true','y','si'} else 0
        if col_ret is not None:
            d['retained'] = int(pd.to_numeric(row[col_ret], errors='coerce') == 1)
        if 'retained' not in d and 'churn' in d:
            d['retained'] = 1 - int(d['churn'])
        if col_rev is not None:
            d['revenue'] = float(pd.to_numeric(row[col_rev], errors='coerce')) if str(row[col_rev]) not in {'', 'nan', 'None'} else 0.0
        out[uid] = d
    _save_ab()
    return {'exp_id': exp_id, 'updated': len(df)}

def _ab_report(exp):
    assign = exp.get('assignments', {})
    out = exp.get('outcomes', {})
    ctrl = []
    trt = []
    for uid, g in assign.items():
        if uid in out:
            rec = out[uid]
            r = int(rec.get('retained', 0))
            ctrl.append(r) if g == 'control' else trt.append(r)
    n1 = len(ctrl)
    n2 = len(trt)
    p1 = float(np.mean(ctrl)) if n1 else 0.0
    p2 = float(np.mean(trt)) if n2 else 0.0
    uplift = p2 - p1
    se = np.sqrt((p1*(1-p1)/max(1,n1)) + (p2*(1-p2)/max(1,n2)))
    z = uplift / se if se > 0 else 0.0
    from math import erf, sqrt
    p_value = float(1 - 0.5*(1+erf(abs(z)/sqrt(2))))*2 if se>0 else 1.0
    return {'n_control': n1, 'n_treatment': n2, 'retencion_control': p1, 'retencion_tratamiento': p2, 'uplift': uplift, 'z': z, 'p_value': p_value}

@app.get('/ab/report')
def ab_report(exp_id: str):
    if exp_id not in ab_data:
        raise HTTPException(status_code=404, detail='Experimento no encontrado')
    rep = _ab_report(ab_data[exp_id])
    ab_data[exp_id]['report'] = rep
    _save_ab()
    return {'exp_id': exp_id, 'report': rep}

@app.get('/ab/list')
def ab_list():
    items = []
    for k, v in ab_data.items():
        it = {'exp_id': k, 'nombre': v.get('nombre',''), 'ratio': v.get('ratio',0.5), 'created_ts': v.get('created_ts',0)}
        if 'report' in v:
            it['report'] = v['report']
        items.append(it)
    items.sort(key=lambda x: x['created_ts'], reverse=True)
    return items

def _load_ab():
    global ab_data
    try:
        if os.path.exists(ab_path):
            with open(ab_path, 'r', encoding='utf-8') as f:
                ab_data = json.load(f)
        else:
            ab_data = {}
    except Exception:
        ab_data = {}

def _save_ab():
    try:
        base = os.path.dirname(ab_path)
        os.makedirs(base, exist_ok=True)
        with open(ab_path, 'w', encoding='utf-8') as f:
            json.dump(ab_data, f)
    except Exception:
        pass

@app.get('/@vite/client', response_class=HTMLResponse)
def vite_stub():
    return HTMLResponse(content='', status_code=204)

@app.get('/favicon.ico')
def favicon_stub():
    import base64, io
    data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+VtHkAAAAASUVORK5CYII=')
    return StreamingResponse(io.BytesIO(data), media_type='image/png')

def _compute_drift(df):
    out = {'numeric': {}, 'categorical': {}, 'score': 0.0}
    try:
        # Numeric PSI
        for feat, base in (baseline_stats.get('numeric') or {}).items():
            if feat not in df.columns:
                continue
            bins = base.get('bins') or []
            if not bins:
                continue
            s = pd.to_numeric(df[feat], errors='coerce').dropna()
            if len(s) == 0:
                psi = 0.0
            else:
                cnt, _ = np.histogram(s.values, bins=np.array(bins))
                q = cnt / max(1, int(cnt.sum()))
                p = np.array(base.get('counts') or [])
                p = p / max(1, int((p.sum() if hasattr(p, 'sum') else sum(p))))
                eps = 1e-9
                psi = float(np.sum((p - q) * np.log((p + eps) / (q + eps))))
            out['numeric'][feat] = psi
        # Categorical TVD
        for feat, base in (baseline_stats.get('categorical') or {}).items():
            if feat not in df.columns:
                continue
            freq = base.get('freq') or {}
            total = max(1, len(df))
            batch_freq = df[feat].astype(str).str.strip().value_counts(dropna=False)
            q = {str(k): float(v)/total for k, v in batch_freq.to_dict().items()}
            cats = set(list(freq.keys()) + list(q.keys()))
            tvd = 0.0
            for c in cats:
                tvd += abs(float(freq.get(c, 0.0)) - float(q.get(c, 0.0)))
            tvd *= 0.5
            out['categorical'][feat] = tvd
        # Aggregate score (max for sensitivity)
        vals = list(out['numeric'].values()) + list(out['categorical'].values())
        out['score'] = float(max(vals) if vals else 0.0)
    except Exception:
        out['score'] = 0.0
    return out

@app.get('/dashboard', response_class=HTMLResponse)
def dashboard():
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>ChurnInsight Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{color-scheme: light}
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px;background:#fff;color:#111}
.row{display:flex;gap:20px;flex-wrap:wrap}
.card{border:1px solid #ddd;border-radius:8px;padding:16px;background:#fff}
table{border-collapse:collapse;width:100%;background:#fff;color:#111}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#fafafa;color:#111}
.controls{display:flex;gap:10px;align-items:center;margin:10px 0;color:#111}
input,select,button{padding:6px 10px;background:#fff;color:#111;border:1px solid #bbb}
</style>
</head>
<body>
<h1>ChurnInsight Dashboard</h1>
<div class="card">
  <div class="controls">
    <input type="file" id="file" accept=".csv" />
    <input type="number" id="umbral" step="0.01" placeholder="umbral opcional" />
    <button id="normalize">Normalizar y descargar</button>
    <button id="upload">Subir CSV y predecir</button>
    <label><input type="checkbox" id="strictTelco" /> Solo Telco</label>
  </div>
  <div class="controls">
    <label>Filtro plan:</label>
    <select id="planFilter"><option value="">Todos</option></select>
    <label>Filtro región:</label>
    <select id="regionFilter"><option value="">Todas</option></select>
  </div>
  <div class="controls">
    <label>CRM webhook:</label>
    <input type="url" id="crmUrl" placeholder="https://crm.example.com/webhook" style="min-width:320px" />
    <label><input type="checkbox" id="crmEnable" /> Enviar alertas si p≥umbral</label>
  </div>
  <div class="controls">
    <label>Paginación:</label>
    <input type="number" id="pageSize" value="200" min="50" step="50" />
    <button id="prevPage">Anterior</button>
    <span id="pageInfo"></span>
    <button id="nextPage">Siguiente</button>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="top1Explain" /> Explicación mínima (top-1)</label>
  </div>
  <div class="controls">
    <button id="clearHistory">Limpiar histórico</button>
  </div>
  <div id="summary"></div>
  <table id="results"><thead><tr>
    <th>ID</th><th>Plan</th><th>Región</th><th>Tipo cliente</th><th>Probabilidad</th><th>Previsión</th><th>Top factores</th>
  </tr></thead><tbody></tbody></table>
</div>

<div class="row">
  <div class="card" style="flex:1 1 400px">
    <h3>Promedio de probabilidad por plan</h3>
    <canvas id="chartPlan"></canvas>
  </div>
  <div class="card" style="flex:1 1 400px">
    <h3>Promedio de probabilidad por región</h3>
    <canvas id="chartRegion"></canvas>
  </div>
</div>
<div class="card" style="margin-top:20px">
  <h3>Tendencia de riesgo promedio (últimos uploads)</h3>
  <canvas id="chartTrend"></canvas>
</div>

<div class="card" style="margin-top:20px">
  <h3>Calibración (curva de fiabilidad)</h3>
  <div class="controls">
    <label>Bins:</label>
    <input type="number" id="calibBins" value="10" min="5" max="50" step="5" />
    <label>Segmento:</label>
    <select id="calibSegment"><option value="">Global</option></select>
    <button id="refreshCalib">Actualizar</button>
  </div>
  <canvas id="calibChart" height="220"></canvas>
  <div id="calibInfo"></div>
</div>

<div class="card" style="margin-top:20px">
  <h3>A/B testing de retención</h3>
  <div class="controls">
    <input type="text" id="abName" placeholder="Nombre experimento" />
    <input type="number" id="abRatio" value="0.5" step="0.05" min="0" max="1" />
    <button id="abCreate">Crear experimento</button>
  </div>
  <div class="controls">
    <select id="abSelect"><option value="">Selecciona experimento</option></select>
  </div>
  <div class="controls">
    <input type="file" id="abAssignFile" accept=".csv" />
    <button id="abAssign">Asignar control/tratamiento</button>
  </div>
  <div class="controls">
    <input type="file" id="abOutcomeFile" accept=".csv" />
    <button id="abOutcome">Subir outcomes</button>
    <button id="abReport">Generar reporte</button>
  </div>
  <div id="abMsg" style="margin-top:8px;color:#555"></div>
  <pre id="abReportView" style="white-space:pre-wrap"></pre>
</div>

<script>
let currentData = []
let page = 0
let pageSize = 200
function renderTable(){
  const planSel = document.getElementById('planFilter').value
  const regionSel = document.getElementById('regionFilter').value
  const tbody = document.querySelector('#results tbody')
  tbody.innerHTML = ''
  let filtered = currentData.filter(r => (!planSel || r.plan===planSel) && (!regionSel || r.region===regionSel))
  const total = filtered.length
  const maxPage = Math.max(0, Math.ceil(total / pageSize) - 1)
  if(page > maxPage) page = maxPage
  const start = page * pageSize
  const end = Math.min(start + pageSize, total)
  const pageData = filtered.slice(start, end)
  pageData.forEach(r => {
    const tr = document.createElement('tr')
    const expl = Array.isArray(r.explicacion)? r.explicacion.join(', ') : ''
    tr.innerHTML = `<td>${r.id}</td><td>${r.plan}</td><td>${r.region}</td><td>${r.tipo_cliente}</td><td>${r.probabilidad.toFixed(3)}</td><td>${r.prevision}</td><td>${expl}</td>`
    tbody.appendChild(tr)
  })
  const avg = filtered.length? filtered.reduce((a,b)=>a+b.probabilidad,0)/filtered.length : 0
  const churnRate = filtered.length? filtered.filter(x=>x.prevision==='Va a cancelar').length/filtered.length : 0
  document.getElementById('summary').innerText = `Registros: ${filtered.length} | Promedio p(churn): ${avg.toFixed(3)} | Tasa churn: ${churnRate.toFixed(3)}`
  document.getElementById('pageInfo').innerText = `Página ${Math.min(page+1, maxPage+1)} / ${maxPage+1} | Mostrando ${total? (start+1) : 0}-${end} de ${total}`
}
function updateFilters(){
  const plans = [...new Set(currentData.map(x=>x.plan))]
  const regions = [...new Set(currentData.map(x=>x.region))]
  const planSel = document.getElementById('planFilter')
  const regionSel = document.getElementById('regionFilter')
  planSel.innerHTML = '<option value="">Todos</option>' + plans.map(p=>`<option>${p}</option>`).join('')
  regionSel.innerHTML = '<option value="">Todas</option>' + regions.map(r=>`<option>${r}</option>`).join('')
}
let chartPlan, chartRegion, chartTrend
function renderCharts(){
  const byPlan = {}
  const byRegion = {}
  currentData.forEach(r=>{
    byPlan[r.plan] = (byPlan[r.plan]||[]).concat([r.probabilidad])
    byRegion[r.region] = (byRegion[r.region]||[]).concat([r.probabilidad])
  })
  const planLabels = Object.keys(byPlan)
  const planData = planLabels.map(k=> byPlan[k].reduce((a,b)=>a+b,0)/byPlan[k].length)
  const regionLabels = Object.keys(byRegion)
  const regionData = regionLabels.map(k=> byRegion[k].reduce((a,b)=>a+b,0)/byRegion[k].length)
  if(chartPlan) chartPlan.destroy()
  if(chartRegion) chartRegion.destroy()
  chartPlan = new Chart(document.getElementById('chartPlan'),{
    type:'bar',
    data:{labels:planLabels,datasets:[{label:'p(churn) promedio',data:planData,backgroundColor:'#4e79a7'}]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#111'}}},
      scales:{
        x:{ticks:{color:'#111'},grid:{color:'#eee'}},
        y:{beginAtZero:true,max:1,ticks:{color:'#111'},grid:{color:'#eee'}}
      }
    }
  })
  chartRegion = new Chart(document.getElementById('chartRegion'),{
    type:'bar',
    data:{labels:regionLabels,datasets:[{label:'p(churn) promedio',data:regionData,backgroundColor:'#f28e2b'}]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#111'}}},
      scales:{
        x:{ticks:{color:'#111'},grid:{color:'#eee'}},
        y:{beginAtZero:true,max:1,ticks:{color:'#111'},grid:{color:'#eee'}}
      }
    }
  })
}
async function loadCalibSegments(){
  try{
    const r = await fetch('/stats')
    if(!r.ok) return
    const s = await r.json()
    const segs = s.segmentos ? Object.keys(s.segmentos) : []
    const sel = document.getElementById('calibSegment')
    sel.innerHTML = '<option value="">Global</option>' + segs.map(k=>`<option>${k}</option>`).join('')
  }catch(e){}
}
async function refreshCalib(){
  const bins = Number(document.getElementById('calibBins').value)||10
  const seg = document.getElementById('calibSegment').value || ''
  const url = seg? `/calibration_report?bins=${encodeURIComponent(bins)}&segment=${encodeURIComponent(seg)}` : `/calibration_report?bins=${encodeURIComponent(bins)}`
  const r = await fetch(url)
  if(!r.ok) return
  const d = await r.json()
  const uncal = d.data.map(x=>({x:x.conf_uncal,y:x.freq_uncal}))
  const cal = d.data.map(x=>({x:x.conf_cal,y:x.freq_cal}))
  const ctx = document.getElementById('calibChart').getContext('2d')
  if(window.calibChart && typeof window.calibChart.destroy==='function') window.calibChart.destroy()
  if(window.calibChartInstance && typeof window.calibChartInstance.destroy==='function') window.calibChartInstance.destroy()
  window.calibChartInstance = new Chart(ctx,{
    type:'scatter',
    data:{datasets:[
      {label:'Uncalibrado',data:uncal,borderColor:'#d33',backgroundColor:'#d33',showLine:true},
      {label:'Calibrado',data:cal,borderColor:'#36c',backgroundColor:'#36c',showLine:true},
      {label:'Ideal',data:[{x:0,y:0},{x:1,y:1}],borderColor:'#777',backgroundColor:'#777',showLine:true}
    ]},
    options:{scales:{x:{min:0,max:1},y:{min:0,max:1}}}
  })
  window.calibChart = window.calibChartInstance
  document.getElementById('calibInfo').innerText = `Bins: ${d.bins} | ECE uncal: ${Number(d.ece_uncal).toFixed(4)} | ECE cal: ${Number(d.ece_cal).toFixed(4)}`
}
async function loadExperiments(){
  try{
    const r = await fetch('/ab/list')
    if(!r.ok) return
    const items = await r.json()
    const sel = document.getElementById('abSelect')
    sel.innerHTML = '<option value="">Selecciona experimento</option>' + items.map(x=>`<option value="${x.exp_id}">${x.nombre} (${x.exp_id})</option>`).join('')
  }catch(e){}
}
async function loadTrend(){
  try{
    const r = await fetch('/dashboard_history')
    if(!r.ok) return
    const hist = await r.json()
    let labels = hist.map(h=> new Date(h.ts).toLocaleTimeString())
    let data = hist.map(h=> h.avg)
    if((!data || data.length===0) && currentData && currentData.length){
      const avg = currentData.reduce((a,b)=> a + Number(b.probabilidad||0), 0) / currentData.length
      labels = [new Date().toLocaleTimeString()]
      data = [avg]
    }
    if(chartTrend) chartTrend.destroy()
    chartTrend = new Chart(document.getElementById('chartTrend'),{
      type:'line',
      data:{labels:labels,datasets:[{label:'p(churn) promedio',data:data,borderColor:'#59a14f',fill:false}]},
      options:{responsive:true,plugins:{legend:{labels:{color:'#111'}}},
        scales:{
          x:{ticks:{color:'#111'},grid:{color:'#eee'}},
          y:{beginAtZero:true,max:1,ticks:{color:'#111'},grid:{color:'#eee'}}
        }
      }
    })
  }catch(e){ /* noop */ }
}
document.getElementById('planFilter').addEventListener('change',renderTable)
document.getElementById('regionFilter').addEventListener('change',renderTable)
  document.getElementById('normalize').addEventListener('click', async ()=>{
    const f = document.getElementById('file').files[0]
    if(!f){ alert('Selecciona un CSV'); return }
    const fd = new FormData()
    fd.append('file', f)
    const strict = document.getElementById('strictTelco').checked ? 1 : 0
    const resp = await fetch(`/normalize_csv?strict_telco=${strict}`,{method:'POST',body:fd})
    if(!resp.ok){ const text = await resp.text(); alert(text || 'Error al normalizar CSV'); return }
    const blob = await resp.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'dataset_normalizado.csv'
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  })
  document.getElementById('upload').addEventListener('click', async ()=>{
    const f = document.getElementById('file').files[0]
    if(!f){ alert('Selecciona un CSV'); return }
    const fd = new FormData()
    fd.append('file', f)
    let url = '/predict_bulk'
    const u = document.getElementById('umbral').value
    if(u) url += `?umbral=${encodeURIComponent(u)}`
    const top1 = document.getElementById('top1Explain').checked
    url += (url.includes('?')? '&' : '?') + `explain_top=${top1?1:3}`
    const strict = document.getElementById('strictTelco').checked ? 1 : 0
    url += (url.includes('?')? '&' : '?') + `strict_telco=${strict}`
  const resp = await fetch(url,{method:'POST',body:fd})
  if(!resp.ok){
    const text = await resp.text()
    alert(text || 'Error en predicción masiva')
    return
  }
  currentData = await resp.json()
  updateFilters(); renderTable(); renderCharts(); loadTrend()
  sendAlerts()
})
document.getElementById('pageSize').addEventListener('change', ()=>{ const v = Number(document.getElementById('pageSize').value)||200; pageSize = Math.max(50, v); page = 0; renderTable() })
document.getElementById('prevPage').addEventListener('click', ()=>{ if(page>0){ page--; renderTable() } })
document.getElementById('nextPage').addEventListener('click', ()=>{ page++; renderTable() })
  document.getElementById('clearHistory').addEventListener('click', async ()=>{
  const ok = window.confirm('¿Seguro que quieres limpiar el histórico? Esta acción no se puede deshacer.')
  if(!ok) return
  try{ await fetch('/clear_history', {method:'POST'}) }catch(e){}
  currentData = []
  page = 0
  renderTable(); renderCharts(); loadTrend()
})
document.getElementById('refreshCalib').addEventListener('click', refreshCalib)
  document.getElementById('calibBins').addEventListener('change', refreshCalib)
  document.getElementById('calibSegment').addEventListener('change', refreshCalib)
document.getElementById('abCreate').addEventListener('click', async ()=>{
  const name = document.getElementById('abName').value.trim()
  const ratio = Number(document.getElementById('abRatio').value)||0.5
  if(!name){ alert('Nombre requerido'); return }
  const r = await fetch(`/ab/create?nombre=${encodeURIComponent(name)}&ratio_tratamiento=${encodeURIComponent(ratio)}`, {method:'POST'})
  if(!r.ok){ const t = await r.text(); alert(t||'Error al crear experimento'); return }
  document.getElementById('abMsg').textContent = 'Experimento creado'
  loadExperiments()
})
document.getElementById('abAssign').addEventListener('click', async ()=>{
  const exp = document.getElementById('abSelect').value
  const f = document.getElementById('abAssignFile').files[0]
  if(!exp){ alert('Selecciona experimento'); return }
  if(!f){ alert('Selecciona CSV de usuarios'); return }
  const fd = new FormData(); fd.append('file', f)
  const r = await fetch(`/ab/assign?exp_id=${encodeURIComponent(exp)}`, {method:'POST', body: fd})
  if(!r.ok){ const t = await r.text(); alert(t||'Error al asignar'); return }
  document.getElementById('abMsg').textContent = 'Asignaciones realizadas'
})
document.getElementById('abOutcome').addEventListener('click', async ()=>{
  const exp = document.getElementById('abSelect').value
  const f = document.getElementById('abOutcomeFile').files[0]
  if(!exp){ alert('Selecciona experimento'); return }
  if(!f){ alert('Selecciona CSV de outcomes'); return }
  const fd = new FormData(); fd.append('file', f)
  const r = await fetch(`/ab/outcomes?exp_id=${encodeURIComponent(exp)}`, {method:'POST', body: fd})
  if(!r.ok){ const t = await r.text(); alert(t||'Error al subir outcomes'); return }
  document.getElementById('abMsg').textContent = 'Outcomes cargados'
})
document.getElementById('abReport').addEventListener('click', async ()=>{
  const exp = document.getElementById('abSelect').value
  if(!exp){ alert('Selecciona experimento'); return }
  const r = await fetch(`/ab/report?exp_id=${encodeURIComponent(exp)}`)
  if(!r.ok){ const t = await r.text(); alert(t||'Error al generar reporte'); return }
  const d = await r.json()
  document.getElementById('abReportView').textContent = JSON.stringify(d.report, null, 2)
})
function sendAlerts(){
  const enabled = document.getElementById('crmEnable').checked
  const url = document.getElementById('crmUrl').value
  if(!enabled || !url) return
  const alerts = currentData.filter(x=> Number(x.probabilidad) >= Number(x.umbral)).map(x=>({
    id: x.id,
    probabilidad: x.probabilidad,
    umbral: x.umbral,
    plan: x.plan,
    region: x.region,
    tipo_cliente: x.tipo_cliente,
    explicacion: x.explicacion
  }))
  if(alerts.length===0) return
  try{ fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alerts})}) }catch(e){}
}
loadTrend(); loadCalibSegments(); loadExperiments(); refreshCalib()
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
    category=UserWarning
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)