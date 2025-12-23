import os
import sys
import subprocess
import importlib.util
from fastapi.testclient import TestClient

def ensure_model():
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data-science', 'models', 'model.joblib'))
    if not os.path.exists(model_path):
        script = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data-science', 'src', 'train_model.py'))
        subprocess.run([sys.executable, script], check=True)

ensure_model()

repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
app_path = os.path.join(repo_root, 'api-gateway', 'app.py')
spec = importlib.util.spec_from_file_location('app', app_path)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
client = TestClient(app_module.app)

def test_predict_ok():
    payload = {
        'tiempo_contrato_meses': 12,
        'retrasos_pago': 2,
        'uso_mensual': 14.5,
        'plan': 'Premium',
        'nps': 7,
        'quejas': 1,
        'canal_contacto': 'app',
        'interacciones_soporte': 2,
        'tipo_pago': 'debito_automatico',
        'region': 'norte',
        'tipo_cliente': 'nuevo'
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'prevision' in data and 'probabilidad' in data and 'explicacion' in data
    assert 0.0 <= data['probabilidad'] <= 1.0
    assert isinstance(data['explicacion'], list) and len(data['explicacion']) == 3

def test_stats():
    r = client.get('/stats')
    assert r.status_code == 200
    data = r.json()
    assert 'total_evaluados' in data and 'tasa_churn' in data

def test_validation_error():
    payload = {
        'tiempo_contrato_meses': 12,
        'retrasos_pago': 2,
        'plan': 'Premium',
        'nps': 7,
        'quejas': 1,
        'canal_contacto': 'app',
        'interacciones_soporte': 2,
        'tipo_pago': 'debito_automatico',
        'region': 'norte',
        'tipo_cliente': 'nuevo'
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 422

def test_calibrate_threshold_and_predict_uses_default():
    r = client.post('/calibrate_threshold', params={'modo': 'f1'})
    assert r.status_code == 200
    t = r.json()['umbral']
    payload = {
        'tiempo_contrato_meses': 6,
        'retrasos_pago': 0,
        'uso_mensual': 10.0,
        'plan': 'Basico',
        'nps': 5,
        'quejas': 0,
        'canal_contacto': 'web',
        'interacciones_soporte': 1,
        'tipo_pago': 'tarjeta',
        'region': 'norte',
        'tipo_cliente': 'nuevo'
    }
    r2 = client.post('/predict', json=payload)
    assert r2.status_code == 200
    data = r2.json()
    assert 'umbral' in data and 0.0 <= data['umbral'] <= 1.0

def test_retrain_endpoint_and_predict():
    r = client.post('/retrain')
    assert r.status_code == 200
    m = r.json()
    assert 'status' in m and m['status'] == 'ok'
    payload = {
        'tiempo_contrato_meses': 8,
        'retrasos_pago': 1,
        'uso_mensual': 12.5,
        'plan': 'Estandar',
        'nps': 6,
        'quejas': 1,
        'canal_contacto': 'web',
        'interacciones_soporte': 1,
        'tipo_pago': 'tarjeta',
        'region': 'sur',
        'tipo_cliente': 'antiguo'
    }
    r2 = client.post('/predict', json=payload)
    assert r2.status_code == 200

def test_segmentation_by_plan():
    os.environ['SEGMENTATION_KEY'] = 'plan'
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    app_path = os.path.join(repo_root, 'api-gateway', 'app.py')
    spec = importlib.util.spec_from_file_location('app_seg', app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    client2 = TestClient(app_module.app)
    client2.post('/retrain')
    payload = {
        'tiempo_contrato_meses': 10,
        'retrasos_pago': 1,
        'uso_mensual': 12.0,
        'plan': 'Basico',
        'nps': 5,
        'quejas': 1,
        'canal_contacto': 'web',
        'interacciones_soporte': 1,
        'tipo_pago': 'tarjeta',
        'region': 'este',
        'tipo_cliente': 'corporativo'
    }
    r = client2.post('/predict', json=payload)
    assert r.status_code == 200
    data = r.json()
    if 'segmento' in data:
        assert data['segmento']['key'] == 'plan' and data['segmento']['value'] in ['Basico','Estandar','Premium','Basic','Standard','Premium']

def test_segmentation_by_region():
    os.environ['SEGMENTATION_KEY'] = 'region'
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    app_path = os.path.join(repo_root, 'api-gateway', 'app.py')
    spec = importlib.util.spec_from_file_location('app_seg2', app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    client3 = TestClient(app_module.app)
    client3.post('/retrain')
    payload = {
        'tiempo_contrato_meses': 10,
        'retrasos_pago': 1,
        'uso_mensual': 12.0,
        'plan': 'Basico',
        'nps': 5,
        'quejas': 1,
        'canal_contacto': 'web',
        'interacciones_soporte': 1,
        'tipo_pago': 'tarjeta',
        'region': 'norte',
        'tipo_cliente': 'nuevo'
    }
    r = client3.post('/predict', json=payload)
    assert r.status_code == 200
    data = r.json()
    if 'segmento' in data:
        assert data['segmento']['key'] == 'region' and data['segmento']['value'] in ['norte','sur','este','oeste']

def test_predict_bulk_csv():
    import io
    csv = io.StringIO()
    csv.write('plan,tiempo_contrato_meses,retrasos_pago,uso_mensual,nps,quejas,canal_contacto,interacciones_soporte,tipo_pago,region,tipo_cliente\n')
    csv.write('Basico,10,1,12.0,5,1,web,1,tarjeta,norte,nuevo\n')
    csv.write('Estandar,8,0,18.5,8,0,app,1,debito_automatico,este,corporativo\n')
    body = csv.getvalue()
    files = {'file': ('bulk.csv', body, 'text/csv')}
    r = client.post('/predict_bulk', files=files)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2
    assert 'prevision' in data[0] and 'probabilidad' in data[0]
    data = r.json()
    if 'segmento' in data:
        assert data['segmento']['key'] == 'plan' and data['segmento']['value'] in ['Basico','Estandar','Premium','Basic','Standard','Premium']