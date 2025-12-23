# Guía de ejecución en distintos editores

## Requisitos
- `Python 3.13+`
- `Java 21+`
- `Maven` (para API Java)
- `Docker Desktop` (opcional, para Compose)

## Variables de entorno útiles
- `AUTO_RETRAIN=1` activa reentrenamiento automático por drift en el gateway.
- `DRIFT_THRESHOLD=0.25` fija el umbral de drift.
- `CHURN_THRESHOLD=0.5` umbral de decisión por defecto.
- `TRAIN_DATA_PATH` o `DATASET_PATH` para entrenar con tu CSV.
- `CALIBRATION_METHOD=isotonic|sigmoid` para calibración.
- `SEGMENTATION_KEY=plan|region|tipo_cliente` para modelos por segmento.

## Pasos comunes (terminal)
- Entrenar modelo:
```
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r data-science/requirements.txt
.\.venv\Scripts\pip install -r api-gateway/requirements.txt
.\.venv\Scripts\python data-science\src\train_model.py
```
- Lanzar gateway (FastAPI):
```
$env:AUTO_RETRAIN='1'; $env:DRIFT_THRESHOLD='0.25'; $env:CHURN_THRESHOLD='0.5'
.\.venv\Scripts\python -m uvicorn api-gateway.app:app --host 127.0.0.1 --port 8001
```
- Abrir dashboard: `http://127.0.0.1:8001/dashboard`
- Lanzar API Java:
```
mvn spring-boot:run
# o
C:\Tools\apache-maven-3.9.6\bin\mvn.cmd spring-boot:run
```

## VS Code
- Abrir el folder del proyecto.
- Extensión Python: seleccionar intérprete `\.venv\Scripts\python.exe`.
- Terminal integrado:
  - Entrenar: `.\.venv\Scripts\python data-science\src\train_model.py`
  - Gateway: ejecutar comando uvicorn anterior.
  - API Java: `mvn spring-boot:run` en `api-java/`.
- Debug (opcional): crear `launch.json` para `module: uvicorn` con args `api-gateway.app:app --host 127.0.0.1 --port 8001`.

## PyCharm
- Abrir el proyecto.
- Configurar intérprete usando `\.venv`.
- Run/Debug Configurations:
  - Script: `data-science/src/train_model.py`.
  - Module: `uvicorn` con parámetros `api-gateway.app:app --host 127.0.0.1 --port 8001`.
- Lanzar `mvn spring-boot:run` en una terminal para la API Java.

## IntelliJ IDEA (API Java)
- Abrir `api-java/` como proyecto Maven.
- JDK: seleccionar `Java 17`.
- Run: `spring-boot:run` desde Maven o “Run Application”.
- Para el gateway, usar terminal con los comandos de Python o el plugin Python.

## Eclipse (Maven)
- Importar `api-java/` como “Existing Maven Project”.
- JRE: `Java 17`.
- Run: “Spring Boot App” o `mvn spring-boot:run` en la consola.
- Gateway y entrenamiento: terminal externa con los comandos de Python.

## Visual Studio (Windows)
- Si usas la workload de Python, selecciona `\.venv` como entorno.
- Ejecuta `train_model.py` y el comando de uvicorn desde la Terminal.
- API Java: usar `mvn spring-boot:run` en el directorio `api-java`.

## Docker Compose (opcional)
```
docker compose up --build
```
- Gateway: `http://127.0.0.1:8001`
- API Java: `http://127.0.0.1:8080`
- Dashboard: `http://127.0.0.1:8001/dashboard`

## Pruebas rápidas
- JSON (API Java):
```
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{"tiempo_contrato_meses":12,"retrasos_pago":2,"uso_mensual":14.5,"plan":"Premium"}'
```
- CSV (Gateway):
```
curl -X POST "http://127.0.0.1:8001/predict_bulk?umbral=0.6&explain_top=3&strict_telco=0" -H "Content-Type: multipart/form-data" -F "file=@C:\\Users\\Kabie\\Desktop\\Data\\proyecto-data\\shared\\data\\dataset.csv"
```

## Problemas comunes
- `net::ERR_CONNECTION_REFUSED`: el gateway no está corriendo o el puerto está bloqueado. Relanza uvicorn y espera “Application startup complete”.
- `favicon.ico 404`: resuelto, el gateway expone un favicon mínimo.
- Puertos en uso: cambia `--port` o detén procesos que usen `8001/8080`.
