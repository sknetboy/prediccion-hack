# ChurnInsight — Guía sencilla para usar el modelo

## ¿Qué es?
- ChurnInsight ayuda a anticipar qué clientes podrían cancelar un servicio.
- Devuelve dos datos:
  - `prevision`: "Va a cancelar" o "Va a continuar".
  - `probabilidad`: un número entre 0 y 1 que indica cuán seguro es el resultado.

## ¿Para qué sirve?
- Priorizar a quién contactar primero para evitar cancelaciones.
- Ofrecer beneficios o soporte a las personas con mayor riesgo.
- Medir si las acciones de retención están funcionando.

## ¿Qué información necesito?
- `tiempo_contrato_meses`: cuántos meses lleva el cliente.
- `retrasos_pago`: cuántas veces se atrasó en pagar.
- `uso_mensual`: cuánto usa el servicio en promedio por mes.
- `plan`: tipo de plan (por ejemplo: "Basic", "Standard", "Premium").

## ¿Cómo obtengo una predicción?
- Pide al equipo de tecnología la URL de la API (normalmente `http://127.0.0.1:8080/predict` o una URL interna de la empresa).
- Envía los datos del cliente en formato JSON. Ejemplo:
```
{
  "tiempo_contrato_meses": 12,
  "retrasos_pago": 2,
  "uso_mensual": 14.5,
  "plan": "Premium"
}
```
- La respuesta será:
```
{
  "prevision": "Va a cancelar",
  "probabilidad": 0.81
}
```
- Si usas Postman:
  - Método: `POST`
  - URL: `http://127.0.0.1:8080/predict`
  - Body: tipo `raw`, formato `JSON`, pega el ejemplo de arriba.

## ¿Cómo interpreto los resultados?
- `prevision`:
  - "Va a cancelar": el cliente está en riesgo.
  - "Va a continuar": bajo riesgo.
- `probabilidad`:
  - 0.80 o más: riesgo muy alto.
  - 0.60–0.79: riesgo alto.
  - 0.40–0.59: riesgo medio.
  - menos de 0.40: riesgo bajo.
- Sugerencia: adapta los umbrales a tu negocio (por ejemplo, actuar a partir de 0.60).

## Casos típicos de uso
- Campañas de retención: ordenar lista de clientes por `probabilidad` y contactar primero a los de mayor riesgo.
- Bonos o beneficios: ofrecer upgrade de plan o descuento a quienes marcan riesgo alto.
- Soporte proactivo: priorizar seguimiento de clientes con atrasos de pago y poco uso.

## Buenas prácticas
- Revisa que los datos estén actualizados (los resultados dependen de la calidad de los datos).
- Define reglas claras de actuación: qué hacer con riesgo alto, medio y bajo.
- Mide el impacto: compara la tasa de cancelación antes y después de las acciones.
- Respeta privacidad y normas internas: usa los datos solo para retención y mejora del servicio.

## ¿Con quién hablo si necesito ayuda?
- Equipo de Tecnología (Back-end): para temas de acceso a la API o conexión.
- Equipo de Data Science: para dudas sobre el modelo y su interpretación.
- Equipo de Marketing/Operaciones: para ejecutar campañas y medir resultados.

## Cómo ejecutar (local y con Docker)

### Local (sin Docker)

- Requisitos:
  - `Python 3.13+`
  - `Java 21+` (para la API Java)
  - `Maven 3.9.12+` (para arrancar la API Java)

- Preparar el entorno Python:
  1. Crear entorno virtual e instalar dependencias:
     ```
     python -m venv .venv
     .\.venv\Scripts\python -m pip install -U pip
     .\.venv\Scripts\pip install -r data-science/requirements.txt
     .\.venv\Scripts\pip install -r api-gateway/requirements.txt
     ```
  2. Entrenar el modelo:
     ```
     .\.venv\Scripts\python data-science\src\train_model.py
     ```
  3. Iniciar el Gateway (FastAPI):
     ```
     $env:AUTO_RETRAIN='1'; $env:DRIFT_THRESHOLD='0.25'; $env:CHURN_THRESHOLD='0.5'; 
     .\.venv\Scripts\python -m uvicorn api-gateway.app:app --host 127.0.0.1 --port 8001
     ```
     - Dashboard: abrir `http://127.0.0.1:8001/dashboard`
     - Predicción masiva (CSV): `POST /predict_bulk`
     - Normalización de CSV: `POST /normalize_csv`
     - Estado de drift: `GET /drift_status`

- Iniciar la API Java (Spring Boot):
  - Si tienes Maven en PATH, dentro de `api-java/`:
    ```
    mvn spring-boot:run
    ```
  - Si no tienes Maven en PATH, puedes usar:
    ```
    C:\Tools\apache-maven-3.9.12\bin\mvn.cmd spring-boot:run
    ```
  - La API Java corre en `http://127.0.0.1:8080` y reexpone `POST /predict` y `POST /predict-bulk`.

- Prueba rápida (JSON):
  - Con Postman o curl hacia `http://127.0.0.1:8080/predict` usando el ejemplo del apartado “¿Cómo obtengo una predicción?”.

- Prueba rápida (CSV):
  - Enviar un CSV al Gateway:
    ```
    curl -X POST "http://127.0.0.1:8001/predict_bulk?umbral=0.6&explain_top=3&strict_telco=0" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@C:\\Users\\Kabie\\Desktop\\Data\\proyecto-data\\shared\\data\\dataset.csv"
    ```
  - Enviar un CSV a la API Java:
    ```
    curl -X POST http://127.0.0.1:8080/predict-bulk \
      -H "Content-Type: multipart/form-data" \
      -F "file=@C:\\Users\\Kabie\\Desktop\\Data\\proyecto-data\\shared\\data\\dataset.csv"
    ```

### Docker (opcional)

- Requisitos:
  - `Docker Desktop` instalado y corriendo.

- Arrancar todo con Docker Compose:
  ```
  docker compose up --build
  ```
  - Gateway en `http://127.0.0.1:8001`
  - API Java en `http://127.0.0.1:8080`
  - Dashboard en `http://127.0.0.1:8001/dashboard`

### “Solo Telco” (modo estricto)

- En el Dashboard hay una casilla “Solo Telco”. Al activarla, el sistema solo acepta CSVs con el esquema del dataset original de telecomunicaciones.
- Si subes CSVs de otras fuentes, deja esta casilla desactivada y usa “Normalizar y descargar” para que las columnas se adapten automáticamente.

### CSV flexible y sinónimos
- Puedes subir CSVs que no sigan el esquema Telco; el sistema los ajusta automáticamente.
- Reglas principales:
  - Debe contener al menos 3 de 4 campos clave: `plan`, `tiempo_contrato_meses`, `retrasos_pago`, `uso_mensual`.
  - Si faltan campos secundarios, se completan con valores por defecto razonables.
  - Se reconocen sinónimos comunes:
    - `Subscription Length (Months)` → `tiempo_contrato_meses`
    - `Daily Watch Time (Hours)` → `uso_mensual`
    - Variantes de pago (`card`, `credit_card`) → `tipo_pago = tarjeta`
- Para adaptar tu CSV, usa el botón del Dashboard “Normalizar y descargar” o el endpoint `POST /normalize_csv` con `strict_telco=0`.

### Auto‑reentrenamiento por drift
- El sistema detecta cambios de distribución en los datos (drift) y puede reentrenar automáticamente.
- Activación: definir `AUTO_RETRAIN=1` y un umbral `DRIFT_THRESHOLD` (por ejemplo `0.25`) antes de iniciar el Gateway.
- El entrenamiento guarda un `baseline.json` que sirve de referencia para calcular el drift.
- Consulta de estado: `GET /drift_status` desde el Gateway.

### A/B testing de retención
- Permite medir el impacto de campañas (control vs tratamiento).
- Flujo básico (también disponible desde el Dashboard):
  - Crear experimento: `POST /ab/create` (ej. nombre y ratio de tratamiento).
  - Asignar usuarios: `POST /ab/assign` con un CSV que incluya `id`.
  - Subir resultados: `POST /ab/outcomes` con CSV (`id` y `churn`/`retained`/`revenue`).
  - Ver reporte: `GET /ab/report` (incluye uplift, z‑score, p‑value).
  - Listar: `GET /ab/list`.
- Persistencia: se guarda en `data-science/models/ab_experiments.json`.

### Ejecutar Data Science sin entorno virtual

- Si prefieres usar tu instalación de Python sin venv:
  1. Instala dependencias en tu usuario:
     ```
     python -m pip install --user -r data-science/requirements.txt
     ```
  2. Opcional: define variables de entorno (persistentes) para entrenar con tu CSV y ajustar calibración/segmentación:
     ```
     setx TRAIN_DATA_PATH "C:\\ruta\\a\\tu.csv"
     setx CALIBRATION_METHOD "isotonic"
     setx SEGMENTATION_KEY "region"
     ```
     - O solo para la sesión actual de PowerShell:
     ```
     $env:TRAIN_DATA_PATH = "C:\\ruta\\a\\tu.csv"
     $env:CALIBRATION_METHOD = "sigmoid"
     $env:SEGMENTATION_KEY = "plan"
     ```
  3. Ejecuta el entrenamiento:
     ```
     python data-science\src\train_model.py
     ```
  4. Revisa salidas en `data-science\models\` (`model.joblib`, `metrics.json`, `calibration.json`, `baseline.json`).

## Sugerencias de mejora (si te interesa)
- Añadir más variables: historial de contacto, tipo de canal, quejas, NPS.
- Explicabilidad: mostrar las 3 variables que más influyen en cada predicción.
- Ajustar el umbral: calibrar cuándo considerar "Va a cancelar" según costos/beneficios.
- Reentrenamiento periódico: actualizar el modelo cada mes o trimestre.
- Segmentación: usar modelos distintos por tipo de plan o región.
- Batch prediction: subir un archivo CSV con muchos clientes y obtener resultados masivos.
- Dashboard visual: ver clientes con mayor riesgo y tendencias en el tiempo.
- Notificaciones: alertas automáticas a CRM cuando alguien supera cierto riesgo.
- Integración con BI: enviar resultados a herramientas internas (ej. Power BI, Metabase).

## Ejemplos rápidos
- Cliente A: 4 meses, 3 retrasos, poco uso, plan Standard → riesgo alto.
- Cliente B: 18 meses, 0 retrasos, uso frecuente, plan Premium → riesgo bajo.
- Cliente C: 7 meses, 1 retraso, uso medio, plan Basic → riesgo medio.

---
Este documento está pensado para ayudarte a usar el modelo sin conocimientos técnicos. Si necesitas una interfaz más simple (formulario web), el equipo puede habilitarla sobre esta misma API.
