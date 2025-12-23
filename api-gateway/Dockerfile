FROM python:3.13-slim
WORKDIR /app
COPY api-gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY api-gateway /app/api-gateway
COPY data-science /app/data-science
RUN python /app/data-science/src/train_model.py
EXPOSE 8001
CMD ["uvicorn", "api-gateway.app:app", "--host", "0.0.0.0", "--port", "8001"]