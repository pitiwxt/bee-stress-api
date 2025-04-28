# ====== Dockerfile (Final) ======

FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn tensorflow scikit-learn pandas numpy joblib imbalanced-learn

ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
