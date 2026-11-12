"""Serve the diabetes research classifier and its browser interface."""
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from model.predict import predict

app = FastAPI(title="Đường An · Diabetes Research", docs_url="/api/docs")
inference_lock = Lock()


@app.get("/api/health")
def health():
    return {"status": "ok", "model_version": "ann-v1-2026-11"}


@app.post("/api/predict")
def classify(patient: dict):
    try:
        with inference_lock:
            return predict(patient)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web")
