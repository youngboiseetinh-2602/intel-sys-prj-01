"""Reusable ANN inference. Educational model, NOT a medical diagnosis.
Run: python model/predict.py --json '{"gender":"Female",...}'
"""
import os
os.environ.setdefault('KERAS_BACKEND','torch')
from pathlib import Path
import argparse,json
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

BASE=Path(__file__).resolve().parent
FEATURES=json.loads((BASE/'features.json').read_text(encoding='utf-8'))
METADATA=json.loads((BASE/'model_metadata.json').read_text(encoding='utf-8'))
MODEL=None; PREPROCESSOR=None

def _resources():
    global MODEL, PREPROCESSOR
    if MODEL is None:
        MODEL=load_model(BASE/'diabetes_model.keras',compile=False)
        PREPROCESSOR=joblib.load(BASE/'preprocessor.joblib')
    return MODEL,PREPROCESSOR

def predict(patient: dict) -> dict:
    """Run the *fitted* preprocessing pipeline then predict a research label.

    Raises ValueError for malformed or out-of-domain inputs. This validation is
    duplicated intentionally in this standalone model package, which is also
    usable without the FastAPI application's PatientDataAgent.
    """
    if not isinstance(patient, dict):
        raise ValueError("Patient input must be an object")
    missing = [name for name in FEATURES if name not in patient]
    extra = sorted(set(patient) - set(FEATURES))
    if missing: raise ValueError("Missing input fields: " + ", ".join(missing))
    if extra: raise ValueError("Unknown input fields: " + ", ".join(extra))
    for name in ("gender", "smoking_history"):
        if not isinstance(patient[name], str) or patient[name] not in METADATA["input_domains"][name]:
            raise ValueError("Unsupported " + name + " option")
    for name in ("hypertension", "heart_disease"):
        if type(patient[name]) is not int or patient[name] not in (0, 1):
            raise ValueError(f"{name} must equal integer 0 or 1")
    for name, low, high in (("age", 0, 120), ("bmi", 10, 80),
                            ("HbA1c_level", 2, 20), ("blood_glucose_level", 40, 600)):
        val = patient[name]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(float(val)) or not low <= val <= high:
            raise ValueError(f"{name} must be a finite number within [{low}, {high}]")
    model, preprocessor = _resources()
    row = pd.DataFrame([{key: patient[key] for key in FEATURES}], columns=FEATURES)
    X = np.asarray(preprocessor.transform(row), dtype=np.float32)
    score = float(model.predict(X, verbose=0).reshape(-1)[0])
    threshold = float(METADATA["threshold"])
    return {"score": score, "threshold": threshold, "predicted_label": int(score >= threshold),
            "model_version": "ann-v1-2026-11",
            "note": "Research classification on a public dataset; not a diagnosis or calibrated personal risk."}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--json',type=str,required=True)
    args=parser.parse_args();print(json.dumps(predict(json.loads(args.json)),ensure_ascii=False,indent=2))
