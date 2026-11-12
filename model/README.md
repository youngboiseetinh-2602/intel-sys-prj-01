# ANN model for a new app

Files required together:

1. `diabetes_model.keras` - Keras Sequential ANN (15 transformed input units)
2. `preprocessor.joblib` - fitted StandardScaler + OneHotEncoder, binary passthrough
3. `features.json` - 8 raw features in the expected order
4. `model_metadata.json` - threshold, validation-selection policy, model metrics
5. `predict.py` - utility implementing preprocessing + `.keras` inference

Use Python >=3.11 with `pip install -r model/requirements-inference.txt` from the project root. In a server, import `from model.predict import predict` and pass a dictionary with eight fields; use `PatientDataAgent` from `backend/agents.py` for strict validation. Never use raw 8-column input directly on the network: the input layer expects 15 preprocessed columns. Scores in [0,1] are **not calibrated individual disease probabilities**.

Example (from project root):

```bash
python model/predict.py --json '{"gender":"Female","age":54,"hypertension":1,"heart_disease":0,"smoking_history":"never","bmi":31.2,"HbA1c_level":6.8,"blood_glucose_level":185}'
```

If you build a Node.js or Java backend, expose this Python inference as an HTTP endpoint, or export and validate an equivalent model pipeline. Do **not** load `.keras` directly as a JavaScript model and assume it works without conversion/testing. The original source ZIP's separate browser model.json is based on a **different training run**; do not mix its weights with this newly trained `preprocessor.joblib` or threshold.


## Model transfer to another app

Do not copy the `.keras` file alone. Copy it **together** with `preprocessor.joblib`, `features.json`, `model_metadata.json` and `predict.py`; keep the same column names/order and validation-selected threshold. Example: `from model.predict import predict`. Install the backend dependencies in `requirements-inference.txt`.

Synthetic sample checked on this package: Female, 54 years, hypertension=1, heart_disease=0, never smoking, BMI=31.2, HbA1c=6.8, glucose=185. The sigmoid score is approximately **0.9383**, selected threshold **0.837016**, hence the experimental label is **1**. This is neither calibrated individual disease risk nor a diagnosis.

The model was re-trained on the uploaded dataset in Keras 3.13.2 / Torch 2.10.0 / scikit-learn 1.8.0. Treat loading the `joblib` file as unsafe for untrusted packages; only load this file if it comes from a trusted source. Run a fixed synthetic sample test after changing package versions.
