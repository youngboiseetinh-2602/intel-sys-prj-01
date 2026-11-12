import unittest
from fastapi.testclient import TestClient
from app import app

SAMPLE = dict(gender="Female", age=54, hypertension=1, heart_disease=0,
              smoking_history="former", bmi=31.2, HbA1c_level=6.8,
              blood_glucose_level=185)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_interface(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")

    def test_invalid_inputs(self):
        for changes in ({"age": True}, {"age": 121}, {"gender": "unknown"},
                        {"hypertension": 0.5}, {"bmi": 0}, {"extra": 1}):
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post("/api/predict", json=SAMPLE | changes).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 422)

    def test_real_inference(self):
        response = self.client.post("/api/predict", json=SAMPLE)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(0 <= result["score"] <= 1)
        self.assertEqual(result["threshold"], 0.8370159268379211)
        self.assertEqual(result["predicted_label"], int(result["score"] >= result["threshold"]))
        self.assertEqual(self.client.post("/api/predict", json=SAMPLE).json(), result)


if __name__ == "__main__":
    unittest.main()
