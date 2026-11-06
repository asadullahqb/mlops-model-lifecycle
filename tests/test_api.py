from fastapi.testclient import TestClient

from src import train
from src.serve import create_app


def test_health_and_predict(cfg):
    meta = train.train(cfg)  # promote v1
    client = TestClient(create_app(cfg))

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    features = [0.5] * 10
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float)
    assert body["version"] == meta["version"]


def test_predict_returns_new_version_after_promotion(cfg):
    meta = train.train(cfg)
    client = TestClient(create_app(cfg))

    # promote a second version on top of the first
    meta2 = train.train(cfg)
    assert meta2["version"] != meta["version"]

    r = client.post("/predict", json={"features": [0.5] * 10})
    assert r.json()["version"] == meta2["version"]
