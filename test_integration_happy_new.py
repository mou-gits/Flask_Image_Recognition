"""Improved integration tests for the prediction route."""
from io import BytesIO
from PIL import Image
import app as app_module


def test_successful_prediction_new(client, monkeypatch):
    """Improve `test_successful_prediction` in `test_integration_happy.py`.

    The old test uploaded arbitrary bytes and checked only status and a keyword.
    This version uses a real JPEG, verifies the route passes a prepared value
    between preprocessing and prediction, and checks the returned prediction.
    Model inference remains mocked so this tests the route integration boundary.
    """
    stream = BytesIO()
    Image.new("RGB", (48, 48), (30, 100, 200)).save(stream, format="JPEG")
    stream.seek(0); stream.name = "integration.jpg"
    observed = {}
    monkeypatch.setattr(app_module, "preprocess_img", lambda upload: observed.setdefault("preprocessed", True) and {"prepared": True})
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: observed.setdefault("prepared", prepared) and 9)
    response = client.post("/prediction", data={"file": (stream, stream.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"9" in response.data
    assert observed["prepared"] == {"prepared": True}
