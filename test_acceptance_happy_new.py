"""Improved acceptance tests for successful image uploads."""
from io import BytesIO
from PIL import Image
import app as app_module


def image_stream(fmt="JPEG", size=(32, 32)):
    """Create a real in-memory image for upload tests."""
    stream = BytesIO()
    Image.new("RGB", size, (120, 40, 200)).save(stream, format=fmt)
    stream.seek(0)
    return stream


def test_acceptance_successful_upload_new(client, monkeypatch):
    """Improve `test_acceptance_successful_upload` in `test_acceptance_happy.py`.

    The old test uploaded arbitrary bytes labelled as JPEG. This version creates
    a real JPEG, mocks expensive inference, verifies preprocessing and prediction
    are invoked, and checks the rendered prediction response.
    """
    calls = []
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: calls.append("preprocess") or "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: calls.append("predict") or 7)
    image = image_stream()
    image.name = "valid.jpg"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"7" in response.data
    assert calls == ["preprocess", "predict"]


def test_acceptance_valid_large_image_new(client, monkeypatch):
    """Improve `test_acceptance_valid_large_image` in `test_acceptance_happy.py`.

    The old test repeated arbitrary bytes instead of creating a large image. This
    version uploads a valid 1200-by-1200 JPEG. Inference is mocked; this checks
    route handling, not a file-size limit or model performance.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 3)
    image = image_stream(size=(1200, 1200))
    image.name = "large-valid.jpg"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"3" in response.data


def test_acceptance_valid_image_size_upload_new(client, monkeypatch):
    """Improve `test_acceptance_valid_image_size_upload` in `test_acceptance_happy.py`.

    The old test did not define image dimensions. This version uploads a genuine
    640-by-480 PNG and checks successful route behavior, without claiming to test
    resource limits. Preprocessing and inference are mocked for isolation.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 5)
    image = image_stream("PNG", (640, 480))
    image.name = "640x480.png"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"5" in response.data
