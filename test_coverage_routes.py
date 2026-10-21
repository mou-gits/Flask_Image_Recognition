"""Requirement-focused Flask route tests for CSCN73010 Assignment 2.

Assignment requirement references:
- R1: The Flask home page can be accessed successfully.
- R2: A valid request produces an expected response.
- R3: A request with a missing uploaded file is handled appropriately.
- R4: Invalid or unexpected input is handled appropriately.
- R6: The suite includes both normal cases and edge cases.

These tests exercise the Flask request/response layer. Prediction and preprocessing
are mocked where the goal is route behavior rather than image-processing behavior.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import app as app_module
import model as model_module


@pytest.fixture
def client():
    """Create a Flask test client for exercising the app without starting a server."""
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


def make_png_bytes(color=(20, 40, 60), size=(32, 24)):
    """Build a genuine in-memory PNG so valid-image tests do not rely on fake bytes."""
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def post_image(client, filename="sample.png", content=None):
    """Post image bytes to /prediction using the multipart upload format expected by Flask."""
    payload = make_png_bytes() if content is None else content
    return client.post(
        "/prediction",
        data={"file": (BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def test_r1_home_page_returns_successful_html(client):
    """R1 — Home page: GET / must return HTTP 200 and an HTML response.

    This directly checks the assignment requirement that the Flask home page can
    be accessed successfully. The assertions check status and response type,
    rather than merely checking that some bytes were returned.
    """
    response = client.get("/")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert response.data.strip()


def test_r2_valid_upload_calls_preprocessing_and_prediction(client, monkeypatch):
    """R2 — Valid request: a valid PNG upload must reach preprocessing and prediction.

    The test uses a real PNG but mocks expensive processing/inference. It verifies
    the route passes the prepared tensor to prediction and renders the returned
    prediction. This is route-level evidence, not a test of the ML model itself.
    """
    calls = {}

    def fake_preprocess(stream):
        calls["stream"] = stream
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

    def fake_predict(prepared_image):
        calls["prepared"] = prepared_image
        return 9

    monkeypatch.setattr(app_module, "preprocess_img", fake_preprocess)
    monkeypatch.setattr(app_module, "predict_result", fake_predict)

    response = post_image(client)

    assert response.status_code == 200
    assert "stream" in calls
    assert calls["prepared"].shape == (1, 224, 224, 3)
    assert b"9" in response.data


def test_r3_missing_file_field_returns_controlled_error(client):
    """R3 — Missing upload: a POST without the required 'file' field is handled.

    The test checks the existing application contract: the route returns its
    rendered error page with HTTP 200 and the user-facing processing message.
    """
    response = client.post("/prediction", data={}, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_r4_empty_filename_is_handled(client, monkeypatch):
    """R4/R6 — Unexpected upload metadata: an empty filename must not cause an unhandled error.

    This edge case differs from omitting the 'file' field entirely. Preprocessing
    is mocked to raise a normal input error so this test isolates route handling.
    """
    def fail_preprocessing(stream):
        raise ValueError("empty upload")

    monkeypatch.setattr(app_module, "preprocess_img", fail_preprocessing)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b""), "")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_r4_invalid_image_bytes_return_controlled_error(client, monkeypatch):
    """R4 — Invalid input: non-image bytes must be rejected with the controlled error page.

    Unlike a mocked failure, this test calls the real preprocessing function so
    Pillow must inspect the bytes and reject the invalid image payload.
    """
    monkeypatch.setattr(app_module, "preprocess_img", model_module.preprocess_img)

    response = post_image(
        client,
        filename="not-really-an-image.png",
        content=b"this is not a real image",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_r4_prediction_runtime_error_returns_controlled_error(client, monkeypatch):
    """R4/R6 — Unexpected inference failure: the route must convert RuntimeError to a controlled response.

    This is the regression test that previously exposed an HTTP 500. Preprocessing
    is mocked to succeed, prediction is mocked to fail, and the test checks that
    the route handles the failure rather than leaking an unhandled server error.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")

    def fail_prediction(prepared):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app_module, "predict_result", fail_prediction)

    response = post_image(client)

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
