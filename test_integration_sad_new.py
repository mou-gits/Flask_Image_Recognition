"""Improved integration tests for rejected prediction requests."""
from io import BytesIO
import app as app_module


def test_missing_file_new(client):
    """Improve `test_missing_file` in `test_integration_sad.py`.

    Reuses conftest.py's shared client fixture rather than duplicating it. Checks
    that an absent upload is handled with the established user-facing error.
    Keep this integration-labelled test only if both classifications are required.
    """
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data


def test_prediction_runtime_error_new(client, monkeypatch):
    """Add the report-described runtime-error regression test; no equivalent was
    present in the uploaded original test files.

    Preprocessing is mocked to succeed and prediction raises RuntimeError. The
    route should convert this into its controlled error response rather than an
    unhandled HTTP 500. If this fails, verify and fix the actual route behavior;
    do not weaken the assertion just to make the suite green.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    def fail_prediction(prepared):
        raise RuntimeError("simulated model runtime failure")
    monkeypatch.setattr(app_module, "predict_result", fail_prediction)
    image = BytesIO(b"preprocessing is mocked"); image.name = "runtime-error.jpg"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
