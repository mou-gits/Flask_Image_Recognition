"""Improved acceptance tests for rejected or incomplete uploads."""
from io import BytesIO


def test_acceptance_missing_file_new(client):
    """Improve `test_acceptance_missing_file` in `test_acceptance_sad.py`.

    Reuses the shared client fixture instead of duplicating fixture setup. It
    verifies the current HTML route contract for a missing file: HTTP 200 with
    the user-facing processing error. Keep this expectation aligned with the app.
    """
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data


def test_acceptance_empty_filename_new(client):
    """Extend missing-file coverage from `test_acceptance_missing_file`.

    Covers a multipart file field with an empty filename, a distinct input from
    omitting the file field entirely. The route should show its controlled error
    rather than a prediction.
    """
    response = client.post("/prediction", data={"file": (BytesIO(b""), "")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
