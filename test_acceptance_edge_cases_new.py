"""Improved acceptance tests for image-upload edge cases."""
from io import BytesIO
from PIL import Image
import pytest
import app as app_module


def image_stream(fmt="JPEG", size=(32, 32)):
    """Create a valid image stream in memory."""
    stream = BytesIO()
    Image.new("RGB", size, (10, 90, 170)).save(stream, format=fmt)
    stream.seek(0)
    return stream


def test_edge_case_large_image_upload_new(client, monkeypatch):
    """Improve `test_edge_case_large_image_upload` in `test_acceptance_edge_cases.py`.

    The old test sent repeated text bytes, not an image. This version uploads a
    valid high-resolution image and mocks inference. It does not assert an
    undocumented maximum upload size.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 2)
    image = image_stream(size=(1600, 1200)); image.name = "large.jpg"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"2" in response.data


def test_edge_case_invalid_image_bytes_new(client, monkeypatch):
    """Improve `test_edge_case_invalid_metadata` in `test_acceptance_edge_cases.py`.

    The old test did not create a metadata scenario; it submitted arbitrary bytes
    and expected success. This version explicitly submits corrupt image bytes and
    checks the application's controlled error response. Preprocessing is real so
    the decoder's rejection is exercised.
    """
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 0)
    corrupt = BytesIO(b"this is not an image"); corrupt.name = "corrupt.jpg"
    response = client.post("/prediction", data={"file": (corrupt, corrupt.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data


def test_edge_case_webp_image_new(client, monkeypatch):
    """Improve `test_edge_case_non_standard_image_extensions` in the old edge tests.

    The old test labelled arbitrary bytes as WebP. This creates genuine WebP
    content and verifies route handling. Environments without Pillow WebP support
    skip this specific test explicitly.
    """
    try:
        image = image_stream("WEBP")
    except (KeyError, OSError) as exc:
        pytest.skip(f"Pillow WebP support unavailable: {exc}")
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 4)
    image.name = "valid.webp"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"4" in response.data


def test_edge_case_sequential_uploads_new(client, monkeypatch):
    """Improve `test_edge_case_sequential_image_uploads` in the old edge tests.

    Sends two genuine images sequentially and verifies independent mocked results.
    This tests sequential request handling, not concurrency or multi-step inference.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    predictions = iter((1, 8))
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: next(predictions))
    responses = []
    for filename in ("first.jpg", "second.jpg"):
        image = image_stream(); image.name = filename
        responses.append(client.post("/prediction", data={"file": (image, filename)}, content_type="multipart/form-data"))
    assert [r.status_code for r in responses] == [200, 200]
    assert b"1" in responses[0].data
    assert b"8" in responses[1].data


def test_edge_case_unexpected_header_new(client, monkeypatch):
    """Improve `test_edge_case_unexpected_headers` in the old edge tests.

    Uses a valid image so decoding does not confound the custom-header scenario.
    The unrelated header should not prevent normal route processing.
    """
    monkeypatch.setattr(app_module, "preprocess_img", lambda stream: "prepared")
    monkeypatch.setattr(app_module, "predict_result", lambda prepared: 6)
    image = image_stream(); image.name = "header-test.jpg"
    response = client.post("/prediction", data={"file": (image, image.name)}, content_type="multipart/form-data", headers={"X-Unexpected-Header": "test-value"})
    assert response.status_code == 200
    assert b"6" in response.data


def test_edge_case_http2_test_removed_new():
    """Document why the old `test_edge_case_upload_over_http2` was removed.

    Flask's in-process test client does not negotiate an HTTP/2 connection, so
    the old test could not test HTTP/2. The current assignment requirements do
    not establish HTTP/2 as a requirement; this marker prevents a false claim.
    """
    assert True
