"""Improved component tests for preprocessing and prediction logic."""
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import pytest
import model as model_module


def image_stream(fmt="PNG", size=(80, 60)):
    """Create a valid in-memory image for model preprocessing tests."""
    stream = BytesIO()
    Image.new("RGB", size, (255, 100, 0)).save(stream, format=fmt)
    stream.seek(0)
    return stream


def test_preprocess_img_new():
    """Improve `test_preprocess_img` in `test_model.py`.

    Replaces a hard-coded test_images path with a genuine in-memory image.
    Retains the original expected batch shape and normalization range, and adds
    numeric/finite-value checks so the output is suitable for model input.
    """
    processed = model_module.preprocess_img(image_stream())
    assert processed.shape == (1, 224, 224, 3)
    assert np.issubdtype(processed.dtype, np.number)
    assert np.isfinite(processed).all()
    assert processed.min() >= 0
    assert processed.max() <= 1


def test_predict_result_integer_class_new(monkeypatch):
    """Improve `test_predict_result` in `test_model.py`.

    Instead of loading real weights and checking only the result's type, injects
    deterministic model scores and checks that predict_result selects the index
    of the largest score. This isolates class-selection logic from ML inference.
    """
    class FakeModel:
        def predict(self, batch):
            assert batch.shape == (1, 224, 224, 3)
            return np.array([[0.05, 0.10, 0.80, 0.05]])
    monkeypatch.setattr(model_module, "model", FakeModel())
    result = model_module.predict_result(np.zeros((1, 224, 224, 3)))
    assert isinstance(result, (int, np.integer))
    assert result == 2


def test_preprocess_img_missing_path_new():
    """Improve `test_invalid_image_path` in `test_model.py`.

    Keeps the expected FileNotFoundError behavior while avoiding reliance on
    any particular repository-relative path or sample-image directory.
    """
    with pytest.raises(FileNotFoundError):
        model_module.preprocess_img("__definitely_missing_test_image__.png")


def test_preprocess_img_corrupt_bytes_new():
    """Add corrupt-stream coverage beside the original missing-path test.

    Verifies that the preprocessing component rejects existing bytes that do not
    encode an image. This directly tests Pillow decoding, not Flask error rendering.
    """
    with pytest.raises(UnidentifiedImageError):
        model_module.preprocess_img(BytesIO(b"not a real image"))


def test_predict_result_uses_model_output_new(monkeypatch):
    """Improve `test_image_shape_on_prediction` in `test_model.py`.

    The old test repeated an integer-type assertion on another hard-coded image.
    This test uses controlled model scores and verifies the returned class index.
    """
    class FakeModel:
        def predict(self, batch):
            return np.array([[0.30, 0.20, 0.10, 0.40]])
    monkeypatch.setattr(model_module, "model", FakeModel())
    assert model_module.predict_result(np.zeros((1, 224, 224, 3))) == 3


def test_model_prediction_consistency_replaced_new(monkeypatch):
    """Replace the real-inference approach in `test_model_predictions_consistency`.

    Five real predictions on one hard-coded image show repeatability, not accuracy,
    and require model/image artifacts. This uses a stable fake model to verify the
    wrapper consistently maps a controlled output to the expected class index.
    """
    class StableModel:
        def predict(self, batch):
            return np.array([[0.0, 0.0, 1.0]])
    monkeypatch.setattr(model_module, "model", StableModel())
    batch = np.zeros((1, 224, 224, 3))
    assert [model_module.predict_result(batch) for _ in range(3)] == [2, 2, 2]
