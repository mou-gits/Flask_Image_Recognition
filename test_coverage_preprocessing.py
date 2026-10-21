"""Image-preprocessing tests mapped to CSCN73010 Assignment 2 requirements.

Assignment requirement references:
- R4: Invalid or unexpected input is handled appropriately.
- R5: At least one function/component associated with preprocessing or prediction is tested.
- R6: The test suite includes normal cases and edge cases.

These tests call model.preprocess_img directly and use generated in-memory images,
so they do not depend on repository-specific test_images paths.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

import model as model_module


def make_png_bytes(color=(20, 40, 60), size=(32, 24)):
    """Generate a valid RGB PNG in memory for repeatable preprocessing tests."""
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_r5_preprocessing_shape_dtype_and_normalization():
    """R5 — Preprocessing component: output must match the model's documented input contract.

    The current model preprocessing resizes images to 224x224, creates a batch
    dimension, returns numeric data, and scales pixel values into [0, 1].
    """
    processed = model_module.preprocess_img(BytesIO(make_png_bytes()))

    assert processed.shape == (1, 224, 224, 3)
    assert np.issubdtype(processed.dtype, np.number)
    assert np.isfinite(processed).all()
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_r4_invalid_image_bytes_raise_pillow_image_error():
    """R4/R6 — Invalid image edge case: preprocessing must reject bytes that are not an image.

    This checks the component-level behavior directly. The route-level test in
    test_coverage_routes.py separately verifies that the application turns this
    error into a controlled HTTP response.
    """
    with pytest.raises(UnidentifiedImageError):
        model_module.preprocess_img(BytesIO(b"not a valid image"))


def test_r4_missing_image_path_raises_file_not_found():
    """R4/R6 — Missing source file edge case: preprocessing must not accept a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        model_module.preprocess_img("this_file_should_not_exist_for_test.png")


@pytest.mark.parametrize(
    ("image_size", "color"),
    [
        ((1, 1), (0, 0, 0)),
        ((512, 384), (255, 255, 255)),
    ],
)
def test_r5_preprocessing_resizes_small_and_large_images(image_size, color):
    """R5/R6 — Boundary inputs: tiny and larger valid images must be resized and normalized.

    Pytest runs this test once for each parameter set. Both cases assert the same
    output shape and normalized range required by the model input contract.
    """
    processed = model_module.preprocess_img(
        BytesIO(make_png_bytes(color=color, size=image_size))
    )

    assert processed.shape == (1, 224, 224, 3)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
