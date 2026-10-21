"""Prediction-component tests mapped to CSCN73010 Assignment 2 requirements.

Assignment requirement references:
- R5: At least one function/component associated with preprocessing or prediction is tested.
- R4/R6: Relevant failure and edge behavior should be tested as well as normal behavior.

The trained model is replaced with a small fake predictor in these tests, so they
test predict_result's result-selection logic without running neural-network inference.
"""

import numpy as np
import pytest

import model as model_module


def test_r5_prediction_returns_index_of_highest_score(monkeypatch):
    """R5 — Prediction component: return the class index with the highest model score.

    A deterministic fake model output makes the expected class unambiguous and
    tests the production function's argmax behavior without loading inference work
    into this test.
    """
    scores = np.array([[0.02, 0.03, 0.90, 0.05]])
    observed = {}

    def fake_model_predict(input_data):
        observed["input"] = input_data
        return scores

    monkeypatch.setattr(model_module.model, "predict", fake_model_predict)
    sample_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    result = model_module.predict_result(sample_input)

    assert result == 2
    assert observed["input"] is sample_input


def test_r4_prediction_component_propagates_model_runtime_error(monkeypatch):
    """R4/R6 — Prediction failure edge case: model errors propagate to the caller.

    This component test verifies predict_result does not silently invent a class
    when the underlying model fails. The route-level test verifies the Flask
    boundary converts the same type of failure into a controlled response.
    """
    def fail_model_predict(input_data):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(model_module.model, "predict", fail_model_predict)

    with pytest.raises(RuntimeError, match="model unavailable"):
        model_module.predict_result(
            np.zeros((1, 224, 224, 3), dtype=np.float32)
        )
