from __future__ import annotations

from itertools import product
from typing import Callable, Literal

import numpy as np
import pytest
import sklearn.datasets
import torch
from sklearn.base import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

feature_shift_decoders = ["shuffle", "rotate"]
multiclass_decoders = ["shuffle", "rotate"]
fit_modes = [
    "low_memory",
    "fit_preprocessors",
    "fit_with_cache",
]
inference_precision_methods = ["auto", "autocast", torch.float64]
remove_remove_outliers_stds = [None, 12]

all_combinations = list(
    product(
        devices,
        feature_shift_decoders,
        multiclass_decoders,
        fit_modes,
        inference_precision_methods,
        remove_remove_outliers_stds,
    ),
)


# Wrap in fixture so it's only loaded in if a test using it is run
@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    return X, y  # type: ignore


@pytest.mark.parametrize(
    (
        "device",
        "feature_shift_decoder",
        "multiclass_decoder",
        "fit_mode",
        "inference_precision",
        "remove_outliers_std",
    ),
    all_combinations,
)
def test_fit(
    device: Literal["cuda", "cpu"],
    feature_shift_decoder: Literal["shuffle", "rotate"],
    multiclass_decoder: Literal["shuffle", "rotate"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Only GPU supports inference_precision")

    model = TabPFNClassifier(
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        inference_config={
            "OUTLIER_REMOVAL_STD": remove_outliers_std,
            "CLASS_SHIFT_METHOD": multiclass_decoder,
            "FEATURE_SHIFT_METHOD": feature_shift_decoder,
        },
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (
        X.shape[0],
        len(np.unique(y)),
    ), "Probabilities shape is incorrect"

    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect!"


# TODO(eddiebergman): Should probably run a larger suite with different configurations
@parametrize_with_checks(
    [TabPFNClassifier(inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True})],
)
def test_sklearn_compatible_estimator(
    estimator: TabPFNClassifier,
    check: Callable[[TabPFNClassifier], None],
) -> None:
    if check.func.__name__ in (  # type: ignore
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
    ):
        estimator.inference_precision = torch.float64

    check(estimator)


def test_balanced_probabilities(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that balance_probabilities=True works correctly."""
    X, y = X_y

    model = TabPFNClassifier(
        balance_probabilities=True,
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    # Check that probabilities sum to 1 for each prediction
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Check that the mean probability for each class is roughly equal
    mean_probs = probabilities.mean(axis=0)
    expected_mean = 1.0 / len(np.unique(y))
    assert np.allclose(
        mean_probs,
        expected_mean,
        rtol=0.1,
    ), "Class probabilities are not properly balanced"


def test_classifier_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that TabPFNClassifier works correctly within a sklearn pipeline."""
    X, y = X_y

    # Create a simple preprocessing pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                TabPFNClassifier(
                    n_estimators=2,  # Fewer estimators for faster testing
                ),
            ),
        ],
    )

    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)

    # Check that probabilities sum to 1 for each prediction
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Check that the mean probability for each class is roughly equal
    mean_probs = probabilities.mean(axis=0)
    expected_mean = 1.0 / len(np.unique(y))
    assert np.allclose(
        mean_probs,
        expected_mean,
        rtol=0.1,
    ), "Class probabilities are not properly balanced in pipeline"


def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that dict configs behave identically to PreprocessorConfig objects."""
    X, y = X_y

    # Define same config as both dict and object
    dict_config = {
        "name": "quantile_uni_coarse",
        "append_original": False,  # changed from default
        "categorical_name": "ordinal_very_common_categories_shuffled",
        "global_transformer_name": "svd",
        "subsample_features": -1,
    }

    object_config = PreprocessorConfig(
        name="quantile_uni_coarse",
        append_original=False,  # changed from default
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd",
        subsample_features=-1,
    )

    # Create two models with same random state
    model_dict = TabPFNClassifier(
        inference_config={"PREPROCESS_TRANSFORMS": [dict_config]},
        n_estimators=2,
        random_state=42,
    )

    model_obj = TabPFNClassifier(
        inference_config={"PREPROCESS_TRANSFORMS": [object_config]},
        n_estimators=2,
        random_state=42,
    )

    # Fit both models
    model_dict.fit(X, y)
    model_obj.fit(X, y)

    # Compare predictions
    pred_dict = model_dict.predict(X)
    pred_obj = model_obj.predict(X)
    np.testing.assert_array_equal(pred_dict, pred_obj)

    # Compare probabilities
    prob_dict = model_dict.predict_proba(X)
    prob_obj = model_obj.predict_proba(X)
    np.testing.assert_array_almost_equal(prob_dict, prob_obj)
