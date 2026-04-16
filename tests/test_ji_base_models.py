import pandas as pd
import numpy as np

from hc.ji_base import JIBaseConfig
from hc.ji_base.models import (
    IdentityCalibrator,
    MarginProbabilityMapper,
    build_lr_control_pipeline,
    fit_gender_calibrator,
    fit_predict_raw,
)


def test_margin_probability_mapper_is_monotonic_and_bounded():
    mapper = MarginProbabilityMapper(residual_scale=7.5)
    margins = np.array([-25.0, -10.0, -2.0, 0.0, 3.0, 12.0, 30.0], dtype=float)

    probabilities = mapper.predict(margins)

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.all(np.diff(probabilities) >= 0.0)
    assert probabilities[0] < 0.5 < probabilities[-1]


def test_margin_probability_mapper_fit_from_residuals_stays_positive():
    predicted_margin = np.array([-8.0, -3.0, 0.5, 5.0, 11.0], dtype=float)
    actual_margin = np.array([-10.0, -2.0, 1.0, 7.0, 9.0], dtype=float)

    mapper = MarginProbabilityMapper.fit(predicted_margin=predicted_margin, actual_margin=actual_margin)
    probabilities = mapper.predict(predicted_margin)

    assert mapper.residual_scale > 0.0
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_fit_predict_raw_supports_ji_node_control_with_bounded_probabilities():
    x_train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 0.5, -0.5, 1.5, -1.0],
            "b": [1.0, 0.0, 0.25, 1.25, -0.25, 0.75],
            "c": [0.1, -0.2, 0.3, -0.4, 0.8, -0.6],
        }
    )
    y_train = pd.Series([0, 1, 1, 0, 1, 0], dtype=float)
    x_test = pd.DataFrame({"a": [0.2, 0.9], "b": [0.7, -0.1], "c": [0.0, 0.5]})

    pred = fit_predict_raw(JIBaseConfig(gender="M", model_family="JI_node_control"), x_train, y_train, x_test)

    assert pred.shape == (2,)
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)


def test_fit_gender_calibrator_respects_isotonic_min_samples_gate():
    probabilities = np.linspace(0.1, 0.9, 40)
    labels = np.array([0, 1] * 20, dtype=float)

    calibrator = fit_gender_calibrator(
        probabilities=probabilities,
        labels=labels,
        calibration_mode="isotonic_gender",
        isotonic_min_samples=50,
    )

    assert isinstance(calibrator, IdentityCalibrator)


def test_build_lr_control_pipeline_respects_gender_specific_c_overrides():
    men_pipeline = build_lr_control_pipeline(JIBaseConfig(gender="M", lr_c_m=0.7, lr_c_w=0.5))
    women_pipeline = build_lr_control_pipeline(JIBaseConfig(gender="W", lr_c_m=0.7, lr_c_w=0.5))

    assert men_pipeline.named_steps["model"].C == 0.7
    assert women_pipeline.named_steps["model"].C == 0.5
