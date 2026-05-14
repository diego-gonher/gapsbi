import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip("torch")

from gapsbi.preprocessing.scalers import (
    infer_x_transform,
    scale_theta_train_val_test,
    scale_x_train_val_test,
)


def test_scale_theta_shapes_preserved_and_torch_float32() -> None:
    theta_train = np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float64)
    theta_val = np.array([[0.5, 3.0]], dtype=np.float64)
    theta_test = np.array([[0.25, 2.5], [0.75, 3.5]], dtype=np.float64)

    train_s, val_s, test_s, _ = scale_theta_train_val_test(
        theta_train=theta_train,
        theta_val=theta_val,
        theta_test=theta_test,
    )

    assert train_s.shape == theta_train.shape
    assert val_s.shape == theta_val.shape
    assert test_s.shape == theta_test.shape
    assert train_s.dtype == torch.float32
    assert val_s.dtype == torch.float32
    assert test_s.dtype == torch.float32


def test_theta_scaler_fit_only_on_train() -> None:
    theta_train = np.array([[0.0], [1.0]], dtype=np.float64)
    theta_val = np.array([[100.0]], dtype=np.float64)
    theta_test = np.array([[-100.0]], dtype=np.float64)

    _, _, _, theta_scaler = scale_theta_train_val_test(
        theta_train=theta_train,
        theta_val=theta_val,
        theta_test=theta_test,
    )

    assert np.allclose(theta_scaler.data_min_, np.array([0.0]))
    assert np.allclose(theta_scaler.data_max_, np.array([1.0]))


def test_scale_x_log1p_standard_matches_manual_pipeline() -> None:
    x_train = np.array([[0.0, 1.0], [3.0, 7.0]], dtype=np.float64)
    x_val = np.array([[15.0, 31.0]], dtype=np.float64)
    x_test = np.array([[63.0, 127.0]], dtype=np.float64)

    train_s, val_s, test_s, x_scaler, metadata = scale_x_train_val_test(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        transform="log1p_standard",
    )

    manual_scaler = StandardScaler()
    expected_train = manual_scaler.fit_transform(np.log1p(x_train))
    expected_val = manual_scaler.transform(np.log1p(x_val))
    expected_test = manual_scaler.transform(np.log1p(x_test))

    assert np.allclose(train_s.numpy(), expected_train)
    assert np.allclose(val_s.numpy(), expected_val)
    assert np.allclose(test_s.numpy(), expected_test)
    assert np.allclose(x_scaler.mean_, manual_scaler.mean_)
    assert metadata["transform"] == "log1p_standard"
    assert metadata["pre_transform"] == "log1p"
    assert metadata["scaler"] == "standard"


def test_scale_x_standard_fit_only_on_train() -> None:
    x_train = np.array([[0.0], [2.0]], dtype=np.float64)
    x_val = np.array([[100.0]], dtype=np.float64)
    x_test = np.array([[-100.0]], dtype=np.float64)

    _, _, _, x_scaler, metadata = scale_x_train_val_test(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        transform="standard",
    )

    assert np.allclose(x_scaler.mean_, np.array([1.0]))
    assert metadata["transform"] == "standard"
    assert metadata["pre_transform"] == "identity"


def test_scale_x_invalid_transform_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid transform"):
        scale_x_train_val_test(
            x_train=np.array([[1.0]]),
            x_val=np.array([[1.0]]),
            x_test=np.array([[1.0]]),
            transform="does_not_exist",
        )


def test_infer_x_transform_ricker_vs_other() -> None:
    assert infer_x_transform("ricker") == "log1p_standard"
    assert infer_x_transform("RICKER") == "log1p_standard"
    assert infer_x_transform("oup") == "standard"
