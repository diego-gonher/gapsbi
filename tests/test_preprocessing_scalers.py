import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch = pytest.importorskip("torch")

from gapsbi.preprocessing.scalers import (
    infer_theta_transform,
    infer_x_transform,
    make_scaled_theta_prior,
    scale_theta_train_val_test,
    scale_x_train_val_test,
    theta_scaling_metadata,
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


@pytest.mark.parametrize("problem_name", ["glm", "lotka_volterra"])
def test_unbounded_theta_transform_uses_standard_scaler(problem_name: str) -> None:
    theta_train = np.array([[-1.0, 2.0], [1.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    theta_val = np.array([[5.0, 8.0]], dtype=np.float64)
    theta_test = np.array([[7.0, 10.0]], dtype=np.float64)

    train_s, val_s, test_s, theta_scaler = scale_theta_train_val_test(
        theta_train=theta_train,
        theta_val=theta_val,
        theta_test=theta_test,
        transform=infer_theta_transform(problem_name),
    )

    manual_scaler = StandardScaler()
    expected_train = manual_scaler.fit_transform(theta_train)
    expected_val = manual_scaler.transform(theta_val)
    expected_test = manual_scaler.transform(theta_test)

    assert isinstance(theta_scaler, StandardScaler)
    assert np.allclose(train_s.numpy(), expected_train)
    assert np.allclose(val_s.numpy(), expected_val)
    assert np.allclose(test_s.numpy(), expected_test)


def test_non_glm_theta_transform_keeps_minmax_scaler() -> None:
    _, _, _, theta_scaler = scale_theta_train_val_test(
        theta_train=np.array([[0.0], [1.0]], dtype=np.float64),
        theta_val=np.array([[0.5]], dtype=np.float64),
        theta_test=np.array([[0.25]], dtype=np.float64),
        transform=infer_theta_transform("oup"),
    )

    assert isinstance(theta_scaler, MinMaxScaler)


def test_scaled_theta_prior_matches_theta_transform() -> None:
    theta_train_scaled = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, -1.0],
        ],
        dtype=torch.float32,
    )
    glm_prior = make_scaled_theta_prior(
        "glm",
        theta_dim=3,
        theta_train_scaled=theta_train_scaled,
    )
    oup_prior = make_scaled_theta_prior("oup", theta_dim=2)

    glm_samples = glm_prior.sample((8,))
    oup_samples = oup_prior.sample((8,))

    assert glm_samples.shape == (8, 3)
    assert oup_samples.shape == (8, 2)
    centered = theta_train_scaled - theta_train_scaled.mean(dim=0, keepdim=True)
    np.testing.assert_allclose(
        glm_prior.covariance_matrix.numpy(),
        (
            centered.T @ centered / theta_train_scaled.shape[0]
            + 1e-5 * torch.eye(3)
        ).numpy(),
    )
    assert torch.isfinite(glm_prior.log_prob(torch.zeros(3)))
    assert torch.isfinite(oup_prior.log_prob(torch.zeros(2)))


def test_theta_scaling_metadata_is_serializable() -> None:
    assert theta_scaling_metadata("standard") == {
        "transform": "standard",
        "scaler": "standard",
        "fit_source": "theta_train",
    }
    assert theta_scaling_metadata("minmax_minus_one_one") == {
        "transform": "minmax_minus_one_one",
        "scaler": "minmax",
        "fit_source": "theta_train",
    }


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
