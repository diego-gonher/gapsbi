import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

import gapsbi.methods.sbi_npe as sbi_npe


class _DummyInference:
    def __init__(self, prior, density_estimator, device):  # noqa: ANN001, ANN204
        self.prior = prior
        self.density_estimator = density_estimator
        self.device = device
        self.appended = None
        self.split = None

    def append_simulations(self, theta, x):  # noqa: ANN001, ANN202
        self.appended = (theta.clone(), x.clone())

    def set_fixed_train_val_split(self, n_train, n_val):  # noqa: ANN001, ANN202
        self.split = (n_train, n_val)

    def train(self, **kwargs):  # noqa: ANN003, ANN202
        self.train_kwargs = kwargs
        return "trained-density"

    def build_posterior(self, density_estimator):  # noqa: ANN001, ANN202
        return f"posterior-{density_estimator}"


def test_train_fixed_split_npe_concatenates_in_train_val_order(monkeypatch) -> None:
    monkeypatch.setattr(sbi_npe, "FixedSplitNPE_C", _DummyInference)

    theta_train = torch.tensor([[1.0], [2.0]])
    x_train = torch.tensor([[10.0], [20.0]])
    theta_val = torch.tensor([[3.0]])
    x_val = torch.tensor([[30.0]])

    inference, density_estimator, posterior = sbi_npe.train_fixed_split_npe(
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        prior="dummy-prior",
        density_estimator="nsf",
        device="cpu",
        training_batch_size=2,
        stop_after_epochs=5,
        max_num_epochs=10,
    )

    appended_theta, appended_x = inference.appended
    assert torch.equal(appended_theta, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(appended_x, torch.tensor([[10.0], [20.0], [30.0]]))
    assert inference.split == (2, 1)
    assert density_estimator == "trained-density"
    assert posterior == "posterior-trained-density"
