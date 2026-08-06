import pytest

torch = pytest.importorskip("torch")

from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.evaluation.sbc import compute_sbc_ranks_from_samples
from gapsbi.evaluation.tarp import compute_tarp_from_samples


class _DeterministicPosterior:
    def __init__(self, theta_dim: int) -> None:
        self.theta_dim = theta_dim

    def sample(self, shape, x, show_progress_bars=False):  # noqa: ANN001, ANN202
        del show_progress_bars
        num = int(shape[0])
        base = x.repeat(num, 1)
        noise = torch.randn(num, self.theta_dim) * 0.01
        return base + noise


class _KwargPosterior:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(self, shape, x, show_progress_bars=False, **kwargs):  # noqa: ANN001, ANN202
        del show_progress_bars
        self.calls.append(kwargs)
        num = int(shape[0])
        return x.repeat(num, 1)


class _FallbackPosterior:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(self, shape, x, show_progress_bars=False, **kwargs):  # noqa: ANN001, ANN202
        del show_progress_bars
        self.calls.append(kwargs)
        if kwargs.get("reject_outside_prior", False):
            raise RuntimeError("max_sampling_time exceeded")
        num = int(shape[0])
        return x.repeat(num, 1)


def test_sample_posteriors_once_shape_and_seed_reproducible() -> None:
    posterior = _DeterministicPosterior(theta_dim=2)
    x_eval = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    samples_1 = sample_posteriors_once(
        posterior=posterior,
        x_eval=x_eval,
        num_posterior_samples=5,
        seed=123,
    )
    samples_2 = sample_posteriors_once(
        posterior=posterior,
        x_eval=x_eval,
        num_posterior_samples=5,
        seed=123,
    )

    assert samples_1.shape == (2, 5, 2)
    assert torch.allclose(samples_1, samples_2)


def test_sample_posteriors_once_passes_optional_sampling_kwargs() -> None:
    posterior = _KwargPosterior()
    x_eval = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    samples, failures, fallbacks, fallback_used = sample_posteriors_once(
        posterior=posterior,
        x_eval=x_eval,
        num_posterior_samples=3,
        seed=123,
        reject_outside_prior=True,
        max_sampling_time=30.0,
        return_num_sampling_failures=True,
    )

    assert failures == 0
    assert fallbacks == 0
    assert fallback_used is False
    assert samples.shape == (1, 3, 2)
    assert posterior.calls[0]["reject_outside_prior"] is True
    assert posterior.calls[0]["max_sampling_time"] == 30.0


def test_sample_posteriors_once_disables_rejection_sampling_by_default() -> None:
    posterior = _KwargPosterior()
    x_eval = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    sample_posteriors_once(
        posterior=posterior,
        x_eval=x_eval,
        num_posterior_samples=3,
        seed=123,
    )

    assert posterior.calls[0]["reject_outside_prior"] is False


def test_sample_posteriors_once_runtime_error_uses_fallback() -> None:
    posterior = _FallbackPosterior()
    x_eval = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    samples, failures, fallbacks, fallback_used = sample_posteriors_once(
        posterior=posterior,
        x_eval=x_eval,
        num_posterior_samples=2,
        seed=123,
        reject_outside_prior=True,
        max_sampling_time=0.01,
        return_num_sampling_failures=True,
    )

    assert samples.shape == (1, 2, 2)
    assert failures == 1
    assert fallbacks == 1
    assert fallback_used is True


def test_compute_sbc_ranks_from_samples_shape_and_values() -> None:
    posterior_samples = torch.tensor(
        [
            [[0.2, 0.6], [0.7, 0.1], [0.5, 0.4]],
            [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    theta_eval = torch.tensor([[0.6, 0.3], [2.5, 0.5]], dtype=torch.float32)

    ranks = compute_sbc_ranks_from_samples(
        posterior_samples=posterior_samples,
        theta_eval=theta_eval,
    )

    assert ranks.shape == (2, 2)
    assert torch.equal(ranks, torch.tensor([[2, 1], [2, 2]]))


def test_compute_tarp_from_samples_shapes_and_bounds() -> None:
    posterior_samples = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            [[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    theta_eval = torch.tensor([[0.4, 0.4], [1.6, 1.6]], dtype=torch.float32)
    references = torch.tensor([[0.2, 0.2], [1.2, 1.2]], dtype=torch.float32)

    ecp, alpha, tarp_probs = compute_tarp_from_samples(
        posterior_samples=posterior_samples,
        theta_eval=theta_eval,
        references=references,
        num_alpha_grid=11,
    )

    assert ecp.shape == (11,)
    assert alpha.shape == (11,)
    assert tarp_probs.shape == (2,)
    assert torch.all((tarp_probs >= 0.0) & (tarp_probs <= 1.0))
    assert torch.isclose(alpha[0], torch.tensor(0.0))
    assert torch.isclose(alpha[-1], torch.tensor(1.0))


def test_sample_posteriors_once_records_assertion_failures() -> None:
    from gapsbi.evaluation.posterior_sampling import sample_posteriors_once

    class AssertionPosterior:
        def sample(self, *args, **kwargs):
            raise AssertionError("spline discriminant failure")

    x_eval = torch.zeros(2, 3)
    samples, failures, fallbacks, fallback_used = sample_posteriors_once(
        posterior=AssertionPosterior(),
        x_eval=x_eval,
        num_posterior_samples=4,
        seed=123,
        return_num_sampling_failures=True,
    )

    assert samples.shape == (2, 4, 3)
    assert torch.isnan(samples).all()
    assert failures == 2
    assert fallbacks == 2
    assert fallback_used


def test_sample_posteriors_once_falls_back_on_nonfinite_samples() -> None:
    from gapsbi.evaluation.posterior_sampling import sample_posteriors_once

    class NonfiniteThenFinitePosterior:
        def __init__(self) -> None:
            self.num_calls = 0

        def sample(self, shape, x, show_progress_bars=False, **kwargs):
            del show_progress_bars, kwargs
            self.num_calls += 1
            num = int(shape[0])
            if self.num_calls == 1:
                return torch.full((num, x.shape[-1]), float("inf"))
            return torch.zeros(num, x.shape[-1])

    x_eval = torch.zeros(1, 3)
    samples, failures, fallbacks, fallback_used = sample_posteriors_once(
        posterior=NonfiniteThenFinitePosterior(),
        x_eval=x_eval,
        num_posterior_samples=4,
        seed=123,
        return_num_sampling_failures=True,
    )

    assert samples.shape == (1, 4, 3)
    assert torch.isfinite(samples).all()
    assert failures == 1
    assert fallbacks == 1
    assert fallback_used
