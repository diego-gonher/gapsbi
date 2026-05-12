import numpy as np
import pytest

from gapsbi.priors import LogUniformPrior, UniformPrior
from gapsbi.rng import make_rng


def test_uniform_prior_sample_shape() -> None:
    prior = UniformPrior(low=np.array([0.0, 2.0]), high=np.array([1.0, 4.0]))

    samples = prior.sample(7, make_rng(123))

    assert samples.shape == (7, 2)


def test_uniform_prior_samples_lie_within_bounds() -> None:
    prior = UniformPrior(low=np.array([0.0, 2.0]), high=np.array([1.0, 4.0]))

    samples = prior.sample(100, make_rng(123))

    assert np.all(samples >= prior.low)
    assert np.all(samples <= prior.high)


def test_uniform_prior_log_prob_inside_and_outside_support() -> None:
    prior = UniformPrior(low=np.array([0.0, 2.0]), high=np.array([1.0, 4.0]))

    log_prob = prior.log_prob(np.array([[0.5, 3.0], [-0.1, 3.0], [0.5, 5.0]]))

    assert np.isfinite(log_prob[0])
    assert log_prob[1] == -np.inf
    assert log_prob[2] == -np.inf


def test_uniform_prior_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        UniformPrior(low=np.array([0.0]), high=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="1D"):
        UniformPrior(low=np.array([[0.0]]), high=np.array([[1.0]]))

    with pytest.raises(ValueError, match="greater than low"):
        UniformPrior(low=np.array([1.0]), high=np.array([1.0]))


def test_log_uniform_prior_sample_shape() -> None:
    prior = LogUniformPrior(
        log_low=np.log(np.array([1.0, 2.0])),
        log_high=np.log(np.array([4.0, 8.0])),
    )

    samples = prior.sample(7, make_rng(123))

    assert samples.shape == (7, 2)


def test_log_uniform_prior_samples_lie_within_bounds() -> None:
    low = np.array([1.0, 2.0])
    high = np.array([4.0, 8.0])
    prior = LogUniformPrior(log_low=np.log(low), log_high=np.log(high))

    samples = prior.sample(100, make_rng(123))

    assert np.all(samples >= low)
    assert np.all(samples <= high)


def test_log_uniform_prior_log_prob_inside_and_outside_support() -> None:
    prior = LogUniformPrior(
        log_low=np.log(np.array([1.0, 2.0])),
        log_high=np.log(np.array([4.0, 8.0])),
    )

    log_prob = prior.log_prob(np.array([[2.0, 4.0], [0.5, 4.0], [2.0, 9.0]]))

    assert np.isfinite(log_prob[0])
    assert log_prob[1] == -np.inf
    assert log_prob[2] == -np.inf


def test_log_uniform_prior_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        LogUniformPrior(log_low=np.array([0.0]), log_high=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="1D"):
        LogUniformPrior(log_low=np.array([[0.0]]), log_high=np.array([[1.0]]))

    with pytest.raises(ValueError, match="greater than log_low"):
        LogUniformPrior(log_low=np.array([1.0]), log_high=np.array([1.0]))
