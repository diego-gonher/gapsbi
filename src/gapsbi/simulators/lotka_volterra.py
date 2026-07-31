from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from gapsbi.simulators.base import Simulator


class LotkaVolterraSimulator(Simulator):
    """SBIBM-inspired Lotka-Volterra simulator with interleaved 1D observations."""

    def __init__(
        self,
        num_timepoints: int = 50,
        days: float = 20.0,
        observation_noise_scale: float = 0.1,
        initial_state: tuple[float, float] = (30.0, 1.0),
        prior_log_mean: np.ndarray | None = None,
        prior_log_std: np.ndarray | None = None,
        max_state: float = 10_000.0,
        ode_rtol: float = 1e-6,
        ode_atol: float = 1e-8,
    ) -> None:
        self.num_timepoints = int(num_timepoints)
        self.days = float(days)
        self.observation_noise_scale = float(observation_noise_scale)
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.prior_log_mean = (
            np.array([-0.125, -3.0, -0.125, -3.0], dtype=float)
            if prior_log_mean is None
            else np.asarray(prior_log_mean, dtype=float)
        )
        self.prior_log_std = (
            np.full(4, 0.5, dtype=float)
            if prior_log_std is None
            else np.asarray(prior_log_std, dtype=float)
        )
        self.max_state = float(max_state)
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)

        if self.num_timepoints <= 1:
            raise ValueError("num_timepoints must be greater than 1")
        if self.days <= 0:
            raise ValueError("days must be positive")
        if self.observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative")
        if self.initial_state.shape != (2,) or np.any(self.initial_state <= 0):
            raise ValueError("initial_state must contain two positive values")
        if self.prior_log_mean.shape != (4,) or self.prior_log_std.shape != (4,):
            raise ValueError("prior_log_mean and prior_log_std must have shape (4,)")
        if np.any(self.prior_log_std <= 0):
            raise ValueError("prior_log_std entries must be positive")
        if self.max_state <= 0:
            raise ValueError("max_state must be positive")

        self.timepoints = np.linspace(0.0, self.days, self.num_timepoints)

    @property
    def name(self) -> str:
        return "lotka_volterra"

    @property
    def theta_dim(self) -> int:
        return 4

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (2 * self.num_timepoints,)

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return rng.lognormal(
            mean=self.prior_log_mean,
            sigma=self.prior_log_std,
            size=(n, self.theta_dim),
        )

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        single = theta_array.ndim == 1

        if single:
            if theta_array.shape != (self.theta_dim,):
                raise ValueError("theta must have shape (4,) or (batch, 4)")
            theta_batch = theta_array[None, :]
        elif theta_array.ndim == 2 and theta_array.shape[1] == self.theta_dim:
            theta_batch = theta_array
        else:
            raise ValueError("theta must have shape (4,) or (batch, 4)")
        if np.any(theta_batch <= 0):
            raise ValueError("Lotka-Volterra parameters must be positive")

        observations = np.empty((theta_batch.shape[0], *self.x_shape), dtype=float)
        for i, theta_i in enumerate(theta_batch):
            states = self._solve_states(theta_i)
            if self.observation_noise_scale > 0:
                noisy = rng.lognormal(
                    mean=np.log(np.clip(states, 1e-10, self.max_state)),
                    sigma=self.observation_noise_scale,
                )
            else:
                noisy = states
            observations[i] = noisy.T.reshape(-1)

        if not np.all(np.isfinite(observations)):
            raise FloatingPointError("Lotka-Volterra simulation produced nonfinite values")
        if np.any(observations <= 0):
            raise FloatingPointError("Lotka-Volterra observations must be positive")

        return observations[0] if single else observations

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "num_timepoints": self.num_timepoints,
            "days": self.days,
            "timepoints": self.timepoints.tolist(),
            "observation_layout": "interleaved_prey_predator",
            "observation_noise": "lognormal",
            "observation_noise_scale": self.observation_noise_scale,
            "initial_state": self.initial_state.tolist(),
            "prior": "lognormal",
            "prior_log_mean": self.prior_log_mean.tolist(),
            "prior_log_std": self.prior_log_std.tolist(),
            "max_state": self.max_state,
            "ode_rtol": self.ode_rtol,
            "ode_atol": self.ode_atol,
        }

    def _solve_states(self, theta: np.ndarray) -> np.ndarray:
        alpha, beta, gamma, delta = theta

        def rhs(_time: float, state: np.ndarray) -> tuple[float, float]:
            prey = max(float(state[0]), 0.0)
            predator = max(float(state[1]), 0.0)
            return (
                alpha * prey - beta * prey * predator,
                -gamma * predator + delta * prey * predator,
            )

        solution = solve_ivp(
            rhs,
            (0.0, self.days),
            self.initial_state,
            t_eval=self.timepoints,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
            method="LSODA",
        )
        if not solution.success or solution.y.shape != (2, self.num_timepoints):
            raise FloatingPointError("Lotka-Volterra ODE solve failed")
        states = np.clip(solution.y, 1e-10, self.max_state)
        if not np.all(np.isfinite(states)):
            raise FloatingPointError("Lotka-Volterra ODE solve produced nonfinite states")
        return states
