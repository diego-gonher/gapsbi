# reference code: https://github.com/montefiore-institute/hypothesis/blob/master/hypothesis/benchmark/spatialsir/simulator.py
from typing import Any

import numpy as np
from scipy.signal import convolve2d

from gapsbi.priors import UniformPrior
from gapsbi.simulators.base import Simulator


class SpatialSIRSimulator(Simulator):
    """Spatial SIR lattice simulator with one final snapshot observation."""

    def __init__(
        self,
        lattice_shape: tuple[int, int] = (16, 16),
        measurement_time: float = 0.25,
        simulation_step_size: float = 0.01,
        initial_infection_rate: float = 3.0,
        flatten: bool = True,
    ) -> None:
        self.lattice_shape = tuple(int(value) for value in lattice_shape)
        self.measurement_time = float(measurement_time)
        self.simulation_step_size = float(simulation_step_size)
        self.initial_infection_rate = float(initial_infection_rate)
        self.flatten = bool(flatten)
        self.prior = UniformPrior(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
        )
        self.prior_low = self.prior.low
        self.prior_high = self.prior.high

        if len(self.lattice_shape) != 2:
            raise ValueError("lattice_shape must have length 2")
        if self.lattice_shape[0] <= 0 or self.lattice_shape[1] <= 0:
            raise ValueError("lattice_shape dimensions must be positive")
        if self.measurement_time < 0:
            raise ValueError("measurement_time must be nonnegative")
        if self.simulation_step_size <= 0:
            raise ValueError("simulation_step_size must be positive")
        if self.initial_infection_rate < 0:
            raise ValueError("initial_infection_rate must be nonnegative")

        self.original_x_shape = (3, *self.lattice_shape)

    @property
    def name(self) -> str:
        return "spatial_sir"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def x_shape(self) -> tuple[int, ...]:
        if self.flatten:
            return (int(np.prod(self.original_x_shape)),)
        return self.original_x_shape

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return self.prior.sample(n, rng)

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        single = theta_array.ndim == 1

        if theta_array.ndim < 1 or theta_array.shape[-1] != self.theta_dim:
            raise ValueError("theta must have shape (2,) or (batch, 2)")
        if single:
            theta_batch = theta_array[None, :]
        elif theta_array.ndim == 2:
            theta_batch = theta_array
        else:
            raise ValueError("theta must have shape (2,) or (batch, 2)")

        if np.any((theta_batch < 0.0) | (theta_batch > 1.0)):
            raise ValueError("theta values must lie in [0, 1]")

        x = np.stack([self._simulate_one(row, rng) for row in theta_batch])
        if self.flatten:
            x = x.reshape((theta_batch.shape[0], -1))

        return x[0] if single else x

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "original_x_shape": self.original_x_shape,
            "lattice_shape": self.lattice_shape,
            "measurement_time": self.measurement_time,
            "simulation_step_size": self.simulation_step_size,
            "initial_infection_rate": self.initial_infection_rate,
            "flatten": self.flatten,
            "prior": self.prior.metadata(),
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
        }

    def _simulate_one(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        beta, gamma = theta
        infected = np.zeros(self.lattice_shape, dtype=bool)
        recovered = np.zeros(self.lattice_shape, dtype=bool)
        kernel = np.ones((3, 3), dtype=np.int8)

        num_initial = 1 + int(rng.poisson(self.initial_infection_rate))
        for _ in range(num_initial):
            h = int(rng.integers(0, self.lattice_shape[0]))
            w = int(rng.integers(0, self.lattice_shape[1]))
            infected[h, w] = True

        susceptible = ~recovered & ~infected
        simulation_steps = int(self.measurement_time / self.simulation_step_size)

        for _ in range(simulation_steps):
            if not np.any(infected):
                break

            neighbor_count = convolve2d(
                infected.astype(np.int8),
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            infection_potential = susceptible.astype(float) * beta * neighbor_count / 8.0
            infection_potential = np.clip(infection_potential, 0.0, 1.0)
            newly_infected = rng.uniform(size=self.lattice_shape) < infection_potential
            next_infected = (newly_infected | infected) & ~recovered

            recovery_potential = infected.astype(float) * gamma
            recovery_potential = np.clip(recovery_potential, 0.0, 1.0)
            newly_recovered = rng.uniform(size=self.lattice_shape) < recovery_potential
            next_recovered = recovered | newly_recovered

            infected = next_infected
            recovered = next_recovered
            susceptible = ~recovered & ~infected

        snapshot = np.stack([susceptible, infected, recovered]).astype(np.float32)
        return snapshot
