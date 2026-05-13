from typing import Any

import numpy as np

from gapsbi.priors import UniformPrior
from gapsbi.simulators.base import Simulator


class HodgkinHuxleySimulator(Simulator):
    """Hodgkin-Huxley simulator with downsampled raw voltage traces."""

    def __init__(
        self,
        duration: float = 120.0,
        dt: float = 0.01,
        t_on: float = 10.0,
        curr_level: float = 5e-4,
        V0: float = -70.0,
        downsample: int = 20,
        mode: str = "raw",
    ) -> None:
        if mode == "summary":
            raise NotImplementedError("summary mode is not implemented; use mode='raw'.")
        if mode != "raw":
            raise ValueError("mode must be 'raw' or 'summary'.")

        self.duration = float(duration)
        self.dt = float(dt)
        self.t_on = float(t_on)
        self.t_off = self.duration - self.t_on
        self.curr_level = float(curr_level)
        self.V0 = float(V0)
        self.downsample = int(downsample)
        self.mode = mode
        self.prior = UniformPrior(
            low=np.array([0.5, 1e-4]),
            high=np.array([80.0, 15.0]),
        )
        self.prior_low = self.prior.low
        self.prior_high = self.prior.high

        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.t_on < 0:
            raise ValueError("t_on must be nonnegative")
        if self.curr_level < 0:
            raise ValueError("curr_level must be nonnegative")
        if self.downsample < 1:
            raise ValueError("downsample must be >= 1")

        self.I_inj, self.t, self.A_soma = self._syn_current()
        self.full_trace_length = int(self.t.shape[0])
        self.x_dim = int(self.t[:: self.downsample].shape[0])

    @property
    def name(self) -> str:
        return "hodgkin_huxley"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (self.x_dim,)

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

        x = np.stack([self._simulate_one(row, rng) for row in theta_batch])
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Hodgkin-Huxley simulation produced nonfinite values")
        return x[0] if single else x

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "mode": self.mode,
            "duration": self.duration,
            "dt": self.dt,
            "t_on": self.t_on,
            "t_off": self.t_off,
            "curr_level": self.curr_level,
            "V0": self.V0,
            "downsample": self.downsample,
            "full_trace_length": self.full_trace_length,
            "full_x_shape": (self.full_trace_length,),
            "prior": self.prior.metadata(),
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
        }

    def _syn_current(self) -> tuple[np.ndarray, np.ndarray, float]:
        full_trace_length = int(round(self.duration / self.dt)) + 1
        t = np.arange(full_trace_length, dtype=float) * self.dt
        a_soma = np.pi * ((70.0 * 1e-4) ** 2)
        i_inj = np.zeros_like(t)

        start = int(np.round(self.t_on / self.dt))
        stop = int(np.round(self.t_off / self.dt))
        start = min(max(start, 0), full_trace_length)
        stop = min(max(stop, 0), full_trace_length)
        if stop > start:
            i_inj[start:stop] = self.curr_level / a_soma

        return i_inj, t, a_soma

    def _simulate_one(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gbar_Na = float(theta[0])
        gbar_K = float(theta[1])

        g_leak = 0.1
        gbar_M = 0.07
        tau_max = 6e2
        Vt = -60.0
        nois_fact = 0.1
        E_leak = -70.0
        C = 1.0
        E_Na = 53.0
        E_K = -107.0
        tstep = float(self.dt)

        def efun(z: float) -> float:
            if np.abs(z) < 1e-4:
                return float(1.0 - z / 2.0)
            return float(z / (np.exp(z) - 1.0))

        def alpha_m(x: float) -> float:
            v1 = x - Vt - 13.0
            return 0.32 * efun(-0.25 * v1) / 0.25

        def beta_m(x: float) -> float:
            v1 = x - Vt - 40.0
            return 0.28 * efun(0.2 * v1) / 0.2

        def alpha_h(x: float) -> float:
            v1 = x - Vt - 17.0
            return float(0.128 * np.exp(-v1 / 18.0))

        def beta_h(x: float) -> float:
            v1 = x - Vt - 40.0
            return float(4.0 / (1.0 + np.exp(-0.2 * v1)))

        def alpha_n(x: float) -> float:
            v1 = x - Vt - 15.0
            return 0.032 * efun(-0.2 * v1) / 0.2

        def beta_n(x: float) -> float:
            v1 = x - Vt - 10.0
            return float(0.5 * np.exp(-v1 / 40.0))

        def tau_n(x: float) -> float:
            return 1.0 / (alpha_n(x) + beta_n(x))

        def n_inf(x: float) -> float:
            return alpha_n(x) / (alpha_n(x) + beta_n(x))

        def tau_m(x: float) -> float:
            return 1.0 / (alpha_m(x) + beta_m(x))

        def m_inf(x: float) -> float:
            return alpha_m(x) / (alpha_m(x) + beta_m(x))

        def tau_h(x: float) -> float:
            return 1.0 / (alpha_h(x) + beta_h(x))

        def h_inf(x: float) -> float:
            return alpha_h(x) / (alpha_h(x) + beta_h(x))

        def p_inf(x: float) -> float:
            v1 = x + 35.0
            return float(1.0 / (1.0 + np.exp(-0.1 * v1)))

        def tau_p(x: float) -> float:
            v1 = x + 35.0
            return float(tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1)))

        V = np.zeros(self.full_trace_length, dtype=float)
        n = np.zeros_like(V)
        m = np.zeros_like(V)
        h = np.zeros_like(V)
        p = np.zeros_like(V)

        V[0] = self.V0
        n[0] = n_inf(V[0])
        m[0] = m_inf(V[0])
        h[0] = h_inf(V[0])
        p[0] = p_inf(V[0])

        for i in range(1, self.full_trace_length):
            tau_V_inv = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1]
                + (n[i - 1] ** 4) * gbar_K
                + g_leak
                + gbar_M * p[i - 1]
            ) / C
            V_inf = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na
                + (n[i - 1] ** 4) * gbar_K * E_K
                + g_leak * E_leak
                + gbar_M * p[i - 1] * E_K
                + self.I_inj[i - 1]
                + nois_fact * rng.standard_normal() / (tstep**0.5)
            ) / (tau_V_inv * C)
            V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
            n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
            m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
            h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
            p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

        return V[:: self.downsample].astype(np.float32)
