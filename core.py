from __future__ import annotations

import numpy as np


def fbm_covariance_grid(t: np.ndarray, H: float) -> np.ndarray:
    """
    Build the covariance matrix for fractional Brownian motion B^H
    on a fixed time grid.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time points in [0, T], sorted, typically equidistant.
    H : float
        Hurst parameter in (0, 1).

    Returns
    -------
    C : ndarray, shape (N, N)
        Covariance matrix C_ij = Cov(B^H_{t_i}, B^H_{t_j}).
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array of time points.")
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1).")

    T_i = t[:, None]   # shape (N, 1)
    T_j = t[None, :]   # shape (1, N)

    # Covariance of fBm:
    # R_H(s, t) = 0.5 ( |s|^{2H} + |t|^{2H} - |t - s|^{2H} )
    C = 0.5 * (
        np.power(np.abs(T_i), 2.0 * H)
        + np.power(np.abs(T_j), 2.0 * H)
        - np.power(np.abs(T_i - T_j), 2.0 * H)
    )
    return C


def fbm_kl_truncated(
    H: float,
    T: float = 1.0,
    K: int = 20,
    n_steps: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate fractional Brownian motion on [0, T] using a truncated
    Karhunen–Loève expansion on an equidistant grid.

    This uses a Nyström-type discretization of the covariance operator:
    we build the covariance matrix C_ij = R_H(t_i, t_j) on the grid,
    diagonalize it, and use the eigenvectors/eigenvalues to sample
    a Gaussian vector with the desired covariance.

    Parameters
    ----------
    H : float
        Hurst parameter in (0, 1).
    T : float, optional
        Time horizon (default 1.0).
    K : int, optional
        Number of KL terms (eigenpairs) to keep in the expansion.
        Must satisfy 1 <= K <= N, where N = n_steps + 1.
    n_steps : int, optional
        Number of time steps. The grid has N = n_steps + 1 points
        0, T/n_steps, 2T/n_steps, ..., T.
    random_state : None, int, or numpy.random.Generator, optional
        For reproducibility. If int, used as seed.

    Returns
    -------
    t : ndarray, shape (N,)
        Time grid from 0 to T.
    path : ndarray, shape (N,)
        Simulated trajectory of fBm at the grid points.

    """
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1).")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if T <= 0:
        raise ValueError("T must be positive.")

    N = n_steps + 1
    if not (1 <= K <= N):
        raise ValueError(f"K must be in [1, {N}], got K={K}.")
    
    t = np.linspace(0.0, T, N)

    C = fbm_covariance_grid(t, H)

    w, V = np.linalg.eigh(C)

    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w = np.clip(w, 0.0, None)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    z = rng.normal(size=K)

    coeffs = np.sqrt(w[:K]) * z   # shape (K,)
    path = V[:, :K] @ coeffs      # shape (N,)

    return t, path


# __all__ = ["fbm_covariance_grid", "fbm_kl_truncated"]
