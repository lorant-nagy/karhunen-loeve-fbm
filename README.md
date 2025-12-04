# Fractional Brownian Motion via Karhunen–Loève Expansion

This repository provides a **numerical Karhunen–Loève (KL) representation** of
fractional Brownian motion (fBm) on a finite interval \([0, T]\).

The core idea:

- fBm is a centered Gaussian process with a known covariance kernel.
- On a finite time grid, the covariance kernel induces a **covariance matrix**.
- Diagonalizing that matrix yields a **finite-dimensional KL expansion**.
- Truncating the expansion gives a practical and interpretable simulator.

Mathematically, this is a **Nyström-type discretization** of the covariance
operator of fBm.

---

## Theory Overview

### 1. Karhunen–Loève expansion (continuous setting)

Let \((X_t)_{t \in [0,T]}\) be a centered, square-integrable stochastic process
with continuous covariance kernel
\[
R(s, t) = \mathbb{E}[X_s X_t].
\]

The associated covariance operator \(K\) on \(L^2[0,T]\) is
\[
(Kf)(t) = \int_0^T R(s, t) f(s)\, ds.
\]

Under mild conditions, \(K\) is a compact, self-adjoint, positive operator, so
there exists an orthonormal basis of eigenfunctions
\(\{e_n\}_{n \ge 1}\) with eigenvalues \(\lambda_n \ge 0\), \(\lambda_n \to 0\).

The Karhunen–Loève expansion states that
\[
X_t = \sum_{n=1}^\infty \sqrt{\lambda_n} \, \xi_n \, e_n(t),
\]
where \(\xi_n\) are uncorrelated, mean-zero random variables with
\(\mathbb{E}[\xi_n^2] = 1\). If \(X\) is Gaussian, the \(\xi_n\) are actually
i.i.d. standard normal random variables.

This expansion converges in \(L^2\), and often almost surely.

### 2. Fractional Brownian motion

Fractional Brownian motion (fBm) with Hurst parameter \(H \in (0,1)\) is the
centered Gaussian process \((B^H_t)_{t \in [0,T]}\) with covariance
\[
R_H(s, t) = \mathbb{E}[B^H_s B^H_t]
= \frac{1}{2} \left( |s|^{2H} + |t|^{2H} - |t - s|^{2H} \right).
\]

- For \(H = 1/2\), this reduces to standard Brownian motion.
- For \(H \neq 1/2\), fBm is not a semimartingale and exhibits long-range
  dependence (for \(H > 1/2\)) or short-range dependence (for \(H < 1/2\)).

Because fBm is Gaussian with a continuous covariance kernel, it admits a KL
expansion of the form
\[
B^H_t = \sum_{n=1}^\infty \sqrt{\lambda_n^{(H)}} \, \xi_n \, e_n^{(H)}(t),
\]
where \((\lambda_n^{(H)}, e_n^{(H)})\) are eigenpairs of the covariance
operator associated with \(R_H\).

In general, these eigenfunctions are **not** available in a simple closed form
(except in special cases), which motivates a numerical approach.

### 3. Nyström-type discretization

On a finite time grid
\[
0 = t_0 < t_1 < \dots < t_{N-1} = T,
\]
define the covariance matrix
\[
C_{ij} = R_H(t_i, t_j).
\]

This matrix is symmetric and positive semidefinite. We compute its
eigen-decomposition
\[
C = V \Lambda V^\top,
\]
where \(\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_N)\) and the columns
of \(V\) are eigenvectors \(v_1, \dots, v_N\).

This is the **finite-dimensional analogue** of the KL expansion:
\[
(B^H_{t_0}, \dots, B^H_{t_{N-1}})^\top
= \sum_{n=1}^N \sqrt{\lambda_n} \, \xi_n \, v_n,
\]
with \(\xi_n \sim \mathcal{N}(0,1)\) independent.

Truncating after the first \(K\) eigenpairs (largest eigenvalues) yields
\[
X \approx \sum_{n=1}^K \sqrt{\lambda_n} \, \xi_n \, v_n.
\]

This is exactly the KL expansion of the **discretized process** on the grid.
As the mesh is refined and \(K\) increases, this approximates the continuous
KL expansion of fBm; this is a classical Nyström-type discretization of the
covariance operator.
