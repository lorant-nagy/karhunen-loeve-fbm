# Karhunen–Loève approximation of fractional Brownian motion

We consider fractional Brownian motion on a finite grid $t_0,\dots,t_n$.

The vector

$$
B = \begin{pmatrix} B^H_{t_0} \\ \vdots \\ B^H_{t_n} \end{pmatrix}
$$

is Gaussian with covariance

$$
\Sigma_{ij} = \frac{1}{2}(t_i^{2H}+t_j^{2H}-|t_i-t_j|^{2H}).
$$

We diagonalize the covariance matrix as

$$
\Sigma = Q\Lambda Q^T.
$$

Here $q_i$ are the eigenvectors and $\lambda_i$ are the eigenvalues.

A sample can then be written as

$$
B = \sum_i \sqrt{\lambda_i} Z_i q_i, \qquad Z_i \sim N(0,1).
$$

Keeping only the first $K$ eigenvectors gives the truncated approximation

$$
B^{(K)} = \sum_{i=1}^{K} \sqrt{\lambda_i} Z_i q_i.
$$

This is the Karhunen–Loève/PCA representation of the discretized process on the chosen grid. Using all nonzero modes gives the exact Gaussian distribution on that grid; using fewer modes gives a low-rank approximation.

## Example

```python
from core import fbm_kl_truncated

t, path = fbm_kl_truncated(
    H=0.7,
    T=1.0,
    n_steps=500,
    K=30,
    random_state=1,
)

print(path.shape)
```