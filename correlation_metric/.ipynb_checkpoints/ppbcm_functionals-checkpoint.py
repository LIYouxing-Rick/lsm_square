import torch as th

@th.jit.script
def hemisphere_to_poincare(x):
    """
    Transform points from the open hemisphere HS^n to the Poincaré ball P^n.

    Parameters:
    - x: torch.Tensor of shape [..., n+1], where the last dimension
         represents the coordinates (x^T, x_{n+1}).

    Returns:
    - y: torch.Tensor of shape [..., n], transformed coordinates in the Poincaré ball.
    """
    x_T, x_n1 = x[..., :-1], x[..., -1]  # Split x into (x_T, x_n+1)
    y = x_T / (1 + x_n1.unsqueeze(-1))  # Apply the transformation: y = x_T / (1 + x_n1)
    return y

@th.jit.script
def poincare_to_hemisphere(y):
    """Map from Poincaré ball to open hemisphere."""
    norm_y = y.norm(p=2,dim=-1, keepdim=True) ** 2
    factor = 1 / (1 + norm_y)
    mapped = th.cat((2 * y * factor, (1 - norm_y) * factor), dim=-1)
    return mapped

import torch as th
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from scipy.special import beta
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import mobius_fn_apply
arsinh = th.asinh

def matrix_logarithm_lt1(L):
    """
    Computes the matrix logarithm for matrices in LT^1 (lower triangular with unit diagonal).
    Parameters:
        L (torch.Tensor): The input matrix of shape [..., n, n] assumed to be in LT^1.
    """
    n = L.shape[-1]
    L_minus_I = L.tril(-1)  # Directly get the strictly lower part, assuming L has unit diagonal
    log_L = th.zeros_like(L, dtype=L.dtype)
    # Initial power (L - I)^1
    current_power = L_minus_I
    # Series expansion up to (n-1) terms for LT^1 matrices
    for k in range(1, n):  # k goes from 1 to n-1
        term = ((-1) ** (k - 1)) / k * current_power
        log_L = log_L + term
        current_power = current_power @ L_minus_I  # Update to the next power
    return log_L

@th.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    dtype = x.dtype
    device = x.device
    rc = th.sqrt(th.as_tensor(c, dtype=dtype, device=device))
    z_unit = z_unit.to(dtype)
    z_norm = z_norm.to(dtype)
    r = r.to(dtype)
    drcr = 2. * rc * r

    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * arsinh(
        (2. * th.matmul(rcx, z_unit) * th.cosh(drcr) - (1. + cx2) * th.sinh(drcr)) / th.clamp_min(1. - cx2, 1e-15))

@th.jit.script
def hemisphere_to_poincare(x):
    """
    Transform points from the open hemisphere HS^n to the Poincaré ball P^n.

    Parameters:
    - x: torch.Tensor of shape [..., n+1], where the last dimension
         represents the coordinates (x^T, x_{n+1}).

    Returns:
    - y: torch.Tensor of shape [..., n], transformed coordinates in the Poincaré ball.
    """
    x_T, x_n1 = x[..., :-1], x[..., -1]  # Split x into (x_T, x_n+1)
    y = x_T / (1 + x_n1.unsqueeze(-1))  # Apply the transformation: y = x_T / (1 + x_n1)
    return y

@th.jit.script
def poincare_to_hemisphere(y):
    """Map from Poincaré ball to open hemisphere."""
    norm_y = y.norm(p=2,dim=-1, keepdim=True) ** 2
    factor = 1 / (1 + norm_y)
    mapped = th.cat((2 * y * factor, (1 - norm_y) * factor), dim=-1)
    return mapped


class CorPolyHyperbolicCholeskyMetric(nn.Module):
    def __init__(self, n,jitter: float = 1e-5):
        super().__init__()
        self.n = n
        self.pball=PoincareBall(c=1.0, learnable=False)
        self.register_buffer('c', th.tensor(1.0))
        self.jitter=jitter

    def correlation_to_poincare_concate(self, C):
        """PPBCM: Cor^+(n) \rightarrow \prod_{i=1}^{n-1} \pball{i},
                  \phi_{\hs{n} \rightarrow \pball{n}} \circ \Chol,
            input: [bs,...,n,n] correlation
            output: [bs, dim2], dim2 = \prod ... \times dim, with dim = n (n-1) /2
        """
        # Step 1: Perform Cholesky decomposition on C to get L
        I = th.eye(C.shape[-1], device=C.device, dtype=C.dtype)
        L = th.linalg.cholesky(C + self.jitter * I)

        # Step 2: Map the lower triangular part of each row of L (from the 2nd row onward) to the Poincaré ball
        size = L.size()
        product_dims = th.prod(th.tensor(size[1:-2])).item()  # Product of all dims in "..."
        dim_in = int(product_dims * size[-1] * (size[-1] - 1) / 2)  # dim_in = product_dims * n * (n-1) / 2
        beta_n = beta(dim_in / 2, 1 / 2)
        mapped_rows = []
        for i in range(1, L.shape[-2]):  # Skip the first row (index 0)
            hs_r = L[..., i, :i+1]  # Take the first i+1 elements of row i
            pball_r = hemisphere_to_poincare(hs_r)  # Apply the hemisphere-to-Poincare transformation
            # Poincare beta concate
            beta_ni = beta(i / 2, 1 / 2)
            v_r = self.pball.logmap0(pball_r) * beta_n / beta_ni
            mapped_rows.append(v_r)

        x = self.pball.expmap0(th.cat(mapped_rows, dim=-1).contiguous().view(size[0], -1)) # Shape will be [bs, dim_in]
        return x

    def undirectional_RMLR(self, C, weight_g, weight_v, gamma):
        C_phi = self.correlation_to_poincare_concate(C)
        return unidirectional_poincare_mlr(C_phi,weight_g, weight_v / weight_v.norm(dim=0).clamp_min(1e-15), gamma,c=self.c)
