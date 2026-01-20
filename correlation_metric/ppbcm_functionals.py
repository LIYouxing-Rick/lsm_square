import torch as th

import torch.nn as nn
from typing import List, Tuple, Optional, Union
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import mobius_fn_apply
arsinh = th.asinh

def hemisphere_to_poincare(x):
    x_T, x_n1 = x[..., :-1], x[..., -1:]
    return x_T / (1.0 + x_n1)

def poincare_to_hemisphere(y):
    norm_y2 = y.norm(p=2, dim=-1, keepdim=True) ** 2
    factor = 1.0 / (1.0 + norm_y2)
    return th.cat((2.0 * y * factor, (1.0 - norm_y2) * factor), dim=-1)

def project_to_hemisphere(v, eps: float = 1e-12):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    v = v / norm
    sign = th.sign(v[..., -1:])
    sign = th.where(sign == 0, th.ones_like(sign), sign)
    return v * sign

def torch_beta(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    return th.exp(th.lgamma(a) + th.lgamma(b) - th.lgamma(a + b))

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

 


class CorPolyHyperbolicCholeskyMetric(nn.Module):
    def __init__(self, n, jitter: float = 1e-5):
        super().__init__()
        self.n = n
        self.pball = PoincareBall(c=1.0, learnable=False)
        self.register_buffer('c', th.tensor(1.0))
        self.jitter = jitter
        dim_in = n * (n - 1) // 2
        a_i = th.arange(1, n, dtype=th.float32) / 2.0
        b_i = th.full((n - 1,), 0.5, dtype=th.float32)
        self.register_buffer('beta_i', torch_beta(a_i, b_i))
        a_n = th.tensor(dim_in / 2.0, dtype=th.float32)
        b_n = th.tensor(0.5, dtype=th.float32)
        self.register_buffer('beta_n', torch_beta(a_n, b_n))

    def correlation_to_poincare_concate(self, C):
        B, n, _ = C.shape
        I = th.eye(n, device=C.device, dtype=C.dtype)
        L = th.linalg.cholesky(C + self.jitter * I)
        mapped_rows = []
        beta_n = self.beta_n
        beta_i = self.beta_i
        for i in range(1, n):
            hs_r = L[..., i, :i+1]
            hs_r = project_to_hemisphere(hs_r)
            pball_r = hemisphere_to_poincare(hs_r)
            v_r = self.pball.logmap0(pball_r) * beta_n / beta_i[i - 1]
            mapped_rows.append(v_r)
        x_tangent = th.cat(mapped_rows, dim=-1)
        return self.pball.expmap0(x_tangent)

    def undirectional_RMLR(self, C, weight_g, weight_v, gamma):
        C_phi = self.correlation_to_poincare_concate(C)
        return unidirectional_poincare_mlr(C_phi, weight_g, weight_v / weight_v.norm(dim=-1, keepdim=True).clamp_min(1e-15), gamma, c=self.c)
