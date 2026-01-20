import sys
import torch
import torch.nn as nn
from torch.autograd import Function
from .MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer

class CovCorrLog(nn.Module):
    def __init__(self, sqrt_method='none', iterNum=5, correlation=0, corr_method='olm', log_op=None, max_iter=100, is_vec=True, input_dim=2048, dimension_reduction=None):
        super(CovCorrLog, self).__init__()
        self.sqrt_method = sqrt_method
        self.iterNum = iterNum
        self.correlation = int(correlation)
        self.corr_method = corr_method
        self.log_op = log_op
        self.max_iter = int(max_iter)
        self.is_vec = is_vec
        self.dr = dimension_reduction
        if self.dr is not None:
            self.conv_dr_block = nn.Sequential(
                nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dr),
                nn.ReLU(inplace=True)
            )
        output_dim = self.dr if self.dr else input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _matrix_sqrt(self, x):
        if self.sqrt_method == 'none':
            return x
        if self.sqrt_method == 'MPNCOV':
            return SqrtmLayer(x, self.iterNum)
        # SVD-based sqrt
        if self.sqrt_method == 'SVD_Taylor':
            from .SVD_Taylor import Eigen_decomposition as EIG
            vec, diag = EIG.apply(x)
            s = torch.diagonal(diag, dim1=-2, dim2=-1)
            diag_sqrt = torch.diag_embed(torch.clamp(s, min=torch.finfo(s.dtype).eps).sqrt()).type(vec.dtype)
            return vec.bmm(diag_sqrt).bmm(vec.transpose(1, 2))
        if self.sqrt_method == 'SVD_Pade':
            from .SVD_Pade import Eigen_decomposition as EIG
            vec, diag = EIG.apply(x)
            s = torch.diagonal(diag, dim1=-2, dim2=-1)
            diag_sqrt = torch.diag_embed(torch.clamp(s, min=torch.finfo(s.dtype).eps).sqrt()).type(vec.dtype)
            return vec.bmm(diag_sqrt).bmm(vec.transpose(1, 2))
        if self.sqrt_method == 'SVD_TopN':
            from .SVD_TopN import Eigen_decomposition as EIG
            vec, diag = EIG.apply(x)
            s = torch.diagonal(diag, dim1=-2, dim2=-1)
            diag_sqrt = torch.diag_embed(torch.clamp(s, min=torch.finfo(s.dtype).eps).sqrt()).type(vec.dtype)
            return vec.bmm(diag_sqrt).bmm(vec.transpose(1, 2))
        if self.sqrt_method == 'SVD_Trunc':
            from .SVD_Trunc import Eigen_decomposition as EIG
            vec, diag = EIG.apply(x)
            s = torch.diagonal(diag, dim1=-2, dim2=-1)
            diag_sqrt = torch.diag_embed(torch.clamp(s, min=torch.finfo(s.dtype).eps).sqrt()).type(vec.dtype)
            return vec.bmm(diag_sqrt).bmm(vec.transpose(1, 2))
        eigvals, eigvecs = torch.linalg.eigh(x)
        s = torch.clamp(eigvals, min=torch.finfo(eigvals.dtype).eps).sqrt()
        return eigvecs @ torch.diag_embed(s) @ eigvecs.transpose(-1, -2)

    def _cov_pool(self, x):
        return CovpoolLayer(x)

    def _to_correlation(self, C):
        d = torch.diagonal(C, dim1=-2, dim2=-1)
        std = torch.sqrt(torch.clamp(d, min=torch.finfo(C.dtype).eps))
        norm = std.unsqueeze(-1) * std.unsqueeze(-2)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        return C / norm

    def _olm(self, C):
        if self.log_op is None:
            from correlation_metric.sym_functional import sym_logm
            logC = sym_logm.apply(C)
        else:
            logC = self.log_op(C)
        off = logC.tril(-1) + logC.triu(1)
        return (off + off.transpose(-1, -2))

    def _lsm(self, C):
        from correlation_metric.cor_functions import SPDScalingFinder
        D_vec = SPDScalingFinder(max_iter=self.max_iter)(C)
        D = torch.diag_embed(D_vec)
        R1 = D @ C @ D
        if self.log_op is None:
            from correlation_metric.sym_functional import sym_logm
            logR1 = sym_logm.apply(R1)
        else:
            logR1 = self.log_op(R1)
        return logR1

    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        x = self._matrix_sqrt(x)
        if self.correlation == 1:
            C = self._to_correlation(x)
            if self.corr_method == 'olm':
                X = self._olm(C)
            else:
                X = self._lsm(C)
        else:
            if self.log_op is not None:
                X = self.log_op(x)
            else:
                X = x
        if self.is_vec:
            X = TriuvecLayer(X)
        return X
