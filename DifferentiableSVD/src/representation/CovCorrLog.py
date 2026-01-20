import sys
import torch
import torch.nn as nn
from torch.autograd import Function
from .MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer

class CovCorrLog(nn.Module):
    def __init__(self, sqrt_method='none', iterNum=5, correlation=0, corr_method='olm', log_op=None, max_iter=100, corr_k=50, series_order=0, cov_square=False, cov_power_1p5=False, cov_power_n=0, is_vec=True, input_dim=2048, dimension_reduction=None):
        super(CovCorrLog, self).__init__()
        self.sqrt_method = sqrt_method
        self.iterNum = iterNum
        self.correlation = int(correlation)
        self.corr_method = corr_method
        self.log_op = log_op
        self.max_iter = int(max_iter)
        self.corr_k = int(corr_k)
        self.series_order = int(series_order)
        self.cov_square = bool(cov_square)
        self.cov_power_1p5 = bool(cov_power_1p5)
        self.cov_power_n = int(cov_power_n)
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

    def _ecm(self, C):
        from correlation_metric.CorMatrix import CorEuclideanCholeskyMetric
        n = C.shape[-1]
        metric = CorEuclideanCholeskyMetric(n, k=self.corr_k)
        V = metric.deformation(C)
        return V + V.transpose(-1, -2)

    def _lecm(self, C):
        from correlation_metric.CorMatrix import CorLogEuclideanCholeskyMetric
        n = C.shape[-1]
        metric = CorLogEuclideanCholeskyMetric(n, k=self.corr_k, series_order=self.series_order)
        V = metric.deformation(C)
        return V + V.transpose(-1, -2)

    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        if self.cov_power_1p5 and self.corr_method in ['ecm','lecm','olm','lsm']:
            eigvals, eigvecs = torch.linalg.eigh(x)
            s = torch.clamp(eigvals, min=torch.finfo(eigvals.dtype).eps)
            cov_half = eigvecs @ torch.diag_embed(s.sqrt()) @ eigvecs.transpose(-1, -2)
            x = cov_half @ x
        elif self.cov_power_n and self.corr_method in ['ecm','lecm','olm','lsm']:
            eigvals, eigvecs = torch.linalg.eigh(x)
            s = torch.clamp(eigvals, min=torch.finfo(eigvals.dtype).eps)
            pow_s = torch.pow(s, float(self.cov_power_n))
            x = eigvecs @ torch.diag_embed(pow_s) @ eigvecs.transpose(-1, -2)
        elif self.cov_square and self.corr_method in ['ecm','lecm','olm','lsm']:
            x = torch.matmul(x, x)
        x = self._matrix_sqrt(x)
        if self.correlation == 1:
            C = self._to_correlation(x)
            if self.corr_method == 'olm':
                X = self._olm(C)
            elif self.corr_method == 'lsm':
                X = self._lsm(C)
            elif self.corr_method == 'ecm':
                X = self._ecm(C)
            elif self.corr_method == 'lecm':
                X = self._lecm(C)
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
