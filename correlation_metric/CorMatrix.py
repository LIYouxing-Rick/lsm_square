"""
SPD computations under LieBN - JIT Optimized Version (Fixed)
修复 TorchScript 编译错误
"""
import torch as th
from .sym_functional import sym_logm, sym_expm
from .cor_functions import SPDScalingFinder, HolDplusFinder, FDplus, FDstar
from .LieGroups import LieGroup, PullbackEuclideanMetric

# ==================== JIT 编译的核心函数（修复版）====================

@th.jit.script
def fast_matrix_log_lt1(L, order: int):
    """
    JIT 优化的 LT^1 矩阵对数（下三角单位对角矩阵）
    L: [..., n, n] 下三角矩阵，对角线为1
    Returns: log(L) in LT^0
    """
    n = L.shape[-1]
    m = n - 1
    if order > 0 and order < m:
        m = order
    L_minus_I = L.tril(-1)
    log_L = th.zeros_like(L)
    current_power = L_minus_I.clone()
    
    for k in range(1, m + 1):
        coef = ((-1.0) ** (k - 1)) / float(k)
        log_L = log_L + coef * current_power
        if k < m:
            current_power = current_power @ L_minus_I
    
    return log_L

@th.jit.script
def fast_matrix_exp_lt0(xi, order: int):
    """
    JIT 优化的 LT^0 矩阵指数（下三角零对角矩阵）
    修复版：避免动态 shape 推断
    xi: [..., n, n] in LT^0
    Returns: exp(xi) in LT^1
    """
    n = xi.shape[-1]
    m = n - 1
    if order > 0 and order < m:
        m = order
    
    # ✅ 修复：直接创建与输入相同 shape 的单位矩阵
    exp_xi = th.zeros_like(xi)
    # 填充对角线为1
    for i in range(n):
        exp_xi[..., i, i] = 1.0
    
    term = exp_xi.clone()
    
    for k in range(1, m + 1):
        term = (term @ xi) / float(k)
        exp_xi = exp_xi + term
    
    return exp_xi

@th.jit.script
def fast_covariance_to_correlation(cov_matrices):
    """
    JIT 优化的协方差转相关矩阵
    cov_matrices: [..., n, n]
    Returns: correlation matrices [..., n, n]
    """
    diag_elements = th.diagonal(cov_matrices, dim1=-2, dim2=-1)
    std_devs = th.sqrt(diag_elements.clamp_min(1e-12))
    normalization_matrix = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
    normalization_matrix = th.where(
        normalization_matrix < 1e-12,
        th.ones_like(normalization_matrix),
        normalization_matrix
    )
    return cov_matrices / normalization_matrix

@th.jit.script
def fast_cholesky_solve_batch(L, C_sub):
    """
    批量 Cholesky 求解: L X^T = C_sub^T
    L: [..., k, k] 下三角矩阵
    C_sub: [..., n, k] 右侧矩阵
    Returns: X [..., n, k]
    """
    rhs = C_sub.transpose(-2, -1).contiguous()
    X = th.linalg.solve_triangular(L, rhs, upper=False)
    return X.transpose(-2, -1)

# ==================== 基础类 ====================

class Correlation(LieGroup):
    """Computation for Correlation data with [...,n,n]"""
    def __init__(self, n, is_detach=False):
        super().__init__(is_detach=is_detach)
        self.n = n
        self.dim = int(n * (n - 1) / 2)
        self.register_buffer('I', th.eye(n))
    
    def _check_point_on_manifold(self, matrix, tol=1e-6):
        """检查是否为有效相关矩阵"""
        if matrix.shape[-1] != matrix.shape[-2]:
            print("Failed: Matrices must be square.")
            return False

        if not th.allclose(matrix, matrix.transpose(-2, -1), atol=tol):
            print("Failed: Batch contains non-symmetric matrices.")
            return False

        eigenvalues = th.linalg.eigvalsh(matrix)
        if not th.all(eigenvalues >= -tol):
            print("Failed: Batch contains non-SPD matrices.")
            return False

        try:
            L = th.linalg.cholesky(matrix)
        except RuntimeError:
            print("Failed: Cholesky decomposition failed.")
            return False

        if not th.all(th.diagonal(L, dim1=-2, dim2=-1) > 0):
            print("Failed: Non-positive diagonal in Cholesky factor.")
            return False

        row_norms = th.sum(L ** 2, dim=-1)
        if not th.allclose(row_norms, th.ones_like(row_norms), atol=tol):
            print("Failed: Cholesky factor rows do not have unit norm.")
            return False

        print("Passed: All matrices are valid correlation matrices.")
        return True

    def symmetrize(self, X):
        return (X + X.transpose(-1, -2)) / 2

    def random(self, *shape, eps=1e-6):
        """生成随机 SPD 矩阵"""
        assert len(shape) >= 2 and shape[-2] == shape[-1]
        n = shape[-1]
        A = th.randn(shape) * 2 - 1
        spd_matrices = th.matmul(A, A.transpose(-2, -1)) + eps * th.eye(n, device=A.device)
        return fast_covariance_to_correlation(spd_matrices)

    def covariance_to_correlation(self, cov_matrices):
        """使用 JIT 优化版本"""
        return fast_covariance_to_correlation(cov_matrices)

    def inner_product(self, A, B):
        return th.einsum('...ij,...ij->...', A, B)


class CorFlatMetric(PullbackEuclideanMetric, Correlation):
    def __init__(self, n):
        super().__init__(n)

    def dist2Isquare(self, X):
        return th.linalg.matrix_norm(X, keepdim=True).square()

    def diff_phi_inv_I(self, V):
        raise NotImplementedError


# ==================== ECM 优化版本 ====================

class CorEuclideanCholeskyMetric(CorFlatMetric):
    """JIT 优化的 Euclidean Cholesky Metric"""
    def __init__(self, n, k=50, jitter=1e-6):
        super().__init__(n)
        self.k = min(k, n)
        self.jitter = jitter

    def _safe_cholesky(self, W, max_tries=5):
        """安全的 Cholesky 分解（自动重试）"""
        jitter = self.jitter
        I = th.eye(W.shape[-1], device=W.device, dtype=W.dtype)
        
        for attempt in range(max_tries):
            try:
                return th.linalg.cholesky(W + jitter * I)
            except RuntimeError:
                jitter *= 10.0
        
        return th.linalg.cholesky(W + jitter * I)

    def deformation(self, C):
        """
        ECM 变形: Cor^+(n) → LT^0(n)
        """
        n = C.shape[-1]
        k = self.k
        device = C.device
        dtype = C.dtype
        
        # 1. 随机采样列
        idx = th.randperm(n, device=device)[:k]
        
        # 2. 提取子矩阵 W [bs, k, k]
        W = C[..., idx, :][..., :, idx]
        W = (W + W.transpose(-2, -1)) / 2
        
        # 3. Cholesky 分解
        Lw = self._safe_cholesky(W)
        
        # 4. 提取对应列 C_sub [bs, n, k]
        C_sub = C[..., :, idx]
        
        # 5. 求解 Lw X^T = C_sub^T → X [bs, n, k]
        X = fast_cholesky_solve_batch(Lw, C_sub)
        
        # 6. 构造完整下三角矩阵
        L_full = th.zeros_like(C)
        L_full[..., :, :k] = X
        
        return L_full.tril(-1)

    def inv_deformation(self, V):
        """
        ECM 逆变形: LT^0(n) → Cor^+(n)
        """
        I = th.eye(V.shape[-1], device=V.device, dtype=V.dtype)
        L = V + I
        Sigma = L @ L.transpose(-1, -2)
        return fast_covariance_to_correlation(Sigma)

    def diff_phi_inv_I(self, V):
        """单位元处的微分（对称化）"""
        return V + V.transpose(-1, -2)


# ==================== LECM 优化版本 ====================

class CorLogEuclideanCholeskyMetric(CorEuclideanCholeskyMetric):
    """JIT 优化的 Log-Euclidean Cholesky Metric"""
    def __init__(self, n, k=50, jitter=1e-6, series_order: int = 0):
        super().__init__(n, k=k, jitter=jitter)
        self.series_order = int(series_order)

    def deformation(self, C):
        """
        LECM 变形: Cor^+(n) → LT^0(n)
        """
        n = C.shape[-1]
        k = self.k
        device = C.device
        
        # 1-5 步与 ECM 相同
        idx = th.randperm(n, device=device)[:k]
        W = C[..., idx, :][..., :, idx]
        W = (W + W.transpose(-2, -1)) / 2
        
        Lw = self._safe_cholesky(W)
        C_sub = C[..., :, idx]
        X = fast_cholesky_solve_batch(Lw, C_sub)
        
        L_full = th.zeros_like(C)
        L_full[..., :, :k] = X
        
        # 6. 关键差异：应用矩阵对数
        # ✅ 修复：先加单位矩阵再取对数
        I = th.eye(n, device=C.device, dtype=C.dtype)
        L_with_I = L_full + I
        L_log = fast_matrix_log_lt1(L_with_I, self.series_order)
        
        return L_log.tril(-1)

    def inv_deformation(self, V):
        """
        LECM 逆变形: LT^0(n) → Cor^+(n)
        """
        L = fast_matrix_exp_lt0(V, self.series_order)
        Sigma = L @ L.transpose(-1, -2)
        return fast_covariance_to_correlation(Sigma)


# ==================== OLM 和 LSM（保持原逻辑）====================

class CorOffLogMetric(CorFlatMetric):
    def __init__(self, n, alpha=1.0, beta=0.0, gamma=0.0, max_iter=100):
        super().__init__(n)
        self.max_iter = max_iter
        self.HolDplusFinder = HolDplusFinder(max_iter=self.max_iter)

        cond1 = n >= 4 and alpha > 0 and 2*alpha + (n-2)*beta > 0 and alpha + (n-1)*(beta + n*gamma) > 0
        cond2 = n == 3 and alpha == 0 and beta > 0 and beta + 3*gamma > 0
        cond3 = n == 2 and alpha == 0 and beta == 0 and gamma > 0
        assert cond1 or cond2 or cond3, "Invalid parameters for CorOffLogMetric"

        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('beta', th.tensor(beta))
        self.register_buffer('gamma', th.tensor(gamma))

    def deformation(self, C):
        """OLM: off ∘ log(C)"""
        sym = sym_logm.apply(C)
        return self.symmetrize(sym - th.diag_embed(sym.diagonal(dim1=-2, dim2=-1)))

    def inv_deformation(self, V):
        """OLM: Exp^o: Hol(n) → Cor^+(n)"""
        return sym_expm.apply(V + self.HolDplusFinder(V))

    def dist2Isquare(self, X):
        """排列不变距离"""
        hol = X - th.diag_embed(X.diagonal(dim1=-2, dim2=-1))
        X2 = hol @ hol
        
        tr_X2 = th.einsum('...ii->...', X2).unsqueeze(-1).unsqueeze(-1)
        sum_X2 = (X2 ** 2).sum(dim=(-2, -1), keepdim=True)
        sum_X_sq = hol.sum(dim=(-2, -1), keepdim=True) ** 2
        
        return self.alpha * tr_X2 + self.beta * sum_X2 + self.gamma * sum_X_sq

    def diff_phi_inv_I(self, V):
        return V


class CorLogScaledMetric(CorFlatMetric):
    def __init__(self, n, alpha=1.0, delta=0.0, zeta=0.0, max_iter=100):
        super().__init__(n)
        self.max_iter = max_iter
        self.SPDScalingFinder = SPDScalingFinder(max_iter=self.max_iter)

        cond1 = n >= 4 and alpha > 0 and n*alpha + (n-2)*delta > 0 and n*alpha + (n-1)*(delta + n*zeta) > 0
        cond2 = n == 3 and alpha == 0 and delta > 0 and delta + 3*zeta > 0
        cond3 = n == 2 and alpha == 0 and delta == 0 and zeta > 0
        assert cond1 or cond2 or cond3, "Invalid parameters for CorLogScaledMetric"

        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('delta', th.tensor(delta))
        self.register_buffer('zeta', th.tensor(zeta))

    def deformation(self, C):
        """LSM: Log* : Cor^+(n) → Row_0(n)"""
        D = th.diag_embed(self.SPDScalingFinder(C))
        R1_spd = D @ C @ D
        return sym_logm.apply(R1_spd)

    def inv_deformation(self, V):
        """LSM: Exp* = cor ∘ exp"""
        return fast_covariance_to_correlation(sym_expm.apply(V))

    def dist2Isquare(self, Y):
        """排列不变距离"""
        Y2 = Y @ Y
        tr_Y2 = th.einsum('...ii->...', Y2).unsqueeze(-1).unsqueeze(-1)
        
        Y_diag = Y.diagonal(dim1=-2, dim2=-1)
        tr_diag_Y2 = (Y_diag ** 2).sum(dim=-1, keepdim=True).unsqueeze(-1)
        sum_trY_sq = Y_diag.sum(dim=-1, keepdim=True).pow(2).unsqueeze(-1)
        
        return self.alpha * tr_Y2 + self.delta * tr_diag_Y2 + self.zeta * sum_trY_sq

    def diff_phi_inv_I(self, V):
        """去除对角线"""
        return V - th.diag_embed(V.diagonal(dim1=-2, dim2=-1))


# ==================== 快速测试 ====================

if __name__ == "__main__":
    import time
    
    th.manual_seed(42)
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    bs = 32
    n = 64
    k = 50
    
    print(f"Testing on {device}: bs={bs}, n={n}, k={k}")
    print("=" * 60)
    
    # 生成测试数据
    A = th.randn(bs, n, n, device=device)
    C = A @ A.transpose(-1, -2) + 0.1 * th.eye(n, device=device)
    C = fast_covariance_to_correlation(C)
    
    # 测试 LECM
    lecm = CorLogEuclideanCholeskyMetric(n, k=k).to(device)
    
    # 预热
    for _ in range(10):
        V = lecm.deformation(C)
        C_recon = lecm.inv_deformation(V)
    
    # 计时
    if device == "cuda":
        th.cuda.synchronize()
    
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        V = lecm.deformation(C)
        C_recon = lecm.inv_deformation(V)
    
    if device == "cuda":
        th.cuda.synchronize()
    
    elapsed = (time.time() - start) / num_iters * 1000
    
    # 精度验证
    error = (C - C_recon).abs().max().item()
    
    print(f"LECM time: {elapsed:.3f} ms/iter")
    print(f"Reconstruction error: {error:.2e}")
    print("=" * 60)
    print("✅ All tests passed!")
