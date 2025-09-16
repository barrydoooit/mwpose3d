# mwpose3d/models/utils/sdtw_cuda/sdtw.py
import math
import torch
from torch.autograd import Function
import importlib

# Import the compiled CUDA extension (the .so)
_ops = importlib.import_module("mwpose3d.models.utils.sdtw_cuda.sdtw_cuda")

def _euclidean_dist_func(x, y):
    n, m, d = x.size(1), y.size(1), x.size(2)
    xx = x.unsqueeze(2).expand(-1, n, m, d)
    yy = y.unsqueeze(1).expand(-1, n, m, d)
    return (xx - yy).pow(2).sum(dim=3)

def _jacobian_prod_sq_euclidean(X, Y, Bt):
    # X: (B,N,D), Y: (B,M,D), Bt: (B,M,N)
    row_sum = Bt.sum(dim=1)                          # (B,N)
    term1 = row_sum.unsqueeze(-1) * X                # (B,N,D)
    term2 = Y.transpose(1, 2).matmul(Bt).transpose(1, 2)  # (B,N,D)
    return 2.0 * (term1 - term2)

class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, X, Y, gamma: float, bandwidth: float, normalize: bool, dist_func=None):
        assert X.is_cuda and Y.is_cuda
        assert X.dtype == torch.float32 and Y.dtype == torch.float32
        if dist_func is None:
            dist_func = _euclidean_dist_func

        bw = 0 if bandwidth is None else int(bandwidth)

        if normalize:
            x_cat = torch.cat([X, X, Y], dim=0)
            y_cat = torch.cat([Y, X, Y], dim=0)
            D = dist_func(x_cat, y_cat).contiguous()
            cost, R = _ops.forward(D, float(gamma), bw)
            B = X.size(0)
            out_xy, out_xx, out_yy = torch.split(cost, [B, B, B], dim=0)
            out = out_xy - 0.5 * (out_xx + out_yy)
            D_xy = D[:B]
            R_xy = R[:B]
            ctx.save_for_backward(D_xy, X, Y, R_xy,
                                  torch.tensor(float(gamma), device=X.device),
                                  torch.tensor(float(bw), device=X.device))
            ctx.normalize = True
            return out
        else:
            D = dist_func(X, Y).contiguous()
            cost, R = _ops.forward(D, float(gamma), bw)
            ctx.save_for_backward(D, X, Y, R,
                                  torch.tensor(float(gamma), device=X.device),
                                  torch.tensor(float(bw), device=X.device))
            ctx.normalize = False
            return cost

    @staticmethod
    def backward(ctx, grad_output):
        D, X, Y, R, gamma_t, bandwidth_t = ctx.saved_tensors
        inv_gamma = 1.0 / float(gamma_t.item())
        bw = int(bandwidth_t.item())

        E = _ops.backward(D.contiguous(), R.contiguous(), float(inv_gamma), bw)  # (B,N,M)
        Bt = E.transpose(1, 2).contiguous()
        G = _jacobian_prod_sq_euclidean(X, Y, Bt)  # (B,N,D)
        gradX = grad_output.view(-1, 1, 1).expand_as(G) * G

        # match your previous API: gradient for X only
        return gradX, None, None, None, None, None

class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        super().__init__()
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func

    def forward(self, X, Y):
        return _SoftDTW.apply(X, Y, self.gamma, self.bandwidth, self.normalize, self.dist_func)
