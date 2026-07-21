import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        mean = 0.0
        std = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x, :]


# class RMSNorm(nn.Module):
#     def __init__(
#         self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
#         self.eps = eps


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         dtype = x.dtype
#         x = x.to(torch.float32)
#         rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
#         x = x * rms
#         x = x.to(dtype)
#         return x * self.weight
class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        x = x * rms
        result = x * self.weight
        return result.to(dtype)


# class SwiGLU(nn.Module):
#     def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
#         super().__init__()

#         # d_ff = max(64, (((8 * d_model // 3) + 32) // 64) * 64)
#         self.weight_1 = self.__create_param(d_model, d_ff)
#         self.weight_3 = self.__create_param(d_model, d_ff)
#         self.weight_2 = self.__create_param(d_ff, d_model)

#     def __create_param(
#         self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None
#     ):
#         param = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
#         mean = 0.0
#         std = (2 / (d_in + d_out)) ** 0.5
#         torch.nn.init.trunc_normal_(param, mean=mean, std=std, a=-3 * std, b=3 * std)
#         return param

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         silu_input = x @ self.weight_1.T
#         print(f"silu_input: {silu_input.shape}")
#         silu = silu_input * torch.sigmoid(silu_input)
#         print(f"silu: {silu.shape}")
#         gate_path = x @ self.weight_3.T
#         return (silu * gate_path) @ self.weight_2.T


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.weight_1 = SwiGLU._create_param(d_model, d_ff, device, dtype)
        self.weight_3 = SwiGLU._create_param(d_model, d_ff, device, dtype)
        self.weight_2 = SwiGLU._create_param(d_ff, d_model, device, dtype)

    @staticmethod
    def _create_param(d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        param = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        mean = 0.0
        std = (2 / (d_in + d_out)) ** 0.5
        torch.nn.init.trunc_normal_(param, mean=mean, std=std, a=-3 * std, b=3 * std)
        return param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_input = x @ self.weight_1.T
        silu = silu_input * torch.sigmoid(silu_input)
        gate_path = x @ self.weight_3.T
        return (silu * gate_path) @ self.weight_2.T
