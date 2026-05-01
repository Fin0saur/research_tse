import numbers

import torch
import torch.nn as nn


def select_norm(norm, dim, eps=1e-5, group=1):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "LN", "gLN", "GN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "LN":
        # dim can be int or tuple
        return nn.LayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "GN":
        return nn.GroupNorm(group, dim, eps)
    elif norm == "BN":
        return nn.BatchNorm1d(dim, eps)
    else:
        return GlobalChannelLayerNorm(dim, eps, elementwise_affine=True)


class AdaNorm2d(nn.Module):
    """
    严格复刻版自适应层归一化 (AdaNorm)
    来源: "Understanding and Improving Layer Normalization" (NeurIPS 2019)
    """

    def __init__(self, k=0.1, C=1.0, eps=1e-5):
        super().__init__()
        # 严格遵守论文：k 和 C 只是超参数，绝对不使用 nn.Parameter!
        self.k = k
        self.C = C
        self.eps = eps

    def forward(self, x):
        # 1. 沿通道维度 (dim=1) 计算均值和方差，对应公式 y = (x - mu) / sigma
        # 论文中是在 hidden size H 维度上计算 [cite: 1060]，对应我们的通道数 C
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        y = (x - mu) / torch.sqrt(var + self.eps)

        # 2. 计算自适应缩放因子: C * (1 - ky)
        # 严格遵守论文指示：调用 .detach() 切断这部分的梯度回传
        adaptive_weight = self.C * (1.0 - self.k * y).detach()

        # 3. 输出: z = C(1 - ky) \odot y
        return adaptive_weight * y


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        x = torch.transpose(x, 1, -1)
        x = super().forward(x)
        x = torch.transpose(x, 1, -1)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Calculate Global Layer Normalization
    dim: (int or list or torch.Size) –
         input shape from an expected input of size
    eps: a value added to the denominator for numerical stability.
    elementwise_affine: a boolean value that when set to True,
        this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                 self.bias)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class ConditionalLayerNorm(nn.Module):
    """
    https://github.com/HuangZiliAndy/fairseq/blob/multispk/fairseq/models/wavlm/WavLM.py#L1160
    """

    def __init__(self,
                 normalized_shape,
                 embed_dim,
                 modulate_bias=False,
                 eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)

        self.embed_dim = embed_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(*normalized_shape))
        self.bias = nn.Parameter(torch.empty(*normalized_shape))
        assert len(normalized_shape) == 1
        self.ln_weight_modulation = FiLM(normalized_shape[0], embed_dim)
        self.modulate_bias = modulate_bias
        if self.modulate_bias:
            self.ln_bias_modulation = FiLM(normalized_shape[0], embed_dim)
        else:
            self.ln_bias_modulation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, embed):
        mean = torch.mean(input, -1, keepdim=True)
        var = torch.var(input, -1, unbiased=False, keepdim=True)
        weight = self.ln_weight_modulation(
            embed, self.weight.expand(embed.size(0), -1))
        if self.ln_bias_modulation is None:
            bias = self.bias
        else:
            bias = self.ln_bias_modulation(embed,
                                           self.bias.expand(embed.size(0), -1))
        res = (input - mean) / torch.sqrt(var + self.eps) * weight + bias
        return res

    def extra_repr(self):
        return "{normalized_shape}, {embed_dim}, \
            modulate_bias={modulate_bias}, eps={eps}".format(**self.__dict__)


class EnergyNorm(nn.Module):

    def __init__(self, eps=1e-8):
        """
        能量归一化 (E-Norm / Standard Deviation Normalization)
        用于 Speaker Embedding 分支，强制消除音频的能量/音量波动。
        """
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        正向传播：对输入进行标准差归一化
        :param x: 输入张量，通常是复数频谱 (B, C, F, T) 或实数波形/特征
        :return: (归一化后的张量 x_norm, 用于恢复的缩放系数 scale)
        """
        B = x.shape[0]

        # 1. 计算标准差 (Standard Deviation)
        # 如果输入是复数（如 STFT 频谱），需先取绝对值得到幅度包络，再计算方差
        if torch.is_complex(x):
            mag = torch.abs(x)
            # 替换为 reshape，自动处理内存不连续问题
            mag_reshaped = mag.reshape(B, -1)
            scale = torch.std(mag_reshaped, dim=1, keepdim=True)
        else:
            # 同样替换为 reshape
            x_reshaped = x.reshape(B, -1)
            scale = torch.std(x_reshaped, dim=1, keepdim=True)

        # 2. 形状对齐 (Broadcasting)
        # 将 scale 从 (B, 1) 重塑为与输入 x 相同的维度数，例如 (B, 1, 1, 1)
        view_shape = [B] + [1] * (x.dim() - 1)
        scale = scale.view(*view_shape) + self.eps

        # 3. 执行归一化
        x_norm = x / scale

        return x_norm, scale

    def inverse(self, x_norm, scale):
        """
        逆向传播：恢复音频的原始能量范围
        :param x_norm: 经过网络处理后的归一化张量
        :param scale: forward 时保存的缩放系数
        :return: 恢复能量后的张量
        """
        return x_norm * scale


class AmplitudeNorm(nn.Module):
    """
    Simple Amplitude Normalization (Parameter-free)
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, F, T) or (B, F, T)
        if x.dim() == 4:
            ref = x[:, 0]
        else:
            ref = x
        mag = torch.abs(ref).mean(dim=(1, 2), keepdim=True) + self.eps
        if x.dim() == 4:
            scale = mag.unsqueeze(1)  # (B, 1, 1, 1)
        else:
            scale = mag.unsqueeze(1)  # (B, 1, 1)
        return x / scale, scale

    def inverse(self, y, scale):
        if y.dim() == 2: scale = scale.view(y.shape[0], 1)
        elif y.dim() == 3: scale = scale.view(y.shape[0], 1, 1)
        elif y.dim() == 4: scale = scale.view(y.shape[0], 1, 1, 1)
        return y * scale
