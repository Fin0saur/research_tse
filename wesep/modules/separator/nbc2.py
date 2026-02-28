# Reference:
#   [1] Quan C, Li X. NBC2: Multichannel speech separation with revised narrow-band conformer
#   [2] Original codebase: https://github.com/Audio-WestlakeU/NBSS

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple
from wesep.modules.feature.speech import STFT,iSTFT
import numpy as np

class LayerNorm(nn.LayerNorm):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose:
            input = input.transpose(-1, -2)
        o = super().forward(input)
        if self.transpose:
            o = o.transpose(-1, -2)
        return o

class BatchNorm1d(nn.Module):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__()
        self.transpose = transpose
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)
        o = self.bn.forward(input)
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o

class GroupNorm(nn.GroupNorm):
    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)
        o = super().forward(input)
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o

class GroupBatchNorm(Module):
    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    transpose: bool
    share_along_sequence_dim: bool

    def __init__(self, dim_hidden: int, group_size: int, share_along_sequence_dim: bool = False, transpose: bool = False, affine: bool = True, eps: float = 1e-5) -> None:
        super(GroupBatchNorm, self).__init__()
        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.transpose = transpose
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if transpose:
                self.weight = Parameter(torch.empty([dim_hidden, 1]))
                self.bias = Parameter(torch.empty([dim_hidden, 1]))
            else:
                self.weight = Parameter(torch.empty([dim_hidden]))
                self.bias = Parameter(torch.empty([dim_hidden]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        assert (input.shape[0] // self.group_size) * self.group_size, f'batch size {input.shape[0]} is not divisible by group size {self.group_size}'
        if self.transpose == False:
            B, T, H = input.shape
            input = input.reshape(B // self.group_size, self.group_size, T, H)
            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 3), unbiased=False, keepdim=True)
            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            output = output.reshape(B, T, H)
        else:
            B, H, T = input.shape
            input = input.reshape(B // self.group_size, self.group_size, H, T)
            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)
            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            output = output.reshape(B, H, T)
        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, transpose={transpose}, eps={eps}, affine={affine}'.format(**self.__dict__)

class NBC2Encoder(nn.Module):
    def __init__(self, input_size: int, dim_hidden: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size, 
            out_channels=dim_hidden, 
            kernel_size=kernel_size, 
            stride=1, 
            padding="same"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  (B, C_in, F, T)
        Output: (B, C_out, F, T)
        """
        B, C, F, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * F, C, T)
        
        x = self.conv(x)  # (B*F, dim_hidden, T)
        
        C_out = x.shape[1]
        x = x.view(B, F, C_out, T).permute(0, 2, 1, 3).contiguous()
        
        return x

class NBC2Decoder(nn.Module):
    def __init__(self, dim_hidden: int, n_spk: int = 1):
        super().__init__()
        self.nspk = n_spk
        self.linear = nn.Linear(in_features=dim_hidden, out_features=n_spk * 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  (B, C_in, F, T)
        Output: (B, 2, nspk, F, T)
        """
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.linear(x) # (B, F, T, nspk * 2)
        
        x = x.view(B, F, T, self.nspk, 2)
        
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        
        return x

class NBC2Block(nn.Module):
    def __init__(self, dim_hidden: int, dim_ffn: int, n_heads: int, dropout: float = 0, conv_kernel_size: int = 3, n_conv_groups: int = 8, norms: Tuple[str, str, str] = ("LN", "GBN", "GBN"), group_batch_norm_kwargs: Dict[str, Any] = {'group_size': 257, 'share_along_sequence_dim': False}) -> None:
        super().__init__()
        self.norm1 = self._new_norm(norms[0], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.self_attn = MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = self._new_norm(norms[1], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.linear1 = nn.Linear(dim_hidden, dim_ffn)
        
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            self._new_norm(norms[2], dim_ffn, True, n_conv_groups, **group_batch_norm_kwargs),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Linear(dim_ffn, dim_hidden)
        self.dropout2 = nn.Dropout(dropout)
        
        # Init weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Input:  (B, C, F, T)
        Output: (B, C, F, T), attn
        """
        B, C, F, T = x.shape
        x_internal = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)

        x_, attn = self._sa_block(self.norm1(x_internal), att_mask)
        x_internal = x_internal + x_
        x_internal = x_internal + self._ff_block(self.norm2(x_internal))
    
        x_out = x_internal.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()
        
        return x_out, attn

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if isinstance(self.self_attn, MultiheadAttention):
            x, attn = self.self_attn.forward(x, x, x, average_attn_weights=False, attn_mask=attn_mask)
        else:
            x, attn = self.self_attn(x, attn_mask=attn_mask)
        return self.dropout1(x), attn

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.conv(self.linear1(x).transpose(-1, -2)).transpose(-1, -2))
        return self.dropout2(x)

    def _new_norm(self, norm_type: str, dim_hidden: int, transpose: bool, num_conv_groups: int, **freq_norm_kwargs):
        if norm_type == 'LN':
            norm = LayerNorm(normalized_shape=dim_hidden, transpose=transpose)
        elif norm_type == 'GBN':
            norm = GroupBatchNorm(dim_hidden=dim_hidden, transpose=transpose, **freq_norm_kwargs)
        elif norm_type == 'BN':
            norm = BatchNorm1d(num_features=dim_hidden, transpose=transpose)
        elif norm_type == 'GN':
            norm = GroupNorm(num_groups=num_conv_groups, num_channels=dim_hidden, transpose=transpose)
        else:
            raise Exception(norm_type)
        return norm

class NBC2(nn.Module):
    def __init__(
        self, 
        win, 
        stride,
        spec_dim, 
        n_layers, 
        encoder_kernel_size=5, 
        dim_hidden=192, 
        dim_ffn=384, 
        n_spk=1, # For Separation (multiple output)
        block_kwargs={}
    ):
        super().__init__()
        
        self.stft = STFT(win,stride,win)
        self.encoder = NBC2Encoder(input_size=spec_dim, dim_hidden=dim_hidden, kernel_size=encoder_kernel_size)
        
        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(NBC2Block(dim_hidden=dim_hidden, dim_ffn=dim_ffn, **block_kwargs))

        self.decoder = NBC2Decoder(dim_hidden=dim_hidden, n_spk=n_spk)
        self.istft = iSTFT(win,stride,win)

    def forward(self, x):
        """
        Input:  (B, C, T)
        Output: (B, 1, T)
        """
        spec = self.stft(x)[-1]
        
        spec_RI = torch.cat([spec.real,spec.imag],dim=1)
        
        x = self.encoder(spec_RI)
        
        for m in self.sa_layers:
            x, attn = m(x)
            del attn
        y = self.decoder(x)
        
        y_complex = torch.complex(y[:,0],y[:,1])
        
        out_wav = self.istft(y_complex)
        return out_wav

if __name__ == "__main__":
    from thop import profile, clever_format
    block_kwargs = {
        'n_heads': 2,
        'dropout': 0.1,
        'conv_kernel_size': 3,
        'n_conv_groups': 8,
        'norms': ("LN", "GBN", "GBN"),
        'group_batch_norm_kwargs': {
            'share_along_sequence_dim': False,
            'group_size': 257, # win // 2 + 1     
        }
    }
    model = NBC2(
        win=512,
        stride=256,
        input_size=2, # for only ref-channel input 
        n_spk=1,
        n_layers=8,
        dim_hidden=96,
        dim_ffn=96*2,
        block_kwargs=block_kwargs,
    )

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))

    x = torch.randn(1, 1, 16000)
    model = model.eval()
    with torch.no_grad():
        output = model(x)
    print(output.shape)

    exit()
    macs, params = profile(model, inputs=(x, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)