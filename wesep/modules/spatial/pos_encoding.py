import torch
import torch.nn as nn
import math

class PosEncodingFactory:
    @staticmethod
    def create(encoding_config: dict, use_ele: bool = False):
        encoding_type = encoding_config.get("encoding", "oh")
        
        if encoding_type in ["oh", "onehot"]:
            emb_dim = encoding_config.get("emb_dim", 360)
            encoder = OneHotEncoding(embed_dim=emb_dim)
            enc_dim = emb_dim * (2 if use_ele else 1)
        elif encoding_type == "cyc":
            emb_dim = encoding_config.get("cyc_dimension", 40)
            alpha = encoding_config.get("cyc_alpha", 20)
            encoder = CycPosEncoding(embed_dim=emb_dim, alpha=alpha)
            enc_dim = emb_dim * (2 if use_ele else 1)
        elif encoding_type == "sh":
            # SoundCompass 默认使用 5 阶
            order = encoding_config.get("sh_order", 5)
            encoder = SphericalHarmonicsEncoding(order=order)
            enc_dim = encoder.out_dim
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
            
        return encoder, enc_dim

class SphericalHarmonicsEncoding(nn.Module):
    """
    SoundCompass 提出的球谐函数 (Spherical Harmonics, SH) 编码器。
    将 (azi, ele) 映射到 2D 球面的连续正交基空间。
    """
    def __init__(self, order=5):
        super().__init__()
        self.order = order
        # N阶SH的基函数个数为 (N+1)^2，由于拆分实部和虚部，总维度为 2(N+1)^2
        self.out_dim = 2 * (order + 1) ** 2

    def _compute_legendre_polynomials(self, x, order):
        """ 迭代计算连带勒让德多项式 P_n^m(x) """
        B = x.shape[0]
        # P[n][m] 存储 P_n^m
        P = [[torch.zeros_like(x) for _ in range(order + 1)] for _ in range(order + 1)]
        
        # 初始条件
        P[0][0] = torch.ones_like(x)
        if order > 0:
            P[1][0] = x
            sq = torch.sqrt(1 - x**2 + 1e-8)
            P[1][1] = -sq
            
        for n in range(2, order + 1):
            for m in range(n):
                if m == n - 1:
                    P[n][n-1] = x * (2 * n - 1) * P[n-1][n-1]
                else:
                    P[n][m] = (x * (2 * n - 1) * P[n-1][m] - (n + m - 1) * P[n-2][m]) / (n - m)
            # P_n^n
            P[n][n] = -(2 * n - 1) * sq * P[n-1][n-1]
            
        return P

    def forward(self, azi, ele):
        # 物理学标准球坐标：ele 通常是从 z 轴的极角 (0 到 pi)。
        # 如果你的 ele 是仰角 (-pi/2 到 pi/2)，我们需要转换为极角
        theta = math.pi / 2.0 - ele  # 仰角转极角
        phi = azi
        
        cos_theta = torch.cos(theta)
        P = self._compute_legendre_polynomials(cos_theta, self.order)
        
        sh_real = []
        sh_imag = []
        
        for n in range(self.order + 1):
            for m in range(n + 1):
                # 归一化系数 (简化的常数项，网络可以通过 Linear 自行吸收尺度，这里保留数学比例)
                coef = math.sqrt((2 * n + 1) / (4 * math.pi) * math.factorial(n - m) / (math.factorial(n + m) + 1e-8))
                
                P_nm = P[n][m]
                
                # m=0 时没有虚部，但为了结构对齐，虚部填 0
                r_comp = coef * P_nm * torch.cos(m * phi)
                i_comp = coef * P_nm * torch.sin(m * phi)
                
                sh_real.append(r_comp)
                sh_imag.append(i_comp)
                
                # 负 m 部分 (通过对称性)
                if m > 0:
                    sh_real.append(r_comp) # 简化表示
                    sh_imag.append(-i_comp)

        # 拼接实部和虚部
        sh_features = torch.cat(sh_real + sh_imag, dim=-1)
        # 截断或补齐到精准的 out_dim (处理 m 正负项的冗余展开)
        if sh_features.shape[-1] > self.out_dim:
            sh_features = sh_features[..., :self.out_dim]
            
        return sh_features


class ComplexExpEncoding(nn.Module):
    """
    Implementation of the exponential encoding (exp) proposed in:
    "Location-Aware Target Speaker Extraction for Hearing Aids" (Interspeech 2025)
    
    Upgraded to support 3D spatial encoding (Azimuth + Elevation).
    """
    def __init__(self, use_ele: bool = False, ele_mode: str = "sphere"):
        super().__init__()
        self.use_ele = use_ele
        self.ele_mode = ele_mode
        
        if self.use_ele and self.ele_mode not in ["sphere", "concat"]:
            raise ValueError(f"Unsupported ele_mode: {self.ele_mode}. Use 'sphere' or 'concat'.")
            
    def forward(self, azi: torch.Tensor, ele: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            azi: (B,) or (B, 1)。
            ele: (B,) or (B, 1)。 use_ele True 
        Returns:
            - azi: (B, 2)
            - use_ele + 'sphere': (B, 3) 
            - use_ele + 'concat': (B, 4) 
        """
        if azi.dim() == 2:
            azi = azi.squeeze(-1)
            
        if not self.use_ele:
            doa_enc = torch.stack([torch.cos(azi), torch.sin(azi)], dim=-1)
            return doa_enc
            
        if ele is None:
            raise ValueError("Elevation (ele) tensor must be provided when use_ele is True.")
            
        if ele.dim() == 2:
            ele = ele.squeeze(-1)

        if self.ele_mode == "sphere":
            x = torch.cos(ele) * torch.cos(azi)
            y = torch.cos(ele) * torch.sin(azi)
            z = torch.sin(ele)
            
            doa_enc = torch.stack([x, y, z], dim=-1)
            
        elif self.ele_mode == "concat":
            azi_enc = torch.stack([torch.cos(azi), torch.sin(azi)], dim=-1)
            ele_enc = torch.stack([torch.cos(ele), torch.sin(ele)], dim=-1)
            
            doa_enc = torch.cat([azi_enc, ele_enc], dim=-1)
            
        return doa_enc

class CycPosEncoding(nn.Module):
    def __init__(self, embed_dim, alpha=1.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
            
        self.embed_dim = embed_dim
        self.alpha = alpha
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        self.register_buffer('div_term', div_term)

    def forward(self, angle):

        x = angle * self.alpha 
        phase = x.unsqueeze(-1) * self.div_term
    
        output = torch.zeros(*angle.shape, self.embed_dim, device=angle.device, dtype=angle.dtype)
        
        output[..., 0::2] = torch.sin(phase)
        output[..., 1::2] = torch.cos(phase)
        
        return output
class OneHotEncoding(nn.Module):
    def __init__(self, embed_dim, min_val=-math.pi, max_val=math.pi):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_val = min_val
        self.interval = max_val - min_val
        self.register_buffer('identity', torch.eye(embed_dim))

    def forward(self, angle):
        x_norm = (angle - self.min_val) % self.interval
        x_norm = x_norm / self.interval
        
        indices = (x_norm * self.embed_dim).long()
        
        indices = torch.clamp(indices, 0, self.embed_dim - 1)
        
        output = self.identity[indices]
        if output.dtype != angle.dtype:
            output = output.to(angle.dtype)
            
        return output
if __name__ == "__main__":

    encoder = CycPosEncoding(embed_dim=40, alpha=1.0)
    dummy_angles = torch.randn(2, 100) 
    
    encoded_angles = encoder(dummy_angles)
    
    print(f"Input shape: {dummy_angles.shape}")   # torch.Size([2, 100])
    print(f"Output shape: {encoded_angles.shape}") # torch.Size([2, 100, 40])