import torch
import torch.nn as nn
import math

class PosEncodingFactory:
    # manage encoding methods
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
            
        elif encoding_type == "exp":
            ele_mode = encoding_config.get("exp_ele_mode", "sphere")
            encoder = ComplexExpEncoding(use_ele=use_ele, ele_mode=ele_mode)
            enc_dim = 3 if (use_ele and ele_mode == "sphere") else (4 if use_ele else 2)
            
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
            
        return encoder, enc_dim


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