import torch
import torch.nn as nn
import math

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

if __name__ == "__main__":

    encoder = CycPosEncoding(embed_dim=40, alpha=1.0)
    dummy_angles = torch.randn(2, 100) 
    
    encoded_angles = encoder(dummy_angles)
    
    print(f"Input shape: {dummy_angles.shape}")   # torch.Size([2, 100])
    print(f"Output shape: {encoded_angles.shape}") # torch.Size([2, 100, 40])