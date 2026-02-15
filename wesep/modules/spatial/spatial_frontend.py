import torch
import torch.nn as nn
import math
from wesep.modules.common.deep_update import deep_update
from wesep.modules.feature.speech import STFT
from wesep.modules.spatial.pos_encoding import CycPosEncoding

class BaseSpatialFeature(nn.Module):
    def __init__(self, config, geometry_ctx=None):
        super().__init__()
        self.config=config
        self.default_pairs = config.get('pairs', None)
        if geometry_ctx is not None:
            self.register_buffer('mic_pos', geometry_ctx['mic_pos'])
            self.register_buffer('omega_over_c', geometry_ctx['omega_over_c'])

    def _get_pairs(self, pairs_arg):
        if pairs_arg is not None:
            return pairs_arg
        if self.default_pairs is not None:
            return self.default_pairs
        raise ValueError(f"{self.__class__.__name__}: No pairs provided in arg or config.")

    def _compute_tpd(self, azi, ele, F_dim, pairs):
        u_x = torch.cos(ele) * torch.cos(azi)
        u_y = torch.cos(ele) * torch.sin(azi)
        u_z = torch.sin(ele)
        u_vec = torch.stack([u_x, u_y, u_z], dim=1) 

        d_vecs = []
        for (i, j) in pairs:
            d_vecs.append(self.mic_pos[i] - self.mic_pos[j])
        d_tensor = torch.stack(d_vecs, dim=0) 

        dist_delay = torch.matmul(u_vec, d_tensor.T) 

        TPD = self.omega_over_c.view(1, 1, F_dim, 1) * dist_delay.unsqueeze(-1).unsqueeze(-1)
        return TPD 

    def compute(self, azi, ele=None, Y=None,pairs=None):
        raise NotImplementedError

    def post(self, mix_repr, spatial_repr):
        raise NotImplementedError

class CycEncoder(BaseSpatialFeature):
    def __init__(self, config):
        super().__init__(config)
        
        enc_cfg = self.config
        self.embed_dim = enc_cfg['cyc_dimension']  # e.g., 40
        self.alpha = enc_cfg.get('cyc_alpha', 1.0) # e.g., 20
        self.enabled = enc_cfg['enabled']
        self.use_ele = enc_cfg.get('use_ele', False) 
        self.fusion = enc_cfg.get('fusion_type',"concat")
        out_channels = enc_cfg['out_channel']
        
        self.cyc_pos = CycPosEncoding(embed_dim=self.embed_dim, alpha=self.alpha)
        
        mlp_input_dim = self.embed_dim * 2 if self.use_ele else self.embed_dim
        
        # 4. Clue Encoder Structure (Linear -> LN -> PReLU)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.PReLU()
        )
        
        self.out_channels = out_channels

    def compute(self, azi, ele=None, Y=None,pairs=None):
        if not self.enabled:
            return None

        if azi.dim() == 1:
            azi = azi.unsqueeze(1) # (B,) -> (B, 1)
        if ele is not None and ele.dim() == 1:
            ele = ele.unsqueeze(1)
        
        enc_feat = self.cyc_pos(azi)
        if self.use_ele:
            if ele is None:
                raise ValueError("Config indicates 'use_ele=True' but 'ele' input is None!")
            
            # Input: (B, T) -> Output: (B, T, D)
            enc_ele = self.cyc_pos(ele)
            
            # (B, T, D) + (B, T, D) -> (B, T, 2*D)
            enc_feat = torch.cat([enc_feat, enc_ele], dim=-1)

        # Input: (B, T, mlp_input_dim) -> Output: (B, T, out_channels)
        spatial_repr = self.mlp(enc_feat)

        # (B, T, C) -> Permute to (B, C, T) -> Unsqueeze to (B, C, 1, T)
        spatial_repr = spatial_repr.permute(0, 2, 1).unsqueeze(2)
        
        return spatial_repr

    def post(self, mix_repr, spatial_repr):
        """
        Args:
            mix_repr: (B, C_mix, F, T)   <-- 主干特征，例如 (Batch, 192, 257, 100)
            spatial_repr: (B, C_enc, 1, T) <-- DOA特征，例如 (Batch, 192, 1, 100)
        Returns:
            Fused feature: (B, C_out, F, T)
        """
        if spatial_repr is None:
            return mix_repr
            
        if self.fusion == "concat":
            target_F = mix_repr.shape[2]
            target_T = mix_repr.shape[3]
            spatial_repr_expanded = spatial_repr.expand(-1, -1, target_F, target_T)
            out = torch.cat([mix_repr, spatial_repr_expanded], dim=1)
            
        elif self.fusion == "multiply":
            if mix_repr.shape[1] != spatial_repr.shape[1]:
                raise ValueError(
                    f"Fusion 'multiply' requires same channel dimensions. "
                    f"Mix: {mix_repr.shape[1]}, Spatial: {spatial_repr.shape[1]}. "
                    f"Please check config['out_channel']."
                )
            out = mix_repr * spatial_repr

        return out

class IPDFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        return torch.stack(ipd_list, dim=1) # (B, N, F, T)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)

class CDFFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        return torch.cos(IPD - TPD)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class SDFFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1)
        
        _, _, F_dim, _ = Y.shape
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)
        
        return torch.sin(IPD - TPD)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)

class DSTFTFeature(BaseSpatialFeature):
    def compute(self, Y, azi, ele, pairs=None):
        target_pairs = self._get_pairs(pairs)
        d_list = []
        
        for (i, j) in target_pairs:
            diff = Y[:, i] - Y[:, j]
        
            d_list.append(diff.real)
            d_list.append(diff.imag)

        return torch.stack(d_list, dim=1)

    def post(self, mix_repr, spatial_repr):
        return torch.cat([mix_repr, spatial_repr], dim=1)


class SpatialFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ===== Default Config =====
        DEFAULT_CONFIG = {
            "geometry": {
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 512,
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.033333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [[0, 1], [1, 2], [2, 3], [0, 3]], 
            "features": {
                "ipd": {"enabled": False},
                "cdf": {"enabled": False},
                "sdf": {"enabled": False},
                "delta_stft": {"enabled": False},
                "cyc_doaemb":{
                    "enabled": False,
                    "cyc_alpha": 20,
                    "cyc_dimension": 40,
                    "use_ele": True,
                    "out_channel": 1
                }
            }
        }
        self.config = deep_update(DEFAULT_CONFIG, config)
        geo_cfg = self.config['geometry']
        
        freq_bins = geo_cfg['n_fft'] // 2 + 1
        freq_vec = torch.linspace(0, geo_cfg['fs'] / 2, freq_bins)
        
        if 'mic_coords' in geo_cfg:
            mic_pos = torch.tensor(geo_cfg['mic_coords'])
        else:
            M = 4 
            spacing = geo_cfg['mic_spacing']
            mic_pos = torch.zeros(M, 3)
            mic_pos[:, 0] = torch.arange(M) * spacing

        geometry_ctx = {
            'mic_pos': mic_pos,
            'omega_over_c': 2 * math.pi * freq_vec / geo_cfg['c']
        }
        
        self.features = nn.ModuleDict()
        self.default_pairs = self.config['pairs']
        feat_cfg = self.config['features']
        
        if feat_cfg['ipd']['enabled']:
            self.features['ipd'] = IPDFeature({'pairs': self.default_pairs}, geometry_ctx)
        
        if feat_cfg['cdf']['enabled']:
            self.features['cdf'] = CDFFeature({'pairs': self.default_pairs}, geometry_ctx)

        if feat_cfg['sdf']['enabled']:
            self.features['sdf'] = SDFFeature({'pairs': self.default_pairs}, geometry_ctx)

        if feat_cfg['delta_stft']['enabled']:
            self.features['delta_stft'] = DSTFTFeature({'pairs': self.default_pairs}, geometry_ctx)
            
        if feat_cfg['cyc_doaemb']['enabled']:
            self.features['cyc_doaemb']= CycEncoder(feat_cfg['cyc_doaemb'])

    def compute_all(self, Y, azi, ele=None, pairs=None):
        if ele is None:
            ele = torch.zeros_like(azi)
        
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(Y=Y, azi=azi, ele=ele, pairs=pairs)
            
        return out
    def post_all(self, mix_repr, feature_dict):
        current_feat = mix_repr
    
        feat_cfg = self.config['features']
        
        for name in feat_cfg:
            sub_cfg = feat_cfg[name]
            
            if not sub_cfg.get('enabled', False):
                continue
            
            if name in self.features and name in feature_dict:
                module = self.features[name]
                raw_data = feature_dict[name]
                current_feat = module.post(current_feat, raw_data)
        
        return current_feat