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
class PosteriorMaskFeature(BaseSpatialFeature):
    """
    深度后验交互特征 (对应方案一)
    使用 TPD (具备频率维度) 作为 DOA 的表征，与 IPD 在特征空间进行相似度交互，生成软掩码。
    """
    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)
        
        self.enabled = config.get('enabled', False)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.fusion = config.get('fusion_type', 'multiply') # 支持 'multiply' 或 'concat'
        
        # 输入通道为 pairs 的 2 倍 (因为我们要输入 cos(相位) 和 sin(相位) 避免相位卷绕)
        self.num_pairs = len(self._get_pairs(None))
        in_channels = self.num_pairs * 2
        
        # 1. Mixture Encoder (处理实际观测到的 IPD)
        self.proj_mix = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.PReLU()
        )
        
        # 2. DOA Encoder (处理带有频率维度的理论 TPD)
        self.proj_doa = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.PReLU()
        )
        
        # 3. 后验掩码生成器 (处理交互后的特征)
        self.posterior_refiner = nn.Sequential(
            # 使用 3x3 卷积在时频域进行平滑
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.PReLU(),
            # 降维到 1 个通道，生成 0~1 的可信度概率掩码 (Attention Map)
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def compute(self, Y, azi, ele=None, pairs=None):
        if not self.enabled:
            return None
            
        target_pairs = self._get_pairs(pairs)
        B, M, F_dim, T_dim = Y.shape
        
        # --- A. 获取 Mixture 的观测证据 (IPD) ---
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        IPD = torch.stack(ipd_list, dim=1) # (B, P, F, T)
        
        # 展开为 cos 和 sin -> (B, 2P, F, T)
        mix_feat = torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1)
        
        # --- B. 获取 DOA 的物理提示 (TPD) ---
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs) # (B, P, F, 1)
        # 展开为 cos 和 sin -> (B, 2P, F, 1)
        doa_feat = torch.cat([torch.cos(TPD), torch.sin(TPD)], dim=1)
        
        # --- C. 深度交互 (Interaction) ---
        # 映射到相同的隐藏特征空间
        M_x = self.proj_mix(mix_feat) # (B, hidden_dim, F, T)
        M_c = self.proj_doa(doa_feat) # (B, hidden_dim, F, 1)
        
        # 点积/相乘交互 (M_c 会在 T 维度自动 Broadcasting)
        # 这捕捉了观测 IPD 与目标 TPD 在特征空间的匹配程度
        interaction = M_x * M_c 
        
        # 生成后验掩码 z (B, 1, F, T)
        z_mask = self.posterior_refiner(interaction)
        
        return z_mask

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None:
            return mix_repr
            
        if self.fusion == "multiply":
            # mix_repr 可能是 (B, C, F, T)，spatial_repr 是 (B, 1, F, T)
            # Broadcasting 自动生效，充当门控机制
            return mix_repr * spatial_repr
            
        elif self.fusion == "concat":
            # 作为一个额外的 1 通道特征拼接到主特征上
            return torch.cat([mix_repr, spatial_repr], dim=1)
            
        return mix_repr

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
                },
                "posterior_mask": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "fusion_type": "multiply" # 推荐使用 multiply 作为门控
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
        if feat_cfg.get('posterior_mask', {}).get('enabled', False):
            pm_cfg = deep_update({'pairs': self.default_pairs}, feat_cfg['posterior_mask'])
            self.features['posterior_mask'] = PosteriorMaskFeature(pm_cfg, geometry_ctx)    

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