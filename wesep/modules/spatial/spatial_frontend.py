import torch
import torch.nn as nn
import math
import copy
from wesep.modules.common.deep_update import deep_update
from wesep.modules.spatial.pos_encoding import PosEncodingFactory

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

    def compute(self, azi=None, ele=None, Y=None,pairs=None):
        raise NotImplementedError

    def post(self, mix_repr, spatial_repr):
        raise NotImplementedError
class SpatialEncoderGroup(nn.Module):
    def __init__(self, base_encoder: nn.Module, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(base_encoder) for _ in range(num_layers)])

    def compute(self, azi=None, ele=None, Y=None, pairs=None, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].compute(azi=azi, ele=ele, Y=Y, pairs=pairs)
            
        results = []
        for layer in self.layers:
            results.append(layer.compute(azi=azi, ele=ele, Y=Y, pairs=pairs))
        return results

    def post(self, mix_repr, spatial_repr, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].post(mix_repr, spatial_repr)
            
        if self.num_layers == 1:
            repr_item = spatial_repr[0] if isinstance(spatial_repr, list) else spatial_repr
            return self.layers[0].post(mix_repr, repr_item)
            
        out_repr = mix_repr
        for i, layer in enumerate(self.layers):
            out_repr = layer.post(out_repr, spatial_repr[i])
            
        return out_repr
class InitStatesFeature(BaseSpatialFeature): 
    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)
        
        self.hidden_size_f = config["hidden_size_f"]
        self.hidden_size_t = config["hidden_size_t"]
        self.use_ele = config.get("use_ele", False)
        
        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(encoding_cfg, self.use_ele)
        
        self.encoding_type = encoding_cfg.get("encoding", "oh")
        
        self.proj_band_h0 = nn.Linear(self.enc_dim, self.hidden_size_f)
        self.proj_band_c0 = nn.Linear(self.enc_dim, self.hidden_size_f)
        self.proj_comm_h0 = nn.Linear(self.enc_dim, self.hidden_size_t)
        self.proj_comm_c0 = nn.Linear(self.enc_dim, self.hidden_size_t)

    def compute(self, azi, ele=None, Y=None, pairs=None):
        if azi.dim() == 2: azi = azi[:, 0]
        if ele is not None and ele.dim() == 2: ele = ele[:, 0]
            
        if self.encoding_type == "exp":
            doa_enc = self.encoder(azi, ele)
        else:
            doa_enc = self.encoder(azi)
            if self.use_ele and ele is not None:
                ele_input = torch.abs(ele) if self.encoding_type in ["oh", "onehot"] else ele
                ele_enc = self.encoder(ele_input)
                doa_enc = torch.cat([doa_enc, ele_enc], dim=-1)

        return {
            "band_h0": self.proj_band_h0(doa_enc),
            "band_c0": self.proj_band_c0(doa_enc),
            "comm_h0": self.proj_comm_h0(doa_enc),
            "comm_c0": self.proj_comm_c0(doa_enc)
        }

    def post(self, mix_repr, spatial_repr):
        return mix_repr
        
class TimeVariantMultiplyFeature(BaseSpatialFeature): 
    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)
        
        self.out_channels = config['out_channel']
        self.use_ele = config.get('use_ele', False)
        
        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(encoding_cfg, self.use_ele)
        self.encoding_type = encoding_cfg.get("encoding", "cyc")
        
        self.mlp = nn.Sequential(
            nn.Linear(self.enc_dim, self.out_channels),
            nn.LayerNorm(self.out_channels),
            nn.PReLU()
        )
    def compute(self, azi, ele=None, Y=None, pairs=None):
        if azi.dim() == 1: azi = azi.unsqueeze(1)
        if ele is not None and ele.dim() == 1: ele = ele.unsqueeze(1)
        
        if self.encoding_type == "exp":
            doa_enc = self.encoder(azi, ele)
        else:
            doa_enc = self.encoder(azi)
            if self.use_ele and ele is not None:
                ele_input = torch.abs(ele) if self.encoding_type in ["oh", "onehot"] else ele
                ele_enc = self.encoder(ele_input)
                doa_enc = torch.cat([doa_enc, ele_enc], dim=-1)

        # Input: (B, T, enc_dim) -> Output: (B, T, out_channels)
        spatial_repr = self.mlp(doa_enc)

        # (B, T, C) -> Permute to (B, C, T) -> Unsqueeze to (B, C, 1, T)
        spatial_repr = spatial_repr.permute(0, 2, 1).unsqueeze(2)
        
        return spatial_repr

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None:
            return mix_repr
        if mix_repr.shape[1] != spatial_repr.shape[1]:
            raise ValueError(
                f"Fusion 'multiply' requires same channel dimensions. "
                f"Mix: {mix_repr.shape[1]}, Spatial: {spatial_repr.shape[1]}."
            )
        return mix_repr * spatial_repr
    
class IPDFeature(BaseSpatialFeature):
    def compute(self, Y, azi=None, ele=None, pairs=None):
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
    def compute(self, Y, azi, ele=None, pairs=None):
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
    def compute(self, Y, azi, ele=None, pairs=None):
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
    def compute(self, Y, azi=None, ele=None, pairs=None):
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
                "ipd": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "cdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "sdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "delta_stft": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "Multiply_emb": {
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40,
                    },
                    "use_ele": True,
                    "out_channel": 1
                },
                "InitStates_emb": {  
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "oh",
                        "emb_dim": 180,
                    },
                    "hidden_size_f": 256,
                    "hidden_size_t": 256,
                    "use_ele" : True
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
        
        FEATURE_REGISTRY = {
            'ipd': IPDFeature,
            'cdf': CDFFeature,
            'sdf': SDFFeature,
            'delta_stft': DSTFTFeature,
            'Multiply_emb': TimeVariantMultiplyFeature,
            'InitStates_emb': InitStatesFeature,
            'posterior_mask': PosteriorMaskFeature
        }
        
        for feat_name, sub_cfg in feat_cfg.items():
            if not sub_cfg.get('enabled', False):
                continue
                
            if feat_name not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown spatial feature in config: {feat_name}")
                
            num_encoder = sub_cfg.get('num_encoder', 1)
            
            sub_cfg_with_pairs = deep_update({'pairs': self.default_pairs}, sub_cfg)
            
            base_module = FEATURE_REGISTRY[feat_name](sub_cfg_with_pairs, geometry_ctx)
            
            self.features[feat_name] = SpatialEncoderGroup(base_module, num_encoder)
        
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