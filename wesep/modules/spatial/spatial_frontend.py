import torch
import torch.nn as nn
import math
import copy
from wesep.modules.common.deep_update import deep_update
from wesep.modules.spatial.pos_encoding import PosEncodingFactory
from wesep.modules.common.norm import AdaNorm2d


class BaseSpatialFeature(nn.Module):

    def __init__(self, config, geometry_ctx=None):
        super().__init__()
        self.config = config
        self.default_pairs = config.get('pairs', None)
        if geometry_ctx is not None:
            self.register_buffer('mic_pos', geometry_ctx['mic_pos'])
            self.register_buffer('omega_over_c', geometry_ctx['omega_over_c'])

    def _get_pairs(self, pairs_arg):
        if pairs_arg is not None:
            return pairs_arg
        if self.default_pairs is not None:
            return self.default_pairs
        raise ValueError(
            f"{self.__class__.__name__}: No pairs provided in arg or config.")

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

        TPD = self.omega_over_c.view(
            1, 1, F_dim, 1) * dist_delay.unsqueeze(-1).unsqueeze(-1)
        return TPD

    def compute(self, azi=None, ele=None, Y=None, pairs=None):
        raise NotImplementedError

    def post(self, mix_repr, spatial_repr):
        raise NotImplementedError


class SpatialEncoderGroup(nn.Module):

    def __init__(self, base_encoder: nn.Module, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_encoder) for _ in range(num_layers)])

    def compute(self, azi=None, ele=None, Y=None, pairs=None, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].compute(azi=azi,
                                                  ele=ele,
                                                  Y=Y,
                                                  pairs=pairs)

        results = []
        for layer in self.layers:
            results.append(layer.compute(azi=azi, ele=ele, Y=Y, pairs=pairs))
        return results

    def post(self, mix_repr, spatial_repr, layer_idx=None):
        if layer_idx is not None:
            return self.layers[layer_idx].post(mix_repr, spatial_repr)

        if self.num_layers == 1:
            repr_item = spatial_repr[0] if isinstance(spatial_repr,
                                                      list) else spatial_repr
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
        self.encoder, self.enc_dim = PosEncodingFactory.create(
            encoding_cfg, self.use_ele)

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
                ele_input = torch.abs(ele) if self.encoding_type in [
                    "oh", "onehot"
                ] else ele
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


class LatentMixtureAdaptiveFeature(BaseSpatialFeature):
    """
    隐空间混合自适应帧级特征 (Fair Version aligned with DSENet)
    - compute: 提取静态 DOA 编码，映射到声学隐空间。
    - post: 接收主干声学特征 (mix_repr)，与 DOA 拼接后通过轻量网络提炼掩码。
            ★ 公平性对齐: 放弃 Sigmoid，使用与 DSENet 完全一致的 PReLU 收尾，实现无界缩放。
    """

    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)

        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return

        self.hidden_dim = config.get('hidden_dim', 64)
        self.enc_channels = config.get('enc_channels', 192)
        self.fusion_type = config.get('fusion_type', 'multiply')
        self.use_ele = config.get('use_ele', False)

        self.f_dim = geometry_ctx['omega_over_c'].shape[
            0] if geometry_ctx is not None else 257

        # 1. DOA 编码器初始化
        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(
            encoding_cfg, self.use_ele)
        self.encoding_type = encoding_cfg.get("encoding", "cyc")

        # 2. 空间投影器 (Spatial Projector)
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.enc_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.PReLU())

        # 3. 频域物理模式解码器
        self.freq_proj = nn.Sequential(nn.Linear(self.f_dim, self.hidden_dim),
                                       nn.PReLU(),
                                       nn.Linear(self.hidden_dim, 1))

        # ==========================================================
        # ★ 绝对公平对齐的神经网络提炼器 (The SubNetwork) ★
        # ==========================================================
        self.dynamic_generator = nn.Sequential(
            nn.Conv1d(self.enc_channels + self.hidden_dim,
                      self.hidden_dim,
                      kernel_size=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.PReLU(),
            nn.Conv1d(self.hidden_dim, self.enc_channels, kernel_size=1),
            # ★ 核心修改：使用 PReLU 替代 Sigmoid，对齐 DSENet 的 Linear->LN->PReLU 结构
            nn.PReLU() if self.fusion_type == 'multiply' else nn.Identity())

    def compute(self, azi, ele=None, Y=None, pairs=None):
        if not self.enabled:
            return None

        is_missing = (azi <= -998.0)
        safe_azi = torch.where(is_missing, torch.zeros_like(azi), azi)

        if safe_azi.dim() == 1: safe_azi = safe_azi.unsqueeze(1)
        if ele is not None:
            safe_ele = torch.where(ele <= -998.0, torch.zeros_like(ele), ele)
            if safe_ele.dim() == 1: safe_ele = safe_ele.unsqueeze(1)
        else:
            safe_ele = None

        if self.encoding_type == "exp":
            doa_enc = self.encoder(safe_azi, safe_ele)
        else:
            doa_enc = self.encoder(safe_azi)
            if self.use_ele and safe_ele is not None:
                ele_input = torch.abs(safe_ele) if self.encoding_type in [
                    "oh", "onehot"
                ] else safe_ele
                doa_enc = torch.cat([doa_enc, self.encoder(ele_input)], dim=-1)

        doa_enc = doa_enc.squeeze(1)  # (B, enc_dim)

        spatial_repr = self.spatial_proj(doa_enc)  # (B, hidden_dim)

        if is_missing.dim() == 2:
            is_missing = is_missing[:, 0]
        is_missing = is_missing.view(-1, 1)
        spatial_repr = torch.where(is_missing, torch.zeros_like(spatial_repr),
                                   spatial_repr)

        return spatial_repr

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None:
            return mix_repr

        B, C_enc, F_dim, T_dim = mix_repr.shape

        scene_t = mix_repr.permute(0, 1, 3, 2)
        scene_t = self.freq_proj(scene_t)  # (B, C_enc, T_dim, 1)
        scene_t = scene_t.squeeze(-1)  # (B, C_enc, T_dim)

        doa_t = spatial_repr.unsqueeze(-1).expand(-1, -1, T_dim)

        combined_feat = torch.cat([scene_t, doa_t], dim=1)

        # 此时输出的 dynamic_mask 是经过 PReLU 激活的，边界与 DSENet 一致
        dynamic_mask = self.dynamic_generator(combined_feat)
        dynamic_mask = dynamic_mask.unsqueeze(2)

        if self.fusion_type == "multiply":
            return mix_repr * dynamic_mask
        elif self.fusion_type == "add":
            return mix_repr + dynamic_mask
        elif self.fusion_type == "concat":
            mask_expand = dynamic_mask.expand(-1, -1, F_dim, -1)
            return torch.cat([mix_repr, mask_expand], dim=1)

        return mix_repr


class TimeVariantMultiplyFeature(BaseSpatialFeature):

    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)

        self.out_channels = config['out_channel']
        self.use_ele = config.get('use_ele', False)

        encoding_cfg = config.get("encoding_config", {})
        self.encoder, self.enc_dim = PosEncodingFactory.create(
            encoding_cfg, self.use_ele)
        self.encoding_type = encoding_cfg.get("encoding", "cyc")

        self.mlp = nn.Sequential(nn.Linear(self.enc_dim, self.out_channels),
                                 nn.LayerNorm(self.out_channels), nn.PReLU())

    def compute(self, azi, ele=None, Y=None, pairs=None):
        if azi.dim() == 1: azi = azi.unsqueeze(1)
        if ele is not None and ele.dim() == 1: ele = ele.unsqueeze(1)

        if self.encoding_type == "exp":
            doa_enc = self.encoder(azi, ele)
        else:
            doa_enc = self.encoder(azi)
            if self.use_ele and ele is not None:
                ele_input = torch.abs(ele) if self.encoding_type in [
                    "oh", "onehot"
                ] else ele
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
                f"Mix: {mix_repr.shape[1]}, Spatial: {spatial_repr.shape[1]}.")
        return mix_repr * spatial_repr


class IPDFeature(BaseSpatialFeature):

    def compute(self, Y, azi=None, ele=None, pairs=None):
        target_pairs = self._get_pairs(pairs)
        ipd_list = []
        for (i, j) in target_pairs:
            diff = Y[:, i].angle() - Y[:, j].angle()
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            ipd_list.append(diff)
        return torch.stack(ipd_list, dim=1)  # (B, N, F, T)

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
    深度后验交互特征 (V2: IPD + \Delta STFT 双流架构)
    将观测到的 相位差(IPD) 和 归一化复数差(\Delta STFT) 分别与目标 TPD 进行特征交互，
    融合多维度空间线索生成高精度的软掩码。
    """

    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)

        self.enabled = config.get('enabled', False)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.fusion = config.get('fusion_type',
                                 'multiply')  # 支持 'multiply' 或 'concat'

        if not self.enabled:
            return

        self.num_pairs = len(self._get_pairs(None))
        in_channels = self.num_pairs * 2  # IPD(cos,sin) 或 Delta(real,imag) 都是 2P

        # ==================== 分支 1: IPD 交互流 ====================
        self.proj_mix_ipd = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim), nn.PReLU())
        self.proj_doa_ipd = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim), nn.PReLU())

        # ==================== 分支 2: \Delta STFT 交互流 ====================
        self.proj_mix_delta = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim), nn.PReLU())
        self.proj_doa_delta = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim), nn.PReLU())

        # ==================== 后验掩码融合生成器 ====================
        # 接收双流交互的拼接特征 (hidden_dim * 2)
        self.posterior_refiner = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2,
                      self.hidden_dim,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.PReLU(),
            # 降维到 1 个通道，生成 0~1 的可信度概率掩码
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            nn.Sigmoid())

    def compute(self, Y, azi, ele=None, pairs=None):
        if not self.enabled:
            return None

        target_pairs = self._get_pairs(pairs)
        B, M, F_dim, T_dim = Y.shape
        eps = 1e-8

        ipd_list = []
        delta_real = []
        delta_imag = []

        # --- 1. 提取观测证据 (IPD & 归一化 \Delta STFT) ---
        for (i, j) in target_pairs:
            # 提取 IPD
            diff_angle = Y[:, i].angle() - Y[:, j].angle()
            diff_angle = torch.remainder(diff_angle + math.pi,
                                         2 * math.pi) - math.pi
            ipd_list.append(diff_angle)

            # 提取 \Delta STFT 并进行能量归一化 (极其关键，消除绝对音量影响)
            diff_stft = Y[:, i] - Y[:, j]
            mag_sum = Y[:, i].abs() + Y[:, j].abs() + eps
            norm_diff_stft = diff_stft / mag_sum

            delta_real.append(norm_diff_stft.real)
            delta_imag.append(norm_diff_stft.imag)

        IPD = torch.stack(ipd_list, dim=1)  # (B, P, F, T)
        mix_feat_ipd = torch.cat(
            [torch.cos(IPD), torch.sin(IPD)], dim=1)  # (B, 2P, F, T)

        mix_feat_delta = torch.cat(
            [torch.stack(delta_real, dim=1),
             torch.stack(delta_imag, dim=1)],
            dim=1)  # (B, 2P, F, T)

        # --- 2. 提取物理提示 (TPD 作为 DOA 表征) ---
        TPD = self._compute_tpd(azi, ele, F_dim, target_pairs)  # (B, P, F, 1)
        doa_feat = torch.cat([torch.cos(TPD), torch.sin(TPD)],
                             dim=1)  # (B, 2P, F, 1)

        # --- 3. 独立深度交互 (Independent Interaction) ---
        # IPD 流交互
        M_x_ipd = self.proj_mix_ipd(mix_feat_ipd)
        M_c_ipd = self.proj_doa_ipd(doa_feat)
        interaction_ipd = M_x_ipd * M_c_ipd  # (B, hidden_dim, F, T)

        # \Delta STFT 流交互
        M_x_delta = self.proj_mix_delta(mix_feat_delta)
        M_c_delta = self.proj_doa_delta(doa_feat)
        interaction_delta = M_x_delta * M_c_delta  # (B, hidden_dim, F, T)

        # --- 4. 多级特征融合与掩码生成 ---
        # 拼接双流交互特征: (B, hidden_dim * 2, F, T)
        fused_interaction = torch.cat([interaction_ipd, interaction_delta],
                                      dim=1)

        # 生成最终的后验掩码 z (B, 1, F, T)
        z_mask = self.posterior_refiner(fused_interaction)

        return z_mask

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None:
            return mix_repr

        if self.fusion == "multiply":
            return mix_repr * spatial_repr
        elif self.fusion == "concat":
            return torch.cat([mix_repr, spatial_repr], dim=1)

        return mix_repr


class MixtureQueryFusion(nn.Module):
    """
    借鉴 USEF-TSE 思想的融合模块：Mixture (Q) Query Target Spatial (K, V)
    采用 1x1 逐像素卷积交叉注意力，实现时间与频率上的软门控，避免梯度撕裂。
    """

    def __init__(self, mix_dim, spatial_dim, hidden_dim):
        super().__init__()
        # 降维投影，统一审查空间
        self.q_proj = nn.Conv2d(mix_dim, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(spatial_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(spatial_dim, hidden_dim, kernel_size=1)

        # 最终融合特征的通道投射
        self.out_proj = nn.Conv2d(hidden_dim, mix_dim, kernel_size=1)

    def forward(self, mix_repr, spatial_feat):
        # Q 来自主干特征，编码了真实的语音包络与能量 (B, hidden_dim, F, T)
        Q = self.q_proj(mix_repr)
        # K, V 来自合并后的目标空间特征，编码了指向性 (B, hidden_dim, F, T)
        K = self.k_proj(spatial_feat)
        V = self.v_proj(spatial_feat)

        # 逐像素核验：计算 Mixture 包络与空间指向的匹配置信度
        # 除以 sqrt(hidden_dim) 防止点积进入 Sigmoid 的饱和区
        attn_scores = torch.sum(Q * K, dim=1, keepdim=True) / (Q.shape[1]**0.5)

        # 生成严格在 0~1 之间的动态置信度掩码
        attn_weights = torch.sigmoid(attn_scores)

        # 提纯空间特征：斩断没有真实能量支撑的“虚假空间底噪”
        attended_spatial = V * attn_weights

        # 采用加法残差与主干网络融合，保证物理信息无损透传
        fused_out = mix_repr + self.out_proj(attended_spatial)

        return fused_out


# ==========================================================
# 阶段一核心组件：子带内的静态空间提纯
# ==========================================================
class SoundCompassSubbandProcessor(nn.Module):
    """
    严格处理 12-TET 子带内部的空间滤波，使用静态 DOA 提取候选空间特征
    """

    def __init__(self, spin_dim, dim_hidden, enc_dim):
        super().__init__()

        # 1. 空间声学特征底层编码
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(spin_dim, dim_hidden, kernel_size=1),
            AdaNorm2d(),  # 保持你原本的设计
            nn.PReLU())

        # 2. 静态角度查询：生成当前子带的 FiLM 参数
        self.direction_query_gen = nn.Linear(enc_dim, dim_hidden * 2)

        # 3. 目标特征初步解码
        self.target_decoder = nn.Sequential(
            nn.Conv2d(dim_hidden, dim_hidden, kernel_size=1), AdaNorm2d(),
            nn.PReLU())

    def forward(self, spin_sub, doa_enc):
        feat_enc = self.feature_encoder(spin_sub)

        # 生成通道级门控系数
        query_params = self.direction_query_gen(doa_enc)
        gamma, beta = torch.chunk(query_params, 2, dim=-1)

        gamma = gamma.view(feat_enc.shape[0], -1, 1, 1)
        beta = beta.view(feat_enc.shape[0], -1, 1, 1)

        # 利用 DOA 提取候选的 Target 空间特征 (无界特征)
        target_spatial_feat = (gamma * feat_enc + beta) + feat_enc

        dec_sub = self.target_decoder(target_spatial_feat)
        return dec_sub


# ==========================================================
# 主模块：完整的 SoundCompass 融合特征提取器
# ==========================================================
class SoundCompassFusionFeature(BaseSpatialFeature):
    """
    两阶段解耦版 SoundCompass 融合模块
    包含: SPIN + 12-TET + DOA FiLM (Stage 1) + Mixture Query Attention (Stage 2)
    """

    def __init__(self, config, geometry_ctx=None):
        super().__init__(config, geometry_ctx)

        self.enabled = config.get('enabled', False)
        if not self.enabled: return

        self.dim_hidden = config.get('enc_channels', 96)
        # ★ 新增：从 config 获取主干特征的通道数，默认设为 64
        self.mix_channels = config.get('mix_channels', 64)
        self.use_ele = config.get('use_ele', True)

        if geometry_ctx is not None and 'mic_pos' in geometry_ctx:
            M = geometry_ctx['mic_pos'].shape[0]
        else:
            M = 4
        self.spin_dim = (2 * M)**2

        encoding_cfg = config.get("encoding_config", {
            "encoding": "sh",
            "sh_order": 5
        })
        self.encoder, self.enc_dim = PosEncodingFactory.create(
            encoding_cfg, self.use_ele)
        self.encoding_type = encoding_cfg.get("encoding", "sh")

        # 获取真实的物理采样率和 FFT 点数用于 12-TET 计算
        fs = self.config.get('geometry', {}).get('fs', 16000)
        n_fft = self.config.get('geometry', {}).get('n_fft', 512)
        self.f_dim = geometry_ctx['omega_over_c'].shape[
            0] if geometry_ctx is not None else (n_fft // 2 + 1)

        # 划分 K=31 个重叠子带，并获取平滑交叉淡入淡出窗
        self.num_bands = 31
        self.subband_indices, self.subband_windows = self._generate_12tet_overlapping_bands(
            fs, n_fft, self.num_bands)

        # Stage 1: 实例化 31 个包含独立 FiLM 发生器的子带处理器
        self.subbands = nn.ModuleList([
            SoundCompassSubbandProcessor(self.spin_dim, self.dim_hidden,
                                         self.enc_dim)
            for _ in range(self.num_bands)
        ])

        # ★ Stage 2 新增：全局级别的 Mixture Query 融合模块 ★
        self.mix_query_fusion = MixtureQueryFusion(mix_dim=self.dim_hidden,
                                                   spatial_dim=self.dim_hidden,
                                                   hidden_dim=self.dim_hidden)

    def _generate_12tet_overlapping_bands(self, fs, n_fft, num_bands):
        # ... (与你原本完美的 12-TET 切分代码完全保持一致) ...
        f_min = fs / n_fft
        f_max = fs / 2.0
        z_min = 69.0 + 12.0 * math.log2(f_min / 440.0)
        z_max = 69.0 + 12.0 * math.log2(f_max / 440.0)
        z_points = torch.linspace(z_min, z_max, num_bands + 2)
        f_points = 440.0 * (2.0**((z_points - 69.0) / 12.0))
        bin_points = f_points * n_fft / fs
        bin_points = torch.round(bin_points).long()
        bin_points[0] = 0
        num_bins = n_fft // 2 + 1
        indices = []
        windows = []

        for i in range(num_bands):
            start = max(0, bin_points[i].item())
            end = min(num_bins, bin_points[i + 2].item() + 1)
            if end <= start: end = start + 1
            L = end - start
            win = torch.hann_window(L)
            indices.append((start, end))
            windows.append(win)
        return indices, windows

    def compute(self, azi, ele=None, Y=None, pairs=None):
        # ... (与你原本完美的 SPIN 和 DOA 计算代码完全保持一致) ...
        if not self.enabled: return None
        B, M, F, T = Y.shape

        Y_mag = torch.abs(Y)
        Y_norm = Y / (Y_mag + 1e-8)
        components = torch.cat([Y_norm.real, Y_norm.imag], dim=1)
        interaction = components.unsqueeze(2) * components.unsqueeze(1)
        spin_feat = interaction.view(B, self.spin_dim, F, T)

        is_missing = (azi <= -998.0)
        safe_azi = torch.where(is_missing, torch.zeros_like(azi), azi)
        if safe_azi.dim() == 1: safe_azi = safe_azi.unsqueeze(1)

        if ele is not None:
            safe_ele = torch.where(ele <= -998.0, torch.zeros_like(ele), ele)
            if safe_ele.dim() == 1: safe_ele = safe_ele.unsqueeze(1)
        else: safe_ele = torch.zeros_like(safe_azi)

        if self.encoding_type == "sh":
            doa_enc = self.encoder(safe_azi, safe_ele)
        else:
            doa_enc = self.encoder(safe_azi)
            if self.use_ele and ele is not None:
                ele_input = torch.abs(safe_ele) if self.encoding_type in [
                    "oh", "onehot"
                ] else safe_ele
                doa_enc = torch.cat([doa_enc, self.encoder(ele_input)], dim=-1)

        doa_enc = doa_enc.squeeze(1)
        if is_missing.dim() == 2: is_missing = is_missing[:, 0]
        is_missing = is_missing.view(-1, 1)
        doa_enc = torch.where(is_missing, torch.zeros_like(doa_enc), doa_enc)

        return {"spin": spin_feat, "doa": doa_enc}

    def post(self, mix_repr, spatial_repr):
        if spatial_repr is None: return mix_repr

        spin_feat = spatial_repr["spin"]
        doa_enc = spatial_repr["doa"]
        B, D, F, T = mix_repr.shape

        # 盛放 31 个子带合并后的候选空间特征
        merged_spatial = torch.zeros(B,
                                     self.dim_hidden,
                                     F,
                                     T,
                                     device=mix_repr.device)
        weight_sum = torch.zeros(1, 1, F, 1, device=mix_repr.device)

        # ------------------------------------------------------------------
        # Stage 1: Direction Query Spatial (子带并行处理)
        # ------------------------------------------------------------------
        for k in range(self.num_bands):
            start, end = self.subband_indices[k]
            win = self.subband_windows[k].to(mix_repr.device).view(1, 1, -1, 1)

            spin_sub = spin_feat[:, :, start:end, :]

            # 子带解码 (包含 DOA FiLM 调制)
            dec_sub = self.subbands[k](spin_sub, doa_enc)

            merged_spatial[:, :, start:end, :] += dec_sub * win
            weight_sum[:, :, start:end, :] += win

        # 归一化，得到全频带 Target Spatial 候选特征
        merged_spatial = merged_spatial / weight_sum.clamp(min=1e-8)

        # ------------------------------------------------------------------
        # Stage 2: Mixture Query Target Spatial (全局软融合)
        # ------------------------------------------------------------------
        # 利用主干网络的 mix_repr 对 merged_spatial 进行像素级置信度核验与提纯融合
        fused_output = self.mix_query_fusion(mix_repr, merged_spatial)

        # 完美输出！
        return fused_output


class SpatialFrontend(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ===== Default Config =====
        DEFAULT_CONFIG = {
            "geometry": {
                "n_fft":
                512,
                "hop_length":
                128,
                "win_length":
                512,
                "fs":
                16000,
                "c":
                343.0,
                "mic_spacing":
                0.033333,
                "mic_coords": [
                    [-0.05, 0.0, 0.0],
                    [-0.01666667, 0.0, 0.0],
                    [0.01666667, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
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
                    "encoding_config": {
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40
                    },
                    "use_ele": True,
                    "out_channel": 1
                },
                "InitStates_emb": {
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config": {
                        "encoding": "oh",
                        "emb_dim": 180
                    },
                    "hidden_size_f": 256,
                    "hidden_size_t": 256,
                    "use_ele": True
                },
                "posterior_mask": {
                    "enabled": False,
                    "hidden_dim": 32,
                    "fusion_type": "multiply"
                },
                "latent_mixture_adaptive": {
                    "enabled": False,
                    "num_encoder": 1,
                    "hidden_dim": 64,
                    "enc_channels": 192,
                    "fusion_type": "multiply",
                    "use_ele": True,
                    "encoding_config": {
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40
                    }
                },
                # ★ 新增：SoundCompass 融合特征配置
                "soundcompass_fusion": {
                    "enabled": False,
                    "num_encoder": 1,
                    "enc_channels": 96,  # 注意：需与你的分离主干 dim_hidden 对齐
                    "use_ele": True,
                    "encoding_config": {
                        "encoding": "sh",  # 默认使用球谐编码
                        "sh_order": 5
                    }
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
            'ipd': IPDFeature,  # 需确保之前有定义
            'cdf': CDFFeature,  # 需确保之前有定义
            'sdf': SDFFeature,  # 需确保之前有定义
            'delta_stft': DSTFTFeature,  # 需确保之前有定义
            'Multiply_emb': TimeVariantMultiplyFeature,
            'InitStates_emb': InitStatesFeature,
            'posterior_mask': PosteriorMaskFeature,  # 需确保之前有定义
            'latent_mixture_adaptive': LatentMixtureAdaptiveFeature,
            'soundcompass_fusion': SoundCompassFusionFeature  # ★ 挂载完成
        }

        for feat_name, sub_cfg in feat_cfg.items():
            if not sub_cfg.get('enabled', False): continue
            if feat_name not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown spatial feature in config: {feat_name}")

            num_encoder = sub_cfg.get('num_encoder', 1)
            sub_cfg_with_pairs = deep_update({'pairs': self.default_pairs},
                                             sub_cfg)
            base_module = FEATURE_REGISTRY[feat_name](sub_cfg_with_pairs,
                                                      geometry_ctx)
            self.features[feat_name] = SpatialEncoderGroup(
                base_module, num_encoder)  # 需确保之前有定义

    def compute_all(self, Y, azi, ele=None, pairs=None):
        if ele is None: ele = torch.zeros_like(azi)
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(Y=Y, azi=azi, ele=ele, pairs=pairs)
        return out
