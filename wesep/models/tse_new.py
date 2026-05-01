import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update
from wesep.modules.common.norm import AmplitudeNorm, EnergyNorm


class TSE_NEW(nn.Module):

    def __init__(self, config):
        super().__init__()

        # --- 1. top model setting ---
        self.full_input = config.get("full_input", True)

        # --- 2. Merge Configs ---
        spatial_configs = {
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
                    [-0.05, 0.0, 0.0],  # Mic 0
                    [-0.01666667, 0.0, 0.0],  # Mic 1
                    [0.01666667, 0.0, 0.0],  # Mic 2
                    [0.05, 0.0, 0.0],  # Mic 3
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

                # 深度后验掩码配置 (早期物理融合)
                "posterior_mask": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "fusion_type": "multiply",
                    "num_encoder": 1
                },

                # 隐空间混合自适应帧级特征配置 (晚期动态融合)
                "latent_mixture_adaptive": {
                    "enabled": False,
                    "num_encoder": 1,
                    "hidden_dim": 64,
                    "enc_channels": 96,
                    "fusion_type": "multiply",
                    "encoding_config": {
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40
                    },
                    "use_ele": True
                },

                # >>> 新增：SoundCompass 完全体旁路融合模块 <<<
                "soundcompass_fusion": {
                    "enabled": False,
                    "num_encoder": 1,
                    "enc_channels": 96,  # 动态保险丝会强制同步它
                    "use_ele": True,
                    "encoding_config": {
                        "encoding": "sh",
                        "sh_order": 5
                    }
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs,
                                           config.get('spatial', {}))

        block_kwargs = {
            'n_heads': 2,
            'dropout': 0.1,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': 257,
                'share_along_sequence_dim': False,
            },
        }

        sep_configs = dict(
            win=512,
            stride=256,
            spec_dim=2,  # for only ref-channel input
            n_spk=1,
            n_layers=8,
            dim_hidden=96,
            dim_ffn=96 * 2,
            block_kwargs=block_kwargs,
        )
        self.sep_configs = deep_update(sep_configs,
                                       config.get('separator', {}))

        # ==========================================================
        # ★ 工程保险丝：强制同步晚期特征的通道数与 NBC2Encoder 维度一致
        # ==========================================================
        if self.spatial_configs["features"].get("latent_mixture_adaptive",
                                                {}).get("enabled", False):
            self.spatial_configs["features"]["latent_mixture_adaptive"][
                "enc_channels"] = self.sep_configs["dim_hidden"]

        if self.spatial_configs["features"].get("soundcompass_fusion",
                                                {}).get("enabled", False):
            self.spatial_configs["features"]["soundcompass_fusion"][
                "enc_channels"] = self.sep_configs["dim_hidden"]

        # --- 3. Dynamic Input Size Calculation ---
        ### spec_feat dim calculation
        n_pairs = len(self.spatial_configs['pairs'])
        if self.full_input:
            self.sep_configs["spec_dim"] = 2 * len(
                self.spatial_configs['geometry']['mic_coords'])

        if self.spatial_configs["features"]["ipd"]["enabled"]:
            self.sep_configs["spec_dim"] += n_pairs * self.spatial_configs[
                "features"]["ipd"]["num_encoder"]
        if self.spatial_configs["features"]["cdf"]["enabled"]:
            self.sep_configs["spec_dim"] += n_pairs * self.spatial_configs[
                "features"]["cdf"]["num_encoder"]
        if self.spatial_configs["features"]["sdf"]["enabled"]:
            self.sep_configs["spec_dim"] += n_pairs * self.spatial_configs[
                "features"]["sdf"]["num_encoder"]
        if self.spatial_configs["features"]["delta_stft"]["enabled"]:
            self.sep_configs["spec_dim"] += 2 * n_pairs * self.spatial_configs[
                "features"]["delta_stft"]["num_encoder"]

        # 注意：posterior_mask, latent_mixture_adaptive, soundcompass_fusion 都是内部门控/旁路操作，不改变输入通道数

        # 注意：posterior_mask, latent_mixture_adaptive, soundcompass_fusion 都是内部门控/旁路操作，不改变输入通道数

        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)

        # --- 6. Instantiate other ---
        self.A_norm = AmplitudeNorm()
        self.E_norm = EnergyNorm()

    # ★ 核心改动 1：增加 target_wav 参数
    def forward(self, mix, cue, target_wav=None):
        # input shape: (B, C, T)
        spatial_cue = cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]

        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(mix)[-1]

        # S2. A-norm
        spec_norm, norm_scale = self.A_norm(spec)

        # S3. Concat real and imag, split to subbands
        spec_feat = None
        if self.full_input:
            spec_feat = torch.cat([spec_norm.real, spec_norm.imag], dim=1)
        else:
            spec_feat = torch.stack(
                [spec_norm[:, 0].real, spec_norm[:, 0].imag], dim=1)

        #######################################################
        # Level 1: 早期物理特征融合 (Early Fusion)
        if self.spatial_configs['features']['ipd']['enabled']:
            ipd_feature = self.spatial_ft.features['ipd'].compute(Y=spec_norm)
            spec_feat = self.spatial_ft.features['ipd'].post(
                spec_feat, ipd_feature)

        if self.spatial_configs['features']['cdf']['enabled']:
            cdf_feature = self.spatial_ft.features['cdf'].compute(Y=spec_norm,
                                                                  azi=azi_rad,
                                                                  ele=ele_rad)
            spec_feat = self.spatial_ft.features['cdf'].post(
                spec_feat, cdf_feature)

        if self.spatial_configs['features']['sdf']['enabled']:
            sdf_feature = self.spatial_ft.features['sdf'].compute(Y=spec_norm,
                                                                  azi=azi_rad,
                                                                  ele=ele_rad)
            spec_feat = self.spatial_ft.features['sdf'].post(
                spec_feat, sdf_feature)

        if self.spatial_configs['features']['delta_stft']['enabled']:
            dstft_feature = self.spatial_ft.features['delta_stft'].compute(
                Y=spec_norm)
            spec_feat = self.spatial_ft.features['delta_stft'].post(
                spec_feat, dstft_feature)

        ####################################################
        # ★ 恢复被误删的主干特征编码器 ★
        encode_features = self.sep_model.encoder(
            spec_feat)  # Conv: (B, dim_hidden, F, T)

        # 应用深度后验掩码门控 (Early Bottleneck Gating)
        if self.spatial_configs['features'].get('posterior_mask',
                                                {}).get('enabled', False):
            posterior_mask_feature = self.spatial_ft.features[
                'posterior_mask'].compute(Y=spec_norm,
                                          azi=azi_rad,
                                          ele=ele_rad)
            encode_features = self.spatial_ft.features['posterior_mask'].post(
                encode_features, posterior_mask_feature)

        # >>> Level 2 晚期融合 A：隐空间动态特征门控 (Latent Gating) <<<
        if self.spatial_configs['features'].get('latent_mixture_adaptive',
                                                {}).get('enabled', False):
            latent_spatial_repr = self.spatial_ft.features[
                'latent_mixture_adaptive'].compute(Y=None,
                                                   azi=azi_rad,
                                                   ele=ele_rad)
            encode_features = self.spatial_ft.features[
                'latent_mixture_adaptive'].post(
                    mix_repr=encode_features, spatial_repr=latent_spatial_repr)

        if self.spatial_configs['features'].get('soundcompass_fusion',
                                                {}).get('enabled', False):
            # 1. 学生轨道：永远从 Mixture 提取物理相位
            sc_spatial_repr_mix = self.spatial_ft.features[
                'soundcompass_fusion'].compute(Y=spec_norm,
                                               azi=azi_rad,
                                               ele=ele_rad)

            # 3. 学生施法：主干网络真正接受的调制 (脏相位)
            encode_features = self.spatial_ft.features[
                'soundcompass_fusion'].post(mix_repr=encode_features,
                                            spatial_repr=sc_spatial_repr_mix)
        ####################################################

        # 送入彼此隔离的窄带处理核心块
        for idx, m in enumerate(self.sep_model.sa_layers):
            encode_features, _ = m(encode_features)

        est_spec_feat = self.sep_model.decoder(encode_features)

        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        # inverse A-norm
        est_spec = self.A_norm.inverse(est_spec, norm_scale)

        est_wav = self.sep_model.istft(est_spec)

        return est_wav
