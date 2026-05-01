import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update
from wesep.modules.common.norm import AmplitudeNorm, EnergyNorm


class TSE_NBC2_SPK_SPATIAL(nn.Module):

    def __init__(self, config):
        super().__init__()

        # --- 1. top model setting ---
        self.full_input = config.get("full_input", True)

        # --- 2. Merge Configs ---
        # 2.1 Spatial Configs
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
                "Multiply_emb": {
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config": {
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40,
                    },
                    "use_ele": True,
                    "out_channel": 1
                },
            }
        }
        self.spatial_configs = deep_update(spatial_configs,
                                           config.get('spatial', {}))

        # 2.2 Separator Configs
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

        # 2.3 Speaker Configs (Aligning with NBC2 dim_hidden)
        spk_configs = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": self.sep_configs["win"],
                    "hop": self.sep_configs["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": False,
                    "enc_dim": self.sep_configs["win"] // 2 + 1,
                    "emb_dim": self.sep_configs["dim_hidden"] // 2,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": self.sep_configs["dim_hidden"],
                    "atten_dim": self.sep_configs["dim_hidden"]
                },
                "spkemb": {
                    "enabled": False,
                    "mix_dim": self.sep_configs["dim_hidden"]
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": 16000
                },
            },
        }
        self.spk_configs = deep_update(spk_configs, config.get('speaker', {}))

        # --- 3. Dynamic Input Size Calculation ---
        n_pairs = len(self.spatial_configs['pairs'])

        if self.full_input:
            self.sep_configs["spec_dim"] = 2 * len(
                self.spatial_configs['geometry']['mic_coords'])

        # Add spatial dims
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
        if self.spatial_configs["features"]["Multiply_emb"]["enabled"]:
            self.spatial_configs['features']['Multiply_emb'][
                'out_channel'] = self.sep_configs["dim_hidden"]
            self.spatial_configs['features']['Multiply_emb'][
                'num_encoder'] = self.sep_configs["n_layers"]

        # Add speaker dims
        if self.spk_configs["features"]["usef"]["enabled"]:
            self.sep_configs["spec_dim"] += self.spk_configs["features"][
                "usef"]["emb_dim"] * 2
        if self.spk_configs["features"]["tfmap"]["enabled"]:
            self.sep_configs["spec_dim"] += 1

        # Fix context band param for NBC2 (which relies on F bins instead of nband)
        if self.spk_configs["features"]["context"]["enabled"]:
            self.spk_configs["features"]["context"][
                "band"] = self.sep_configs["win"] // 2 + 1

        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        self.spk_ft = SpeakerFrontend(self.spk_configs)

        # --- 6. Instantiate Norm Modules ---
        self.A_norm = AmplitudeNorm()  # 保留以供未来空间特征剥离时使用
        self.E_norm = EnergyNorm()  # ★ 现阶段的全局核心 Norm

    def forward(self, mix, cue):
        # Unpack Multi-modal Cues
        enroll = cue[0].squeeze(1)
        spatial_cue = cue[1]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]

        wav_enroll = enroll
        # print(enroll.shape)
        wav_mix = mix

        ###########################################################
        # C0. Feature: listen (Time-domain manipulation)
        if self.spk_configs['features']['listen']['enabled']:
            B, M, T = wav_mix.shape
            processed_channels = []
            for m in range(M):
                processed_ch = self.spk_ft.listen.compute(
                    wav_enroll, wav_mix[:, m, :])
                processed_channels.append(processed_ch)
            wav_mix = torch.stack(processed_channels, dim=1)

        # S1. Convert into frequency-domain
        spec_complex = self.sep_model.stft(wav_mix)[-1]

        # S2. ★ 核心切换：全局使用 E-norm
        spec_norm, norm_scale = self.E_norm(spec_complex)

        # S3. Concat real and imag, split to subbands
        spec_feat = None
        if self.full_input:
            spec_feat = torch.cat([spec_norm.real, spec_norm.imag], dim=1)
        else:
            spec_feat = torch.stack(
                [spec_norm[:, 0].real, spec_norm[:, 0].imag], dim=1)

        # =========================================================
        # ★ 提取 Enrollment 频域特征并执行严格的 E-norm (为声纹提取护航)
        # =========================================================
        enroll_spec_complex = self.sep_model.stft(wav_enroll)[-1]
        enroll_spec_norm, _ = self.E_norm(enroll_spec_complex)

        #######################################################
        # C1. Feature: usef (Speaker)
        if self.spk_configs['features']['usef']['enabled']:
            # ★ 喂给下游的全部是消除能量波动后的 enroll_spec_norm
            enroll_spec_RI = torch.stack(
                [enroll_spec_norm.real, enroll_spec_norm.imag],
                1)  # (B, 2, F, T)
            spec_RI_single = torch.stack(
                [spec_norm[:, 0].real, spec_norm[:, 0].imag], 1)
            enroll_usef, mix_usef = self.spk_ft.usef.compute(
                enroll_spec_RI, spec_RI_single)
            usef_feat = self.spk_ft.usef.post(mix_usef, enroll_usef)
            spec_feat = torch.cat([spec_feat, usef_feat], dim=1)

        # C2. Feature: tfmap (Speaker)
        if self.spk_configs['features']['tfmap']['enabled']:
            enroll_mag = torch.abs(enroll_spec_norm)
            enroll_tfmap = self.spk_ft.tfmap.compute(
                enroll_mag, torch.abs(spec_norm[:, 0]))
            spec_feat = self.spk_ft.tfmap.post(spec_feat,
                                               enroll_tfmap.unsqueeze(1))

        #######################################################
        # Early Spatial Features
        # (注意：目前这里的空间特征接口吃的是 E-norm 的数据，相对相位会被保留，但绝对幅度关系会被改变)
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
        encode_features = self.sep_model.encoder(
            spec_feat)  # Conv: (B, dim_hidden, F, T)

        # C3. Feature: context (Speaker Frame-level Latent Modulation)
        if self.spk_configs['features']['context']['enabled']:
            enroll_context = self.spk_ft.context.compute(
                wav_enroll)  # (B, F_e, T_e)

            # ★ 1. 维度伪装：对齐 BSRNN 格式 (B, C, F, T) -> (B, F, C, T)
            encode_features = encode_features.permute(0, 2, 1, 3)

            # ★ 2. 注入特征：此时 dim=2 刚好是 C (dim_hidden)，完美满足 context 内部的形状断言
            encode_features = self.spk_ft.context.post(encode_features,
                                                       enroll_context)

            # ★ 3. 维度还原：变回 NBC2 格式 (B, F, C, T) -> (B, C, F, T)
            encode_features = encode_features.permute(0, 2, 1, 3)

        # C4. Feature: spkemb (Speaker Global Latent Modulation)
        if self.spk_configs['features']['spkemb']['enabled']:
            enroll_emb = self.spk_ft.spkemb.compute(wav_enroll)  # (B, F_e)

            # ★ 注意：既然伪装成了 BSRNN，这里的 unsqueeze 也要跟 BSRNN 完全一样！
            # 扩展为 (B, 1, F_e, 1)，以便在 (B, F, C, T) 上进行完美广播
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)

            # 同样进行 伪装 -> 运算 -> 还原 的三步走
            encode_features = encode_features.permute(0, 2, 1, 3)
            encode_features = self.spk_ft.spkemb.post(encode_features,
                                                      enroll_emb)
            encode_features = encode_features.permute(0, 2, 1, 3)

        # Separation Blocks
        for idx, m in enumerate(self.sep_model.sa_layers):  # nbc2_block
            if self.spatial_configs['features']['Multiply_emb']['enabled']:
                cyc_doaemb = self.spatial_ft.features['Multiply_emb'].compute(
                    azi=azi_rad, ele=ele_rad, layer_idx=idx)
                encode_features = self.spatial_ft.features[
                    'Multiply_emb'].post(encode_features,
                                         cyc_doaemb,
                                         layer_idx=idx)
            encode_features, _ = m(encode_features)

        est_spec_feat = self.sep_model.decoder(encode_features)

        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        # ★ inverse 切换为 E-norm 的逆运算
        est_spec = self.E_norm.inverse(est_spec, norm_scale)

        est_wav = self.sep_model.istft(est_spec)

        ###########################################################
        # C0. Feature: listen post
        if self.spk_configs['features']['listen']['enabled']:
            est_wav = self.spk_ft.listen.post(est_wav)  # (B, T)

        return est_wav
