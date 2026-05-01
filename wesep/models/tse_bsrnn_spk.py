# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update


class TSE_BSRNN_SPK(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ===== Merge configs =====
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,  # For Separation (multiple output)
            spec_dim=2,  # For TSE feature, used in self.subband_norm
        )
        sep_configs = {**sep_configs, **config['separator']}
        spk_configs = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": sep_configs["win"],
                    "hop": sep_configs["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": sep_configs["causal"],
                    "enc_dim": sep_configs["win"] // 2 + 1,
                    "emb_dim": sep_configs["feature_dim"] // 2,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"],
                    "atten_dim": sep_configs["feature_dim"]
                },
                "spkemb": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"]
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": sep_configs["sr"]
                },
            },
            # 🌟 新增：SV Head 分类器配置默认值
            "sv_head": {
                "enabled": True,
                "num_speakers": 251
            }
        }
        self.spk_configs = deep_update(spk_configs, config['speaker'])

        # ===== Separator Loading =====
        if self.spk_configs["features"]["usef"]["enabled"]:
            sep_configs["spec_dim"] = self.spk_configs["features"]["usef"][
                "emb_dim"] * 2
        if self.spk_configs["features"]["tfmap"]["enabled"]:
            sep_configs["spec_dim"] = sep_configs["spec_dim"] + 1  #
        self.sep_model = BSRNN(**sep_configs)

        # ===== Speaker Loading =====
        if self.spk_configs["features"]["context"]["enabled"]:
            self.spk_configs["features"]["context"][
                "band"] = self.sep_model.nband  #
        self.spk_ft = SpeakerFrontend(self.spk_configs)

        # 🌟 修改：挂载全连接分类头 (SV Head) 并保留频率维度
        if self.spk_configs.get("sv_head", {}).get("enabled", False):
            if self.spk_configs["features"]["usef"]["enabled"]:
                # 算出 Freq 维度 (BSRNN 的频点数通常是 win // 2 + 1)
                f_dim = sep_configs["win"] // 2 + 1
                emb_dim = self.spk_configs["features"]["usef"]["emb_dim"]
                # 展平后的总维度：通道数 * 频率数
                in_dim = emb_dim * f_dim
            else:
                in_dim = 256  # 默认 fallback 维度
            self.sv_head = nn.Linear(
                in_features=in_dim,
                out_features=self.spk_configs["sv_head"]["num_speakers"])

    def forward(self, mix, enroll):
        """
        Args:
            mix:  Tensor [B, 1, T]
            enroll: list[Tensor]
                each Tensor: [B, 1, T]
        """
        if isinstance(enroll, (list, tuple)):
            enroll = enroll[0]
        mix = mix.squeeze(1)
        enroll = enroll.squeeze(1)

        # input shape: (B, T)
        mix_dims = mix.dim()
        assert mix_dims == 2, "Only support 2D Input"

        ##### Cue of the target speaker
        wav_enroll = enroll
        ###### Extraction with speaker cue
        batch_size, nsamples = mix.shape
        wav_mix = mix

        # 🌟 新增：初始化声纹 logits
        spk_logits = None

        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            # C0.1 Prepend the enroll to the mix in the beginning
            wav_mix = self.spk_ft.listen.compute(wav_enroll,
                                                 wav_mix)  # (B, T_e + T_s + T)
        ###########################################################
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(wav_mix)[-1]
        # S2. Concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # (B, 2, F, T)
        ###########################################################
        # C1. Feature: usef
        if self.spk_configs['features']['usef']['enabled']:
            # C1.1 Generate the USEF feature
            enroll_spec = self.sep_model.stft(wav_enroll)[
                -1]  # (B, F, T_e) complex
            enroll_spec = torch.stack([enroll_spec.real, enroll_spec.imag],
                                      1)  # (B, 2, F, T)

            # 这里的 enroll_usef 就是目标说话人在当前时频单元下的特征映射
            enroll_usef, mix_usef = self.spk_ft.usef.compute(
                enroll_spec, spec_RI)  # (B, embed_dim, F, T)

            # 🌟 修改：仅对时间轴做 Pooling，保留并展平频率轴
            if self.spk_configs.get("sv_head", {}).get("enabled", False):
                # 1. 仅在时间轴 (dim=-1) 上求平均，保留频域分布，变为 (B, emb_dim, F)
                e_time_pooled = enroll_usef.mean(dim=-1)

                # 2. 展平为 (B, emb_dim * F) 喂给线性层
                e_global = e_time_pooled.view(batch_size, -1)
                spk_logits = self.sv_head(e_global)

            # C1.2 Concate the USEF feature to the mix_repr's spec
            spec_RI = self.spk_ft.usef.post(
                mix_usef, enroll_usef)  # (B, embed_dim*2, F, T)

        # C2. Feature: tfmap
        if self.spk_configs['features']['tfmap']['enabled']:
            # C2.1 Generate the TF-Map feature
            enroll_mag = self.sep_model.stft(wav_enroll)[0]  # (B, F, T_e)
            enroll_tfmap = self.spk_ft.tfmap.compute(
                enroll_mag, torch.abs(spec))  # (B, F, T)
            # C2.2 Concate the TF-Map feature to the mix_repr's spec
            spec_RI = self.spk_ft.tfmap.post(
                spec_RI, enroll_tfmap.unsqueeze(1))  # (B, 3, F, T)
        ###########################################################
        subband_spec = self.sep_model.band_split(
            spec_RI)  # list of (B, 2/3/2*usef.emb_dim, BW, T)
        subband_mix_spec = self.sep_model.band_split(
            spec)  # list of (B, BW, T) complex
        # S3. Normalization and bottleneck
        subband_feature = self.sep_model.subband_norm(
            subband_spec)  # (B, nband, feat, T)
        ###########################################################
        # C3. Feature: context
        if self.spk_configs['features']['context']['enabled']:
            # C3.1 Generate the frame-level speaker embeddings
            enroll_context = self.spk_ft.context.compute(
                wav_enroll)  # (B, F_e, T_e)
            # C3.2 Fuse the frame-level speaker embeddings into the mix_repr
            subband_feature = self.spk_ft.context.post(
                subband_feature, enroll_context)  # (B, nband, feat, T)
        # C4. Feature: spkemb
        if self.spk_configs['features']['spkemb']['enabled']:
            # C4.1 Generate the speaker embedding
            enroll_emb = self.spk_ft.spkemb.compute(wav_enroll)  # (B, F_e)
            # C4.2 Fuse the speaker embeeding into the mix_repr
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)  # (B, 1, F_e, 1)
            subband_feature = self.spk_ft.spkemb.post(
                subband_feature, enroll_emb)  # (B, nband, feat, T)
        ###########################################################
        # S4. Separation
        sep_output = self.sep_model.separator(
            subband_feature)  # (B, nband, feat, T)
        # S5. Complex Mask
        est_spec_RI = self.sep_model.band_masker(
            sep_output, subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0],
                                    est_spec_RI[:, 1])  # (B, S, F, T)
        # S6. Back into waveform
        s = self.sep_model.istft(est_complex)  # (B, S, T)
        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            # C0.2 Prepend the enroll to the mix in the beginning
            s = self.spk_ft.listen.post(s)  # (B, T)
        ###########################################################

        # 🌟 新增：如果启用了 SV Head，将分离出的音频和声纹 logits 一起返回
        if self.spk_configs.get("sv_head", {}).get("enabled", False):
            return s, spk_logits

        return s
