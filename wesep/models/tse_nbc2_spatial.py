import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update

class TSE_NBC2_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. Basic Configs ---
        self.win = config.get("win",512)
        self.stride = config.get("stride",256)
        
        self.window = torch.hann_window(self.win)
        
        freq_bins = self.win // 2 + 1
        
        # --- 2. Spatial Configs ---
        spatial_configs = {
            "geometry": {
                "n_fft": self.win,              
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.03333333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [
                [0, 1], [1, 2], [2, 3], [0, 3]
            ],
            "features": {
                "ipd": {"enabled": True},
                "cdf": {"enabled": True},
                "sdf": {"enabled": True},
                "delta_stft": {"enabled": True},
                "cyc_doaemb":{
                    "enabled": True,
                    "cyc_alpha": 20,
                    "cyc_dimension": 40,
                    "use_ele": True,
                    "out_channel": 1, # only use when concat
                    "fusion_type": "multiply" # concat or multiply
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        
        # --- 3. Dynamic Input Size Calculation ---
        spec_feat_dim = 2 
        
        n_pairs = len(self.spatial_configs['pairs'])
        feat_cfg = self.spatial_configs['features']
        spatial_dim=0
        if feat_cfg.get('ipd', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False): spatial_dim += 2*n_pairs
        if feat_cfg.get('cyc_doaemb',{}).get('enabled',False): 
            if feat_cfg.get('cyc_doaemb',{}).get('fusion_type') == "concat":
                spatial_dim += feat_cfg['cyc_doaemb']['out_channel']
            elif feat_cfg.get('cyc_doaemb',{}).get('fusion_type') == "multiply":
                feat_cfg['cyc_doaemb']['out_channel'] = 96 # dim_hidden    
        total_input_size = spec_feat_dim + spatial_dim

        # --- 4. Backbone Configs ---
        block_kwargs = {
            'n_heads': 2,
            'dropout': 0.1,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': freq_bins,
                'share_along_sequence_dim': False,
            },
        }
        
        sep_configs = dict(
            input_size=total_input_size,
            output_size=2,
            n_layers=8,
            dim_hidden=96,
            dim_ffn=96*2,
            block_kwargs=block_kwargs
        )
        self.sep_configs = deep_update(sep_configs, config.get('separator', {}))
        
        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
    def forward(self, mix,cue):
        # input shape: (B, C, T)
        spatial_cue=cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]        
        
        B, M, T_wav = mix.shape
        self.window = self.window.to(mix.device)
        mix_reshape = mix.view(B * M, T_wav)
        
        spec = torch.stft(
            mix_reshape,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            return_complex=True
        )
        
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim)
        
        # --- 2. A-Norm ---
        Y_ref = Y[:, 0]
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)
        
        # Spectral: (B, 2, F, T)
        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1)
        
        # Spatial: (B, 16, F, T)
        spatial_feat_dict = self.spatial_ft.compute_all(Y_norm,azi_rad, ele_rad)

        # features = self.spatial_ft.post_all(spec_feat, spatial_feat_dict) 
        features = spec_feat
        
        if self.spatial_configs['features']['ipd']['enabled'] :
            features=self.spatial_ft.features['ipd'].post(features,spatial_feat_dict['ipd'])
        
        if self.spatial_configs['features']['cdf']['enabled'] :
            features=self.spatial_ft.features['cdf'].post(features,spatial_feat_dict['cdf'])
        
        if self.spatial_configs['features']['sdf']['enabled']:
            features=self.spatial_ft.features['sdf'].post(features,spatial_feat_dict['sdf'])
        
        if self.spatial_configs['features']['delta_stft']['enabled']:
            features=self.spatial_ft.features['delta_stft'].post(features,spatial_feat_dict['delta_stft'])
        
        if self.spatial_configs['features']['cyc_doaemb']['enabled'] and self.spatial_configs['features']['cyc_doaemb']['fusion_type'] == 'concat':
            features=self.spatial_ft.features['cyc_doaemb'].post(features,spatial_feat_dict['cyc_doaemb'])
            
        # --- Backbone ---
        encode_features = self.sep_model.encoder(features)
        for m in self.sep_model.sa_layers:
            if self.spatial_configs['features']['cyc_doaemb']['enabled'] and self.spatial_configs['features']['cyc_doaemb']['fusion_type'] == 'multiply':
                encode_features=self.spatial_ft.features['cyc_doaemb'].post(encode_features,spatial_feat_dict['cyc_doaemb'])
            encode_features , _ = m(encode_features)
        
        est_spec_feat = self.sep_model.decoder(encode_features)
        
        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        
        # Inverse Normalization
        est_spec = est_spec * ref_mag_mean
        
        # --- iSTFT ---
        est_wav = torch.istft(
            est_spec,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            length=T_wav
        )
        
        est_wav=est_wav.unsqueeze(1) # [B 1 T]
        
        return est_wav
    