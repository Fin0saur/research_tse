# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np

from wesep.utils.file_utils import load_json


def _build_lookup_key(sample, spk_slot, key_field):
    """
    Build lookup key for speaker cue resource.

    key_field semantics:
      - "spk_id"       -> use sample[spk_slot]
      - "mix_spk_id"   -> use f"{sample['key']}::{sample[spk_slot]}"
    """
    if key_field == "spk_id":
        return sample[spk_slot]

    elif key_field == "mix_spk_id":
        mix_key = sample.get("key", None)
        if mix_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mix_key}::{sample[spk_slot]}"

    else:
        raise ValueError(f"Unsupported key_field for speaker cue: {key_field}")


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def _get_spk_resource(resource_path):
    """
    Lazy-load and cache speaker cue resources.

    Cache is keyed by resource_path to avoid train/val or
    multi-dataset cross-contamination.
    """
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    return _SPK_RESOURCE_CACHE[resource_path]


def sample_fixed_spatial_cue(
    data,
    resource_path,
    spatial_fields,
    key_field,
    scope="speaker",
    required=True,
):
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    spk_resource = _get_spk_resource(resource_path)

    for sample in data:
        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        if scope == "utterance":
            spk_slots = [spk_slots[0]]

        for slot in spk_slots:
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(
                        f"fixed speaker cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(
                        f"empty fixed speaker cue: {lookup_key}")
                continue
            target_spk = items["target_spk"]
            sources_info = items["sources"]
            
            target_source_info = None
            for src in sources_info:
                if src["spk"] == target_spk:
                    target_source_info = src
                    break
            
            if target_source_info is None:
                if required:
                    raise RuntimeError(f"Target speaker {target_spk} not found in sources list for {lookup_key}")
                continue
            
            feats = []
            for field in spatial_fields:
                if field == "azimuth":
                    val = target_source_info.get("azimuth")
                    if val is None:
                        if required: raise ValueError(f"Azimuth missing in {lookup_key}")
                        val = 0.0
                    feats.append(val)
                
                elif field == "elevation":
                    val = target_source_info.get("elevation", 0.0) 
                    feats.append(val)
                    
                # elif field == "distance":
                #     val = target_source_info.get("distance", 1.0) # 默认1米?
                #     feats.append(val)
            
            
            spatial_vec = np.array(feats, dtype=np.float32) 
            sample[f"spatial_{slot}"] = spatial_vec

        if scope == "utterance":
            emb = sample[f"spatial_{spk_slots[0]}"]
            for slot in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"spatial_{slot}"] = emb

        yield sample
