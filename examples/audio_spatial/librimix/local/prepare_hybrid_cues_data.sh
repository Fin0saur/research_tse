#!/bin/bash
# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
#               2026 Ke Zhang (kylezhang1118@gmail.com)

stage=-1
stop_stage=-1

mix_data_path='./Libri2Mix/wav16k/min/'

data=data
noise_type=clean
num_spk=2

# 设置保留真实线索的比例 (例如 0.6 表示保留 60%，丢失 40%)
KEEP_RATIO=0.7

. tools/parse_options.sh || exit 1

real_data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the datasets (JSONL)"

  for dataset in dev test train-100; do
    echo "Preparing JSONL for" $dataset

    dataset_path=$mix_data_path/$dataset/mix_${noise_type}
    out_dir="${real_data}/${noise_type}/${dataset}"
    mkdir -p "${out_dir}"

    python local/scan_librimix.py \
      "${dataset_path}" \
      --outfile "${out_dir}/samples.jsonl"

    ln -sf samples.jsonl "${out_dir}/raw.list"
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Build random audio cues for training set from samples.jsonl"

  for dset in train-100; do
  # for dset in train-360; do
    mix_index="${real_data}/${noise_type}/${dset}/samples.jsonl"
    out_dir="${real_data}/${noise_type}/${dset}/cues"
    mkdir -p "${out_dir}"

    # 1) Generate full audio cues
    python local/build_audio_cues.py \
      --samples_jsonl "${mix_index}" \
      --outfile "${out_dir}/audio_full.json"
    
    # 随机剔除部分 Audio 线索
    python local/drop_cues.py \
      --in_json "${out_dir}/audio_full.json" \
      --out_json "${out_dir}/audio.json" \
      --keep_ratio ${KEEP_RATIO} \
      --seed 101
  
    # Generate full spatial cues
    spatial_root=${mix_data_path}/${dset}/spatial
    python local/build_spatial_cues.py \
      --samples_jsonl "${mix_index}" \
      --spatial_root "${spatial_root}" \
      --outfile "${out_dir}/spatial_full.json"
      
    # 随机剔除部分 Spatial 线索 (使用不同的 seed 保证两者随机独立)
    python local/drop_cues.py \
      --in_json "${out_dir}/spatial_full.json" \
      --out_json "${out_dir}/spatial.json" \
      --keep_ratio ${KEEP_RATIO} \
      --seed 202

    # 2) Generate cues.yaml (注意: guaranteed 设置为 false)
cat > ${real_data}/${noise_type}/${dset}/cues.yaml << EOF
cues:
  audio:
    type: raw
    guaranteed: false
    scope: speaker
    policy:
      type: random
      key: spk_id
      resource: ${real_data}/${noise_type}/${dset}/cues/audio.json
  spatial:
    type: npy
    guaranteed: false
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${real_data}/${noise_type}/${dset}/cues/spatial.json
    spatial_fields: ["azimuth","elevation"]  
EOF
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Build fixed audio cues for eval and test sets from samples.jsonl"

  for dset in dev test; do
    mix_index="${real_data}/${noise_type}/${dset}/samples.jsonl"
    out_dir="${real_data}/${noise_type}/${dset}/cues"
    spatial_root=${mix_data_path}/${dset}/spatial
    mkdir -p "${out_dir}"

    # A) Generate speech.json(sanity check)
    python local/build_audio_cues.py \
      --samples_jsonl "${mix_index}" \
      --outfile "${out_dir}/audio.json"

    python local/generate_mix2enroll.py \
      --samples_jsonl "${mix_index}" \
      --speech_json "${out_dir}/audio.json" \
      --outfile "${out_dir}/mixture2enrollment" \
      --seed 42

    # C) Generate fixed_enroll_full.json
    python local/build_fixed_enroll.py \
      --mixture2enrollment "${out_dir}/mixture2enrollment" \
      --speech_json "${out_dir}/audio.json" \
      --outfile "${out_dir}/fixed_enroll_full.json"
      
    # 随机剔除部分测试集 Audio 线索
    python local/drop_cues.py \
      --in_json "${out_dir}/fixed_enroll_full.json" \
      --out_json "${out_dir}/fixed_enroll.json" \
      --keep_ratio ${KEEP_RATIO} \
      --seed 303
    
    # Generate spatial_full.json
    python local/build_spatial_cues.py \
      --samples_jsonl "${mix_index}" \
      --spatial_root "${spatial_root}" \
      --outfile "${out_dir}/spatial_full.json"
      
    # 随机剔除部分测试集 Spatial 线索
    python local/drop_cues.py \
      --in_json "${out_dir}/spatial_full.json" \
      --out_json "${out_dir}/spatial.json" \
      --keep_ratio ${KEEP_RATIO} \
      --seed 404

  # D) Generate cues.yaml for dev/test (注意: guaranteed 设置为 false)
  cat > ${real_data}/${noise_type}/${dset}/cues.yaml << EOF
cues:
  audio:
    type: raw
    guaranteed: false
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${real_data}/${noise_type}/${dset}/cues/fixed_enroll.json
  spatial:
    type: npy
    guaranteed: false
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${real_data}/${noise_type}/${dset}/cues/spatial.json
    spatial_fields: ["azimuth","elevation"]  
EOF
  done
fi

# ============================================================================
# Stage 4, 5, 6 保持你原有的逻辑不变
# ============================================================================

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Download the pre-trained speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker..."
  mkdir -p wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip
  unzip -o voxceleb_resnet34.zip -d wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip
  unzip -o voxceleb_ECAPA512.zip -d wespeaker_models
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ ! -d "${real_data}/raw_data/musan" ]; then
    mkdir -p ${real_data}/raw_data/musan
    echo "Downloading musan.tar.gz ..."
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${real_data}/raw_data
    md5=$(md5sum ${real_data}/raw_data/musan.tar.gz | awk '{print $1}')
    [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

    echo "Decompress all archives ..."
    tar -xzvf ${real_data}/raw_data/musan.tar.gz -C ${real_data}/raw_data
    rm -rf ${real_data}/raw_data/musan.tar.gz
  fi

  echo "Prepare wav.scp for musan ..."
  mkdir -p ${real_data}/musan
  find -L ${real_data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${real_data}/musan/wav.scp

  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${real_data}/musan/wav.scp ${real_data}/musan/lmdb
fi