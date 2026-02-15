#!/bin/bash
# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#

stage=-1
stop_stage=-1

mix_data_path='./Libri2Mix/wav16k/min/'

data=data
noise_type=clean
num_spk=2

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
  echo "stage 2: Build fixed spatial cues from samples.jsonl"

  for dset in train-100 dev test; do
  # for dset in train-360; do
    mix_index="${real_data}/${noise_type}/${dset}/samples.jsonl"
    out_dir="${real_data}/${noise_type}/${dset}/cues"
    mkdir -p "${out_dir}"

    # 1) Generate cues/spatial.json
    spatial_root=${mix_data_path}/${dset}/spatial
    python local/build_spatial_cues.py \
      --samples_jsonl "${mix_index}" \
      --spatial_root "${spatial_root}" \
      --outfile "${out_dir}/spatial.json"

    # 2) Generate cues.yaml  add spatial_fields  
cat > ${data}/${noise_type}/${dset}/cues.yaml << EOF
cues:
  spatial:
    type: npy
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/${dset}/cues/spatial.json
    spatial_fields: ["azimuth","elevation"]  
EOF
  done
fi


# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#   if [ ! -d "${real_data}/raw_data/musan" ]; then
#     mkdir -p ${real_data}/raw_data/musan
#     #
#     echo "Downloading musan.tar.gz ..."
#     echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
#     wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${real_data}/raw_data
#     md5=$(md5sum ${real_data}/raw_data/musan.tar.gz | awk '{print $1}')
#     [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

#     echo "Decompress all archives ..."
#     tar -xzvf ${real_data}/raw_data/musan.tar.gz -C ${real_data}/raw_data

#     rm -rf ${real_data}/raw_data/musan.tar.gz
#   fi

#   echo "Prepare wav.scp for musan ..."
#   mkdir -p ${real_data}/musan
#   find -L ${real_data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${real_data}/musan/wav.scp

#   # Convert all musan data to LMDB
#   echo "conver musan data to LMDB ..."
#   python tools/make_lmdb.py ${real_data}/musan/wav.scp ${real_data}/musan/lmdb
# fi