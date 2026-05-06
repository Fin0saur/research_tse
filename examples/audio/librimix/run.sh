#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)
#           2026 Ke Zhang (kylezhang1118@gmail.com)
. ./path.sh || exit 1

# General configuration
stage=-1
stop_stage=-1

# Data preparation related
data=data
fs=16k
min_max=min
noise_type="clean"
data_type="raw" # shard/raw
Libri2Mix_dir=/data0/import/asr/Libri2Mix
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Training related
gpus="[0]"
config=confs/tse_bsrnn_spk.yaml
exp_dir=/mnt/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_cor
if [ -z "${config}" ] && [ -f "${exp_dir}/config.yaml" ]; then
  config="${exp_dir}/config.yaml"
fi

# TSE model initialization related
checkpoint=

# Inferencing and scoring related
save_results=False
use_pesq=true
use_dnsmos=true
dnsmos_use_gpu=true

# Model average related
num_avg=5

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 4
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${datatype}" = "shard" ]; then
  echo "Making shards from samples.jsonl ..."
  for dset in train-100 dev test; do
  #  for dset in train-360; do
    python tools/make_shards_from_samples.py \
      --samples ${data}/${dset}/samples.jsonl \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/${dset}/shards \
      ${data}/${dset}/shard.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  train_script=wesep/bin/train.py
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/train-100/${data_type}.list \
    --train_cues ${data}/train-100/cues.yaml \
    --train_samples ${data}/train-100/samples.jsonl \
    --val_data ${data}/dev/${data_type}.list \
    --val_cues ${data}/dev/cues.yaml \
    --val_samples ${data}/dev/samples.jsonl \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_best_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    --mode best \
    --epochs "81,82,83,84,85"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  ./tools/score.sh --dset "${data}/test" \
    --exp_dir "${exp_dir}" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp1.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp2.py --config $config \
    --fs ${fs} \
    --gpus 2 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp3.py --config $config \
    --fs ${fs} \
    --gpus 2 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp4.py --config $config \
    --fs ${fs} \
    --gpus 3 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp5.py --config $config \
    --fs ${fs} \
    --gpus 3 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp6.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  python wesep/bin/analyse_gap.py
fi
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_single.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --utt_id "4970-29093-0018_1221-135766-0002" \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp7.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp8.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp9.py --config $config \
    --fs ${fs} \
    --gpus 7 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp10.py --config $config \
    --fs ${fs} \
    --gpus 1 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp11.py --config $config \
    --fs ${fs} \
    --gpus 7 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp12.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp13.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi


if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer_exp14.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
  echo "🚀 Start inferencing in parallel (8 GPUs) ..."

  # NUM_SHARDS=4

  # # GPU 分配：2 个进程跑 GPU 0，2 个进程跑 GPU 6
  # declare -a GPU_ARRAY=(1 1 7 7)

  # # 1. 开启 4 个后台进程并行提取特征和音频，分布在 GPU 0 和 GPU 6 上
  # for i in {0..3}; do
  #   gpu=${GPU_ARRAY[$i]}
  #   echo "Submitting Shard $i on GPU $gpu..."
  #   python wesep/bin/infer_exp15.py infer \
  #     --config $config \
  #     --fs ${fs} \
  #     --gpus $gpu \
  #     --num_shards $NUM_SHARDS \
  #     --shard_id $i \
  #     --exp_dir ${exp_dir} \
  #     --data_type "${data_type}" \
  #     --test_data ${data}/test/${data_type}.list \
  #     --test_cues ${data}/test/cues.yaml \
  #     --test_samples ${data}/test/samples.jsonl \
  #     --save_wav ${save_results} \
  #     --checkpoint "/data2/yxy05/exp/TSE_BSRNN_SPK_EMB_SV/models/checkpoint_90.pt" > ${exp_dir}/infer_shard_${i}.log 2>&1 &
  # done

  # # 2. wait 极其关键！它会锁住当前终端，直到上面 8 个挂起(&)的任务全部执行完毕
  # echo "⏳ All 8 shards submitted. Waiting for feature extraction to finish..."
  # wait
  # echo "✅ Feature extraction on 8 GPUs completed!"

  # 3. 提取全部完成后，单卡调用绘图引擎自动筛选并画图
  echo "🎨 Start automated cherry-picking and global t-SNE plotting ..."
  # 注意这里调用的是 'plot' 指令
  python wesep/bin/infer_exp15.py plot \
      --config $config \
      --exp_dir ${exp_dir}

  echo "🎉 Stage 23 All Done!"
fi

save_results="True"

# if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
#   echo "🚀 Start inferencing in parallel (4 processes on GPUs 0,6) ..."

#   # NUM_SHARDS=4

#   # # GPU 分配：2 个进程跑 GPU 0，2 个进程跑 GPU 6
#   # declare -a GPU_ARRAY=(0 0 6 6)

#   # # 1. 开启 4 个后台进程并行提取特征和音频，分布在 GPU 0 和 GPU 6 上
#   # for i in {0..3}; do
#   #   gpu=${GPU_ARRAY[$i]}
#   #   echo "Submitting Shard $i on GPU $gpu..."
#   #   python wesep/bin/infer_exp16.py infer \
#   #     --config $config \
#   #     --fs ${fs} \
#   #     --gpus $gpu \
#   #     --num_shards $NUM_SHARDS \
#   #     --shard_id $i \
#   #     --exp_dir ${exp_dir} \
#   #     --data_type "${data_type}" \
#   #     --test_data ${data}/test/${data_type}.list \
#   #     --test_cues ${data}/test/cues.yaml \
#   #     --test_samples ${data}/test/samples.jsonl \
#   #     --save_wav ${save_results} \
#   #     --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt" > ${exp_dir}/infer_shard_${i}.log 2>&1 &
#   # done

#   # # 2. wait 极其关键！它会锁住当前终端，直到上面 4 个挂起(&)的任务全部执行完毕
#   # echo "⏳ All 4 shards submitted. Waiting for feature extraction to finish..."
#   # wait
#   # echo "✅ Feature extraction on GPU 0 completed!"

#   # # 3. 提取全部完成后，调用绘图引擎自动筛选并画图 (生成 10 张高亮神图)
#   # echo "🎨 Start automated cherry-picking and global t-SNE plotting ..."
#   # python wesep/bin/infer_exp16.py plot \
#   #     --config $config \
#   #     --exp_dir ${exp_dir}

#   #   # 4. 🌟 调用象限验证引擎，解耦【混淆】与【畸变】
#   # echo "🔬 Start hypothesis verification (S_tgt vs S_int Quadrant Plot) ..."
#   # python wesep/bin/infer_exp16.py verify \
#   #     --config $config \
#   #     --exp_dir ${exp_dir}

#   # 5. 🎨 调用两类样本 t-SNE 绘图引擎
#   echo "🎨 Start two-category t-SNE plotting (Cat1: Interferer Extracted, Cat2: Audio Distortion) ..."
#   python wesep/bin/infer_exp16.py plot_two_category \
#       --config $config \
#       --exp_dir ${exp_dir}

#   echo "🎉 Stage 24 All Done! 请去 ${exp_dir} 查看图表和音频！"
# fi

# ========================================================
# Stage 24: 生成偏差注册数据集 (1扩20)
# 用于探针训练时的数据增强
# ========================================================
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
  echo "🚀 生成偏差注册数据集 (1扩20) ..."

  # train-100 集: 3,000 混合 × 20 cue = 60,000 样本
  python wesep/bin/generate_bias_list.py \
    --input ${data}/train-100/raw.list \
    --audio_dict ${data}/train-100/cues/audio.json \
    --output ${data}/enroll_bias/train-100/bias_enroll20_seed42.jsonl

  # dev 集: 3,000 混合 × 20 cue = 60,000 样本
  python wesep/bin/generate_bias_list.py \
    --input ${data}/dev/raw.list \
    --audio_dict ${data}/dev/cues/audio.json \
    --output ${data}/enroll_bias/dev/bias_enroll20_seed42.jsonl

  # test 集: 同样处理
  python wesep/bin/generate_bias_list.py \
    --input ${data}/test/raw.list \
    --audio_dict ${data}/test/cues/audio.json \
    --output ${data}/enroll_bias/test/bias_enroll20_seed42.jsonl

  echo "🎉 Stage 24 Done! 偏差数据集已生成至 ${data}/enroll_bias/"
fi

# ========================================================
# Stage 25: 收集训练/验证/测试集上的推理中间特征 (单卡运行，拆分到不同 GPU)
# ========================================================
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ]; then
  echo "🚀 Start collecting inference features (prior, pmap, post_concat) on train/val/test ..."

  CHECKPOINT="/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt"

  # GPU 0: test
  echo "   Collecting test features on GPU 0 ..."
  python wesep/bin/collect_inference_features.py \
    --config $config \
    --gpus 0 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --dataset_split=test \
    --num_shards 1 \
    --shard_id 0 > ${exp_dir}/collect_test.log 2>&1 &
  PID_TEST=$!

  # GPU 1: val
  echo "   Collecting val features on GPU 1 ..."
  python wesep/bin/collect_inference_features.py \
    --config $config \
    --gpus 1 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --val_data ${data}/dev/${data_type}.list \
    --val_cues ${data}/dev/cues.yaml \
    --val_samples ${data}/dev/samples.jsonl \
    --dataset_split=val \
    --num_shards 1 \
    --shard_id 0 > ${exp_dir}/collect_val.log 2>&1 &
  PID_VAL=$!

  # GPU 2: train
  echo "   Collecting train features on GPU 2 ..."
  python wesep/bin/collect_inference_features.py \
    --config $config \
    --gpus 2 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --train_data ${data}/train-100/${data_type}.list \
    --train_cues ${data}/train-100/cues.yaml \
    --train_samples ${data}/train-100/samples.jsonl \
    --dataset_split=train \
    --num_shards 1 \
    --shard_id 0 > ${exp_dir}/collect_train.log 2>&1 &
  PID_TRAIN=$!

  echo "   Waiting for all tasks to finish ..."
  wait $PID_TEST
  echo "   ✅ Test features collected!"
  wait $PID_VAL
  echo "   ✅ Val features collected!"
  wait $PID_TRAIN
  echo "   ✅ Train features collected!"

  echo "🎉 Stage 25 Done! 特征已保存至 ${exp_dir}/{train,val,test}_features/"
fi

# ========================================================
# Stage 26: 统计数据集质量分布 (C0/C1)
# ========================================================
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ]; then
  echo "🚀 统计数据集质量分布 ..."
  python wesep/bin/stat_quality.py --exp_dir ${exp_dir}
fi

# ========================================================
# Stage 27: 探针训练 (Direct Probe)
# ========================================================
if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ]; then
  echo "🚀 探针训练: prior ..."

  for feat in prior pmap post; do
    echo "   Training probe on ${feat} features ..."
    python wesep/bin/train_probe_v2.py \
      --exp_dir ${exp_dir} \
      --feat_type ${feat} \
      --hidden_dim 256 \
      --dropout 0.3 \
      --batch_size 128 \
      --lr 1e-3 \
      --epochs 30 \
      --device cuda:0 \
      --seed 42
  done

  echo "🎉 Stage 27 Done! 探针结果已保存至 ${exp_dir}/probe_results/"
fi

# ========================================================
# Stage 28: 偏差数据集预打分 (Pre-scoring for Joint Training)
# 用冻结 BSRNN 计算 SI-SNRi 并打标签，生成 scored.jsonl
# ========================================================
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ]; then
  echo "🚀 偏差数据集预打分 ..."

  CHECKPOINT="/mnt/data0/ckpt/avg_best_model.pt"

  # ===== Phase 1: train-100 分 4 张 GPU =====
  echo "   [Phase 1] 正在打分 train-100 偏差数据集 (4 GPU 并行) ..."

  # GPU 0: train-100 shard 0
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 0 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/train-100/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/train-100/bias_enroll20_scored.jsonl \
    --num_shards 4 \
    --shard_id 0 > ${exp_dir}/score_train_s0.log 2>&1 &
  PID_S0=$!

  # GPU 1: train-100 shard 1
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 1 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/train-100/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/train-100/bias_enroll20_scored.jsonl \
    --num_shards 4 \
    --shard_id 1 > ${exp_dir}/score_train_s1.log 2>&1 &
  PID_S1=$!

  # GPU 2: train-100 shard 2
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 2 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/train-100/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/train-100/bias_enroll20_scored.jsonl \
    --num_shards 4 \
    --shard_id 2 > ${exp_dir}/score_train_s2.log 2>&1 &
  PID_S2=$!

  # GPU 3: train-100 shard 3
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 3 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/train-100/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/train-100/bias_enroll20_scored.jsonl \
    --num_shards 4 \
    --shard_id 3 > ${exp_dir}/score_train_s3.log 2>&1 &
  PID_S3=$!

  # echo "   等待 train-100 4 个 shard 完成 ..."
  # wait $PID_S0
  # echo "   ✅ Train shard 0 完成"
  # wait $PID_S1
  # echo "   ✅ Train shard 1 完成"
  # wait $PID_S2
  # echo "   ✅ Train shard 2 完成"
  # wait $PID_S3
  # echo "   ✅ Train shard 3 完成"

  # # ===== Phase 2: dev 和 test 各分 2 shard，4 GPU 同时跑 =====
  # echo "   [Phase 2] 正在打分 dev 和 test 偏差数据集 (4 GPU 并行) ..."

  # GPU 0: dev shard 0
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 0 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/dev/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/dev/bias_enroll20_scored.jsonl \
    --num_shards 2 \
    --shard_id 0 > ${exp_dir}/score_dev_s0.log 2>&1 &
  PID_DEV_S0=$!

  # GPU 1: dev shard 1
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 1 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/dev/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/dev/bias_enroll20_scored.jsonl \
    --num_shards 2 \
    --shard_id 1 > ${exp_dir}/score_dev_s1.log 2>&1 &
  PID_DEV_S1=$!

  # GPU 2: test shard 0
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 2 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/test/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/test/bias_enroll20_scored.jsonl \
    --num_shards 2 \
    --shard_id 0 > ${exp_dir}/score_test_s0.log 2>&1 &
  PID_TEST_S0=$!

  # GPU 3: test shard 1
  python wesep/bin/score_bias_dataset.py \
    --config $config \
    --gpus 3 \
    --checkpoint "${CHECKPOINT}" \
    --exp_dir ${exp_dir} \
    --input_jsonl ${data}/enroll_bias/test/bias_enroll20_seed42.jsonl \
    --output_jsonl ${data}/enroll_bias/test/bias_enroll20_scored.jsonl \
    --num_shards 2 \
    --shard_id 1 > ${exp_dir}/score_test_s1.log 2>&1 &
  PID_TEST_S1=$!

  wait $PID_DEV_S0
  echo "   ✅ Dev shard 0 完成"
  wait $PID_DEV_S1
  echo "   ✅ Dev shard 1 完成"
  wait $PID_TEST_S0
  echo "   ✅ Test shard 0 完成"
  wait $PID_TEST_S1
  echo "   ✅ Test shard 1 完成"

  echo "🎉 Stage 28 Done! 打分结果已保存至 ${data}/enroll_bias/{train-100,dev,test}/bias_enroll20_scored.jsonl"
fi

# ========================================================
# Stage 29: 独立探针训练 (Independent Probe Training)
# 冻结主干网络，分别训练 prior/pmap/post_concat 三个探针
# 比较哪个特征最适合预测分离效果
# ========================================================
if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ]; then
  echo "🚀 独立探针训练: prior / pmap / post ..."

  # 预训练权重路径
  BSRNN_CKPT="/mnt/data0/ckpt/avg_best_model.pt"
  ECAPA_PRETRAINED="./wespeaker_models/voxceleb_ECAPA512/avg_model.pt"

  # 输出目录
  PROBE_EXP_DIR="${exp_dir}/probe_single"

  # GPU数量
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  # 分别训练三个探针
  for probe_type in pmap post; do
    echo "=========================================="
    echo "Training ${probe_type} probe ..."
    echo "=========================================="

    mkdir -p ${PROBE_EXP_DIR}/probe_${probe_type}/models

    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
      wesep/bin/train_joint_probe.py \
      --config $config \
      --exp_dir ${PROBE_EXP_DIR} \
      --scored_data_dir ${data}/enroll_bias/train-100 \
      --eval_data_dir ${data}/enroll_bias/dev \
      --checkpoint "${BSRNN_CKPT}" \
      --ecapa_pretrained "${ECAPA_PRETRAINED}" \
      --probe_type ${probe_type} \
      --num_epochs 50 \
      --lr 1e-4 \
      --batch_size 4 \
      --sample_num_per_epoch 10000 \
      --eval_max_batches 1000 \
      --log_interval 10

    echo "✅ ${probe_type} probe training completed!"
  done

  echo "🎉 Stage 29 Done! 三个探针已分别训练完成，保存至 ${PROBE_EXP_DIR}/probe_{prior,pmap,post}/"
fi