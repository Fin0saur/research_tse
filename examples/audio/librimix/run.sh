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
gpus="[1,7]"
config=confs/tse_bsrnn_spk.yaml
exp_dir=/data2/yxy05/exp/TSE_BSRNN_SPK_EMB_cor
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

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
  echo "🚀 Start inferencing in parallel (4 processes on GPUs 0,6) ..."

  # NUM_SHARDS=4

  # # GPU 分配：2 个进程跑 GPU 0，2 个进程跑 GPU 6
  # declare -a GPU_ARRAY=(0 0 6 6)

  # # 1. 开启 4 个后台进程并行提取特征和音频，分布在 GPU 0 和 GPU 6 上
  # for i in {0..3}; do
  #   gpu=${GPU_ARRAY[$i]}
  #   echo "Submitting Shard $i on GPU $gpu..."
  #   python wesep/bin/infer_exp16.py infer \
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
  #     --checkpoint "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB/models/avg_best_model.pt" > ${exp_dir}/infer_shard_${i}.log 2>&1 &
  # done

  # # 2. wait 极其关键！它会锁住当前终端，直到上面 4 个挂起(&)的任务全部执行完毕
  # echo "⏳ All 4 shards submitted. Waiting for feature extraction to finish..."
  # wait
  # echo "✅ Feature extraction on GPU 0 completed!"

  # # 3. 提取全部完成后，调用绘图引擎自动筛选并画图 (生成 10 张高亮神图)
  # echo "🎨 Start automated cherry-picking and global t-SNE plotting ..."
  # python wesep/bin/infer_exp16.py plot \
  #     --config $config \
  #     --exp_dir ${exp_dir}

  #   # 4. 🌟 调用象限验证引擎，解耦【混淆】与【畸变】
  # echo "🔬 Start hypothesis verification (S_tgt vs S_int Quadrant Plot) ..."
  # python wesep/bin/infer_exp16.py verify \
  #     --config $config \
  #     --exp_dir ${exp_dir}

  # 5. 🎨 调用两类样本 t-SNE 绘图引擎
  echo "🎨 Start two-category t-SNE plotting (Cat1: Interferer Extracted, Cat2: Audio Distortion) ..."
  python wesep/bin/infer_exp16.py plot_two_category \
      --config $config \
      --exp_dir ${exp_dir}

  echo "🎉 Stage 24 All Done! 请去 ${exp_dir} 查看图表和音频！"
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