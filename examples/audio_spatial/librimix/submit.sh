source /zhangke/anaconda3/etc/profile.d/conda.sh
conda activate wesep

cd /zhangke/zk/wesep_update/wesep_wenet/examples/spatial/librimix/

./run.sh --stage 3 --stop-stage 3 --gpus "[0]" --config confs/tse_bsrnn_spatial.yaml --exp_dir exp/tse_bsrnn_spatial/test/ #--checkpoint exp/tse_bsrnn_spk/tfmap_context_causal/models/checkpoint_142.pt
# ./run.sh --stage 5 --stop-stage 5 --gpus "[0]" --config confs/tse_bsrnn_spk.yaml --exp_dir exp/tse_bsrnn_spk/tfmap_context_causal/ --checkpoint exp/tse_bsrnn_spk/tfmap_context_causal/models/checkpoint_150.pt

