# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2026 Ke Zhang (kylezhang1118@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import tableprint as tp
import torch
import torch.nn.functional as F

from wesep.utils.funcs import clip_gradients
from wesep.dataset.collate import AUX_KEY_MAP


class Executor:

    def __init__(self, aux_key_map=None, spk2id_dict=None, sv_loss_weight=0.5):
        self.step = 0
        self.aux_key_map = aux_key_map or AUX_KEY_MAP

        self.cue_keys = list(self.aux_key_map.values())
        self.spk2id_dict = spk2id_dict
        self.sv_loss_weight = sv_loss_weight

    # -------------------------
    # helpers
    # -------------------------

    def _extract_model_inputs(self, batch, device):
        if "wav_mix" not in batch:
            raise RuntimeError("[executor] Missing required key: wav_mix")
        if "wav_target" not in batch:
            raise RuntimeError("[executor] Missing required key: wav_target")

        mix = batch["wav_mix"].float().to(device)
        target = batch["wav_target"].float().to(device)

        cues = []
        for k in self.cue_keys:
            if k in batch and batch[k] is not None:
                cues.append(batch[k].float().to(device))

        if len(cues) == 0:
            cues = None

        # 🌟 修正：处理验证集中的开集说话人
        spk_id = None
        if "spk_id" in batch:
            spk_id = batch["spk_id"].long().to(device)
        elif "spk" in batch and self.spk2id_dict is not None:
            try:
                spk_ids_list = [self.spk2id_dict[str(s)] for s in batch["spk"]]
                spk_id = torch.tensor(spk_ids_list,
                                      dtype=torch.long).to(device)
            except KeyError:
                # 在验证集（开集）中，说话人不在训练字典内是正常现象，直接忽略
                spk_id = None

        return mix, cues, target, spk_id

    # -------------------------
    # train
    # -------------------------

    def train(self,
              dataloader,
              models,
              epoch_iter,
              optimizers,
              criterion,
              schedulers,
              scaler,
              epoch,
              enable_amp,
              logger,
              clip_grad=5.0,
              log_batch_interval=100,
              device=torch.device("cuda"),
              se_loss_weight=1.0):

        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval

        losses = []
        losses_se = []
        losses_sv = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                mix, cues, target, spk_id = self._extract_model_inputs(
                    batch, device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if cues is None:
                        outputs = model(mix)
                    else:
                        outputs = model(mix, cues)

                    spk_logits = None
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        est_wav, spk_logits = outputs
                        # print(f"double")
                        outputs = [est_wav]
                    elif not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]

                    loss_se_tensor = 0.0
                    for ii in range(len(criterion)):
                        for ji in range(len(se_loss_weight[0][ii])):
                            out_idx = se_loss_weight[0][ii][ji]
                            w = se_loss_weight[1][ii][ji]
                            loss_se_tensor = loss_se_tensor + w * (
                                criterion[ii](outputs[out_idx], target).mean())

                    loss = loss_se_tensor
                    loss_sv_val = 0.0

                    if spk_logits is not None and spk_id is not None:
                        loss_sv_tensor = F.cross_entropy(spk_logits, spk_id)
                        loss_sv_val = loss_sv_tensor.item()
                        loss = loss + self.sv_loss_weight * loss_sv_tensor

                losses.append(loss.item())
                losses_se.append(loss_se_tensor.item() if isinstance(
                    loss_se_tensor, torch.Tensor) else loss_se_tensor)
                losses_sv.append(loss_sv_val)

                total_loss_avg = sum(losses) / len(losses)
                se_loss_avg = sum(losses_se) / len(losses_se)
                sv_loss_avg = sum(losses_sv) / len(losses_sv)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    loss_str = f"Tot:{total_loss_avg:.3f} (SE:{se_loss_avg:.3f}, SV:{sv_loss_avg:.3f})"
                    logger.info(
                        tp.row(
                            ("TRAIN", epoch, i + 1, loss_str,
                             optimizer.param_groups[0]["lr"]),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == epoch_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0

    # -------------------------
    # cv / validation
    # -------------------------

    def cv(self,
           dataloader,
           models,
           val_iter,
           criterion,
           epoch,
           enable_amp,
           logger,
           log_batch_interval=100,
           device=torch.device("cuda")):

        model = models[0]
        model.eval()

        log_interval = log_batch_interval

        # 🌟 修正：验证集只追踪单一的 SE 分离 Loss
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                mix, cues, target, _ = self._extract_model_inputs(
                    batch, device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if cues is None:
                        outputs = model(mix)
                    else:
                        outputs = model(mix, cues)

                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        est_wav, _ = outputs  # 🌟 修正：直接丢弃 SV logits，完全不参与计算
                        outputs = [est_wav]
                    elif not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]

                    # 验证集只算波形域的分离 loss
                    loss = criterion[0](outputs[0], target).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    # 🌟 修正：验证集纯净输出，不带虚假的 SV loss
                    loss_str = f"SE:{total_loss_avg:.3f}"
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, loss_str, "-"),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == val_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0
