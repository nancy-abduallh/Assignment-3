import os
import re
from typing import Union, Any

import torch
import importlib
import random
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import HfApi
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.train.trainers import LTRTrainer
from lib.models.seqtrack import build_seqtrack
from lib.train.actors import SeqTrackActor
from .base_functions import *

import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# -------------------- HUGGINGFACE --------------------
HF_REPO = os.environ.get("HF_REPO", "NancyAbdullah11/assignment_3")
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# -------------------- SEED CONTROL --------------------
def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- CHECKPOINT SAVE --------------------
def save_checkpoint(epoch, trainer, checkpoint_dir, settings, extra=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_state = (
        trainer.actor.net.module.state_dict()
        if hasattr(trainer.actor.net, "module")
        else trainer.actor.net.state_dict()
    )

    lr_scheduler_state = None
    if trainer.lr_scheduler:
        lr_scheduler_state = {
            "state_dict": trainer.lr_scheduler.state_dict(),
            "last_epoch": getattr(trainer.lr_scheduler, "last_epoch", -1),
            "_step_count": getattr(trainer.lr_scheduler, "_step_count", 0),
            "_last_lr": getattr(trainer.lr_scheduler, "_last_lr", []),
        }

    torch_rng = torch.get_rng_state()
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        cuda_rng = [state.cpu().to(torch.uint8) for state in cuda_states]

    state = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": trainer.optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler_state,
        "rng_state": {
            "torch": torch_rng,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": cuda_rng,
        },
        "settings": getattr(settings, "_dict_", None),
        "extra": extra or {},
    }

    log_path = getattr(settings, "log_file", os.path.join(checkpoint_dir, "training.log"))
    with open(log_path, "a") as log:
        log.write(f"\n--- Checkpoint Epoch {epoch} ---\n")
        log.write(f"Optimizer State Keys: {list(state['optimizer_state'].keys())}\n")
        if state["lr_scheduler_state"]:
            lr_state = state["lr_scheduler_state"]
            log.write("LR Scheduler State:\n")
            log.write(f"  - last_epoch: {lr_state.get('last_epoch', 'N/A')}\n")
            log.write(f"  - _step_count: {lr_state.get('_step_count', 'N/A')}\n")
            log.write(f"  - _last_lr: {lr_state.get('_last_lr', 'N/A')}\n")

        rng_info = state["rng_state"]
        log.write("RNG States Summary:\n")
        log.write(f"  Torch RNG length: {len(rng_info['torch'])}\n")
        log.write(f"  NumPy RNG first 3: {rng_info['numpy'][1][:3]}\n")
        if rng_info["cuda"]:
            log.write(f"  CUDA RNG for {len(rng_info['cuda'])} devices saved\n")

    fname = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, fname)
    torch.save(state, os.path.join(checkpoint_dir, "latest.pth"))
    print(f"💾 Saved checkpoint: {fname}")


# -------------------- CHECKPOINT LOAD --------------------
def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint(trainer, checkpoint_dir, resume_epoch=None, settings=None):
    if resume_epoch is None or resume_epoch == -1:
        path = os.path.join(checkpoint_dir, "latest.pth")
        if not os.path.exists(path):
            print("🔍 No checkpoint found. Starting from epoch 1.")
            return 1
    else:
        checkpoint_epoch = resume_epoch - 1
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint_epoch}.pth")
        if not os.path.exists(path):
            print(f"❌ Missing checkpoint for epoch {checkpoint_epoch}")
            return 1

    print(f"🔁 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")

    if hasattr(trainer.actor.net, "module"):
        trainer.actor.net.module.load_state_dict(checkpoint["model_state"])
    else:
        trainer.actor.net.load_state_dict(checkpoint["model_state"])

    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    _move_optimizer_state_to_device(trainer.optimizer, trainer.device)

    if trainer.lr_scheduler and checkpoint.get("lr_scheduler_state"):
        lr_state = checkpoint["lr_scheduler_state"]
        trainer.lr_scheduler.load_state_dict(lr_state.get("state_dict", {}))
        trainer.lr_scheduler.last_epoch = checkpoint.get("epoch", lr_state.get("last_epoch", -1))

    rng = checkpoint.get("rng_state", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    if "cuda" in rng and rng["cuda"] and torch.cuda.is_available():
        try:
            restored_states = []
            for i, state in enumerate(rng["cuda"]):
                if not isinstance(state, torch.ByteTensor):
                    state = torch.ByteTensor(state.cpu().to(torch.uint8))
                restored_states.append(state.to(f"cuda:{i}"))
            torch.cuda.set_rng_state_all(restored_states)
            print(f"✅ Restored CUDA RNG for {len(restored_states)} device(s)")
        except Exception as e:
            print(f"❌ CUDA RNG restore failed: {e}")
            torch.cuda.manual_seed_all(torch.initial_seed())

    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"📊 Restored from epoch: {checkpoint.get('epoch', 'N/A')}")
    return checkpoint["epoch"] + 1 if resume_epoch == -1 else resume_epoch


# -------------------- HF UPLOAD --------------------
def upload_all_checkpoints_to_hf(checkpoint_dir):
    if not HF_TOKEN:
        print("⚠ HF upload skipped (no token).")
        return
    try:
        api = HfApi()
        for fname in os.listdir(checkpoint_dir):
            if fname.endswith(".pth"):
                api.upload_file(
                    path_or_fileobj=os.path.join(checkpoint_dir, fname),
                    path_in_repo=f"checkpoints/{fname}",
                    repo_id=HF_REPO,
                    token=HF_TOKEN,
                    repo_type="model",
                )
                print(f"⬆ Uploaded {fname}")
    except Exception as e:
        print(f"⚠ HF upload failed: {e}")


# -------------------- PHASE 1 LOG PARSING --------------------
def load_phase1_metrics_from_log(log_file):
    """Extract epoch, loss, and IoU values from Phase 1 log file."""
    if not os.path.exists(log_file):
        print(f"⚠ No Phase 1 log file found at {log_file}")
        return {'epochs': [], 'loss': [], 'iou': []}

    phase1 = {'epochs': [], 'loss': [], 'iou': []}
    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s*:.*Loss/total:\s*([\d.]+),\s*IoU:\s*([\d.]+)", re.IGNORECASE)
    summary_pattern = re.compile(
        r"^Epoch\s+(\d+)\s+Summary:\s+Average\s+Loss:\s*([\d.]+)\s+Average\s+IoU:\s*([\d.]+)",
        re.IGNORECASE | re.MULTILINE
    )

    with open(log_file, "r") as f:
        text = f.read()

    for match in summary_pattern.findall(text):
        epoch, loss, iou = int(match[0]), float(match[1]), float(match[2])
        phase1['epochs'].append(epoch)
        phase1['loss'].append(loss)
        phase1['iou'].append(iou)

    if not phase1['epochs']:  # fallback if summaries are missing
        for match in epoch_pattern.findall(text):
            epoch, loss, iou = int(match[0]), float(match[1]), float(match[2])
            phase1['epochs'].append(epoch)
            phase1['loss'].append(loss)
            phase1['iou'].append(iou)

    print(f"📜 Loaded {len(phase1['epochs'])} epochs from Phase 1 log.")
    return phase1


# -------------------- PLOTTING --------------------
def plot_phase_comparison(phase1_metrics, phase2_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(phase1_metrics['epochs'], phase1_metrics['loss'], label='Phase 1', marker='o')
    plt.plot(phase2_metrics['epochs'], phase2_metrics['loss'], label='Phase 2', marker='s')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Comparison'); plt.legend()
    loss_png = os.path.join(save_dir, 'loss_phase_comparison.png')
    plt.savefig(loss_png); plt.close()
    print(f"📈 Saved {loss_png}")

    plt.figure(figsize=(8, 5))
    plt.plot(phase1_metrics['epochs'], phase1_metrics['iou'], label='Phase 1', marker='o')
    plt.plot(phase2_metrics['epochs'], phase2_metrics['iou'], label='Phase 2', marker='s')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.title('IoU Comparison'); plt.legend()
    iou_png = os.path.join(save_dir, 'iou_phase_comparison.png')
    plt.savefig(iou_png); plt.close()
    print(f"📈 Saved {iou_png}")


# -------------------- MAIN TRAIN FUNCTION --------------------
def run(settings):
    settings.description = "Training script for SeqTrack"
    cfg_module = importlib.import_module(f"lib.config.{settings.script_name}.config")
    cfg = cfg_module.cfg
    cfg_module.update_config_from_file(settings.cfg_file)
    update_settings(settings, cfg)

    checkpoint_dir = os.path.join(settings.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not getattr(settings, "resume_from_epoch", None):
        print(f"⚙ Setting global seed: {settings.seed}")
        set_global_seed(settings.seed)
    else:
        print("⚙ Enforcing deterministic CUDA flags for resume consistency.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loader_train = build_dataloaders(cfg, settings)
    net = build_seqtrack(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if settings.local_rank != -1 and torch.cuda.is_available():
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device(f"cuda:{settings.local_rank}")
    else:
        settings.device = device

    bins = cfg.MODEL.BINS
    weight = torch.ones(bins + 2).to(device)
    weight[bins] = weight[bins + 1] = 0.01
    objective = {"ce": CrossEntropyLoss(weight=weight)}

    actor = SeqTrackActor(net, objective, {"ce": cfg.TRAIN.CE_WEIGHT}, settings, cfg)
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    total_epochs = cfg.TRAIN.EPOCH
    resume_from = getattr(settings, "resume_from_epoch", None)
    start_epoch = load_checkpoint(trainer, checkpoint_dir, resume_from, settings) if resume_from else 1

    log_file = os.path.join("outputs", "logs", "seqtrack-seqtrack_b256.log")
    phase1_metrics = load_phase1_metrics_from_log(log_file) if start_epoch > 1 else {'epochs': [], 'loss': [], 'iou': []}
    phase2_metrics = {'epochs': [], 'loss': [], 'iou': []}

    print(f"🚀 Starting training from epoch {start_epoch}/{total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        trainer.epoch = epoch
        print(f"\n=== Epoch {epoch}/{total_epochs} ===")
        trainer.train_epoch()

        loader_name = list(trainer.stats.keys())[0]
        stats = trainer.stats[loader_name]
        loss_val = stats.get("Loss/total", 0).avg if stats.get("Loss/total") else 0
        iou_val = stats.get("IoU", 0).avg if stats.get("IoU") else 0

        if start_epoch == 1:
            phase1_metrics['epochs'].append(epoch)
            phase1_metrics['loss'].append(loss_val)
            phase1_metrics['iou'].append(iou_val)
        else:
            phase2_metrics['epochs'].append(epoch)
            phase2_metrics['loss'].append(loss_val)
            phase2_metrics['iou'].append(iou_val)

        save_checkpoint(epoch, trainer, checkpoint_dir, settings)

    plot_phase_comparison(phase1_metrics, phase2_metrics, os.path.join(settings.save_dir, "plots"))
    upload_all_checkpoints_to_hf(checkpoint_dir)
    print("🎯 Training complete.")
