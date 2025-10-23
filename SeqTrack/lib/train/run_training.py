import os
import sys
import argparse
import importlib
import time
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn
import cv2 as cv

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TEAM_NUMBER = 9
HF_REPO = "NancyAbdullah11/assignment_3"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

LOG_FILE = os.path.join(project_root, "training_log.txt")
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_data_environment():
    """Setup data environment variables"""
    data_paths = [
        '/home/farha/assignment_3/SeqTrack/data',
        '/home/farha/assignment_3/data',
        './data'
    ]
    for data_path in data_paths:
        expanded_path = os.path.expanduser(data_path)
        if os.path.exists(expanded_path):
            os.environ['DATA_ROOT'] = expanded_path
            print(f"✓ Setting DATA_ROOT to: {expanded_path}")
            return expanded_path
    print("⚠️ No data directory found. Please ensure dataset is downloaded.")
    return None


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1,
                 save_dir=None, base_seed=None, use_lmdb=False, resume_from_epoch=None):
    """Run SeqTrack training pipeline."""

    data_root = setup_data_environment()
    if data_root is None:
        print("❌ Cannot proceed without dataset. Please download LaSOT dataset first.")
        return

    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # --- CRITICAL FIX: Set seed only for Phase 1, not for resuming ---
    is_fresh_start = resume_from_epoch is None or resume_from_epoch == 1

    if is_fresh_start:
        print(f"🔧 Setting fixed seed {base_seed} for Phase 1 (fresh start).")
        set_seed(base_seed)
    else:
        print(f"🔧 Skipping fixed seed setting for Phase 2 (resume_from_epoch={resume_from_epoch}).")
        print("🔧 RNG states will be loaded from checkpoint.")

    # Setup logging
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting SeqTrack training | Script: {script_name}, Config: {config_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Team seed: {TEAM_NUMBER}")

    try:
        import lib.train.admin.settings as ws_settings
        settings = ws_settings.Settings()
        settings.script_name = script_name
        settings.config_name = config_name
        settings.project_path = f"train/{script_name}/{config_name}"
        settings.local_rank = local_rank
        settings.save_dir = os.path.abspath(save_dir or ".")
        settings.use_lmdb = use_lmdb
        settings.device = device
        settings.resume_from_epoch = resume_from_epoch
        settings.seed = base_seed  # Pass seed to train_script

        settings.cfg_file = os.path.join(project_root, f"experiments/{script_name}/{config_name}.yaml")
        if not os.path.exists(settings.cfg_file):
            logger.error(f"Config file not found: {settings.cfg_file}")
            return

        logger.info(f"Using config file: {settings.cfg_file}")

        expr_module = importlib.import_module("lib.train.train_script")
        expr_func = getattr(expr_module, "run")

        # --- Detect Phase ---
        if is_fresh_start:
            phase = "Phase 1 (Initial Training)"
        elif settings.resume_from_epoch == -1:
            phase = "Phase 2 (Auto Resume from latest checkpoint)"
        else:
            phase = f"Phase 2 (Manual Resume from epoch {settings.resume_from_epoch})"

        print(f"\n🚀 Starting {phase}")
        logger.info(f"🚀 Starting {phase}")

        # --- Start training ---
        expr_func(settings)

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Run SeqTrack training.")
    parser.add_argument("--script", type=str, required=True, help="Training script name")
    parser.add_argument("--config", type=str, required=True, help="Config name")
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=TEAM_NUMBER)
    parser.add_argument("--use_lmdb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--resume_from_epoch", type=int, default=None,
                        help="If set, resume training manually from a given epoch (Phase 2 trigger).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training automatically from latest checkpoint.")

    args = parser.parse_args()

    # If --resume is used, automatically activate resume mode
    if args.resume:
        args.resume_from_epoch = -1  # signal for 'auto resume from latest'

    run_training(
        args.script,
        args.config,
        cudnn_benchmark=args.cudnn_benchmark,
        local_rank=args.local_rank,
        save_dir=args.save_dir,
        base_seed=args.seed,
        use_lmdb=args.use_lmdb,
        resume_from_epoch=args.resume_from_epoch
    )


if __name__ == "__main__":
    main()
