import os
import sys
import torch

# ✅ Ensure project root (SeqTrack) is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Go up 2 levels to SeqTrack/
sys.path.insert(0, PROJECT_ROOT)

# ✅ Load checkpoint (relative to this file)
ckpt_path = os.path.join(CURRENT_DIR, "outputs/checkpoints/checkpoint_epoch_3.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")

print(ckpt.keys())
