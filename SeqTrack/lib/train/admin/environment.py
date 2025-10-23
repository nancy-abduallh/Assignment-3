import os
from pathlib import Path

class Env:
    """Environment configuration for SeqTrack datasets and workspace paths."""
    def __init__(self, workspace_dir=None, data_dir=None):
        # Resolve base paths
        self.workspace_dir = Path(workspace_dir or os.getcwd())
        self.data_dir = Path(data_dir or self.workspace_dir / "data")

        # === Dataset directories ===
        self.lasot_dir = str(self.data_dir / "lasot")
        self.got10k_dir = str(self.data_dir / "got10k")
        self.coco_dir = str(self.data_dir / "coco")
        self.imagenet_dir = str(self.data_dir / "imagenet")
        self.trackingnet_dir = str(self.data_dir / "trackingnet")
        self.imagenet1k_dir = str(self.data_dir / "imagenet1k")

        # === LMDB dataset directories ===
        self.lasot_lmdb_dir = str(self.data_dir / "lasot_lmdb")
        self.got10k_lmdb_dir = str(self.data_dir / "got10k_lmdb")
        self.coco_lmdb_dir = str(self.data_dir / "coco_lmdb")
        self.imagenet_lmdb_dir = str(self.data_dir / "imagenet_lmdb")
        self.trackingnet_lmdb_dir = str(self.data_dir / "trackingnet_lmdb")

        # === Output / logs directories ===
        self.tensorboard_dir = str(self.workspace_dir / "tensorboard")
        self.save_dir = str(self.workspace_dir / "outputs")
        self.pretrained_networks = str(self.workspace_dir / "pretrained")

        # Create directories if they don't exist
        for p in [self.tensorboard_dir, self.save_dir, self.pretrained_networks]:
            try:
                os.makedirs(p, exist_ok=True)
            except Exception:
                pass


def env_settings():
    """
    Return default environment settings object.
    Uses WORKSPACE_DIR and DATA_ROOT env vars if present, otherwise defaults to cwd.
    """
    workspace_dir = os.getenv("WORKSPACE_DIR", os.getcwd())
    data_dir = os.getenv("DATA_ROOT", os.path.join(workspace_dir, "data"))
    return Env(workspace_dir, data_dir)


def create_default_local_file(workspace_dir=None, data_dir=None):
    """
    Create a minimal `local.py` file containing environment settings,
    if it doesn't already exist.
    """
    workspace_dir = workspace_dir or os.getcwd()
    data_dir = data_dir or os.path.join(workspace_dir, "data")

    local_file = Path(__file__).parent / "local.py"
    if not local_file.exists():
        with open(local_file, "w") as f:
            f.write(f"workspace_dir = r'{workspace_dir}'\n")
            f.write(f"data_dir = r'{data_dir}'\n")
        print(f"✅ Created default local.py at {local_file}")
    else:
        print(f"ℹ️ Local config already exists at {local_file}")


# === Compatibility helpers ===
def create_default_local_file_ITP_train(workspace_dir=None, data_dir=None):
    """
    Backward-compatible stub used by some SeqTrack scripts.
    Produces a small local.py file if missing.
    """
    print("⚙️ [Info] Using compatibility create_default_local_file_ITP_train.")
    create_default_local_file(workspace_dir, data_dir)


def env_settings_override():
    """
    Compatibility alias: returns same as env_settings().
    Some modules import env_settings_override; keep it for compatibility.
    """
    return env_settings()
