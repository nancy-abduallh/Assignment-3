import os
from lib.train.admin.environment import env_settings  # import the stable function


class Settings:
    """
    Training settings (used across trainer, logger, and experiment setup).
    This class stores global paths and configuration parameters used during training.
    """

    def __init__(self):
        self.set_default()

    def set_default(self):
        """Set default settings including environment paths and logging configuration."""
        # Environment (datasets, checkpoints, outputs)
        self.env = env_settings()

        # Enable GPU usage (default)
        self.use_gpu = True

        # === Define Log File Path ===
        log_dir = os.path.join(self.env.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Use the experiment + config name format for log filename
        self.log_file = os.path.join(log_dir, "seqtrack-seqtrack_b256.log")

        # === Define Checkpoint Directory (Optional but Recommended) ===
        self.checkpoint_dir = os.path.join(self.env.save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # === Other Common Settings ===
        self.save_every_epoch = True  # save checkpoints each epoch
        self.print_interval = 10      # print every N iterations
        self.random_seed = 9          # same seed for reproducibility

        # Defaults that SeqTrack expects (sensible defaults; override via config)
        self.local_rank = -1
        self.use_lmdb = False
        self.cfg = None
        self.script_name = "seqtrack"
        self.config_name = "seqtrack_b256"
        self.seed = self.random_seed
        self.device = None

    def __repr__(self):
        return (
            f"Settings(\n"
            f"  use_gpu={self.use_gpu},\n"
            f"  log_file='{self.log_file}',\n"
            f"  checkpoint_dir='{self.checkpoint_dir}',\n"
            f"  save_dir='{self.env.save_dir}'\n"
            f")"
        )
