import os
import torch
import numpy as np
import pandas as pd
import csv
import random
from collections import OrderedDict

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from yacs.config import CfgNode as CN


class Lasot(BaseVideoDataset):
    """
    LaSOT dataset loader that respects a train split file (lasot_train_split.txt).

    Initialization:
        Lasot(root=None, image_loader=..., vid_ids=None, split=None, data_fraction=None, cfg=None)

    Behavior:
      - If split == 'train', the loader reads 'lib/train/data_specs/lasot_train_split.txt'
        and uses the listed sequences ONLY.
      - The split file may contain lines in either:
            coin-1
            coin/coin-1
        both are supported.
      - If cfg is provided and cfg.DATA.TRAIN.CLASSES exists, only sequences whose class
        is in that list will be used.
      - Missing sequence folders are skipped with a warning.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None,
                 data_fraction=None, cfg: CN = None):
        root = env_settings().lasot_dir if root is None else root
        super().__init__('LaSOT', root, image_loader)

        self.cfg = cfg
        self.image_loader = image_loader

        # Build class list from filesystem
        self.class_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        self.class_list.sort()

        # If config provides allowed classes, filter
        if self.cfg is not None and hasattr(self.cfg.DATA.TRAIN, "CLASSES"):
            allowed_classes = list(self.cfg.DATA.TRAIN.CLASSES)
            # keep only classes present in filesystem and allowed
            self.class_list = [cls for cls in self.class_list if cls in allowed_classes]
            if not self.class_list:
                raise ValueError("No allowed classes found in dataset (check cfg.DATA.TRAIN.CLASSES).")
        else:
            # no filtering; keep all available classes
            pass

        self.class_to_id = {cls_name: idx for idx, cls_name in enumerate(self.class_list)}

        # Build the list of sequence identifiers according to split/vid_ids
        self.sequence_list = self._build_sequence_list(vid_ids, split)

        # Optionally sample fraction
        if data_fraction is not None and 0 < data_fraction < 1:
            sample_n = max(1, int(len(self.sequence_list) * data_fraction))
            self.sequence_list = random.sample(self.sequence_list, sample_n)

        # Build mapping seq per class (seq indices)
        self.seq_per_class = self._build_class_list()

    # -------------------- Sequence list builders --------------------
    def _build_sequence_list(self, vid_ids=None, split=None):
        """
        Returns a list of sequence names (format: 'coin-1') based on:
          - split == 'train' -> read lasot_train_split.txt from lib/train/data_specs
          - vid_ids provided -> generate coin-<id> for ids in vid_ids for each allowed class
        """
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split and vid_ids.')

            # Path to data_specs/lasot_train_split.txt relative to this module
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            split_file = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')

            if not os.path.exists(split_file):
                raise ValueError(f"Train split file not found: {split_file}")

            # Read file: support simple lines or CSV; treat each non-empty line as sequence id
            # Use pandas to handle various newline conventions safely
            try:
                df = pd.read_csv(split_file, header=None, dtype=str)
                raw_list = df.squeeze().astype(str).tolist()
            except Exception:
                # fallback simple read
                with open(split_file, 'r') as f:
                    raw_list = [ln.strip() for ln in f.readlines() if ln.strip()]

            # Normalize entries: allow "coin/coin-1" or "coin-1" -> convert to "coin-1"
            seqs = []
            for entry in raw_list:
                entry = entry.strip()
                if not entry:
                    continue
                if '/' in entry:
                    # e.g., coin/coin-1 -> take last token 'coin-1'
                    token = entry.split('/')[-1]
                else:
                    token = entry
                # Basic validation token format: must contain '-' separating class and id
                if '-' not in token:
                    print(f"⚠️ Warning: split file entry '{entry}' looks malformed; skipping.")
                    continue
                cls_name = token.split('-')[0]
                # Filter by allowed classes if cfg provided
                if self.cfg is not None and hasattr(self.cfg.DATA.TRAIN, "CLASSES"):
                    allowed = list(self.cfg.DATA.TRAIN.CLASSES)
                    if cls_name not in allowed:
                        # skip sequences outside allowed classes
                        continue
                # Verify that the sequence folder exists under root/class/class-<id>
                seq_path = os.path.join(self.root, cls_name, token)
                if not os.path.isdir(seq_path):
                    print(f"⚠️ Warning: Sequence folder not found for '{token}' -> expected {seq_path}. Skipping.")
                    continue
                seqs.append(token)

            if not seqs:
                raise ValueError("No valid sequences found in split file after filtering/matching.")
            return seqs

        elif vid_ids is not None:
            # Build sequence names using vid_ids for each class
            seqs = []
            for cls in self.class_list:
                for vid in vid_ids:
                    seq_name = f"{cls}-{vid}"
                    seq_path = os.path.join(self.root, cls, seq_name)
                    if os.path.isdir(seq_path):
                        seqs.append(seq_name)
            if not seqs:
                raise ValueError("No sequences found for provided vid_ids.")
            return seqs
        else:
            raise ValueError('Either split or vid_ids must be set to build sequence list.')

    # -------------------- Helper builders --------------------
    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            seq_per_class.setdefault(class_name, []).append(seq_id)
        return seq_per_class

    # -------------------- Public API --------------------
    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class.get(class_name, [])

    # -------------------- Sequence path / annotations --------------------
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        if not os.path.exists(bb_anno_file):
            raise FileNotFoundError(f"Annotation file not found: {bb_anno_file}")
        gt = pd.read_csv(bb_anno_file, delimiter=',', header=None, low_memory=False)
        gt = gt.replace(r'^\s*$', np.nan, regex=True).dropna().astype(np.float32).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
        if not os.path.exists(occlusion_file) or not os.path.exists(out_of_view_file):
            # If missing, assume all visible
            length = self._get_annotation_length(seq_path)
            return torch.ones(length, dtype=torch.uint8)
        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r', newline='') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        target_visible = ~(occlusion.bool() | out_of_view.bool())
        return target_visible

    def _get_annotation_length(self, seq_path):
        # helper to infer number of frames from annotation length if available
        gt_file = os.path.join(seq_path, "groundtruth.txt")
        if os.path.exists(gt_file):
            df = pd.read_csv(gt_file, delimiter=',', header=None, low_memory=False)
            return len(df)
        # fallback: count images in img/ folder
        img_dir = os.path.join(seq_path, 'img')
        if os.path.exists(img_dir):
            imgs = [n for n in os.listdir(img_dir) if n.lower().endswith('.jpg')]
            return len(imgs)
        return 0

    def _get_sequence_path(self, seq_id):
        """
        Given sequence id index, returns full path to sequence folder.
        sequence_list entries are 'coin-1' -> class 'coin', vid '1'
        """
        seq_name = self.sequence_list[seq_id]
        # support both 'coin-1' and 'coin_1' or other separators if needed
        if '-' in seq_name:
            class_name, vid_id = seq_name.split('-', 1)
        else:
            # fallback: if user provided 'coin/coin-1' normalized earlier, so shouldn't reach here
            tokens = seq_name.split('/')
            class_name = tokens[-2] if len(tokens) >= 2 else tokens[0]
            vid_id = tokens[-1].split('-', 1)[-1]
        seq_folder = f"{class_name}-{vid_id}"
        return os.path.join(self.root, class_name, seq_folder)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # frame_id is zero-based in code; files are 1-indexed and zero-padded to 8 digits
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = os.path.normpath(seq_path).split(os.sep)[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        return self._get_class(seq_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            # copy per-frame data
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({
            'object_class_name': obj_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames
