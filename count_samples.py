import os

def load_train_split(split_file):
    """
    Reads the training split file (lasot_train_split.txt) and returns a list of sequence names.
    Handles lines that may or may not contain class prefixes.
    """
    if not os.path.exists(split_file):
        raise ValueError(f"Train split file {split_file} not found.")

    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return lines


def count_frames_in_split(data_dir, split_list, target_classes=None):
    """
    Counts frames only for sequences listed in the train split file.
    Supports both 'class/sequence' and 'sequence' formats.
    """
    counts = {}

    for entry in split_list:
        parts = entry.split('/')
        if len(parts) == 2:
            cls, seq = parts
        elif len(parts) == 1:
            # Try to infer class from folder structure
            seq = parts[0]
            found = False
            for cls in (target_classes or []):
                class_seq_path = os.path.join(data_dir, cls, seq)
                if os.path.exists(class_seq_path):
                    found = True
                    break
            if not found:
                print(f"⚠️ Warning: Could not determine class for sequence '{seq}'")
                continue
        else:
            print(f"⚠️ Invalid entry format: {entry}")
            continue

        if target_classes and cls not in target_classes:
            continue

        img_path = os.path.join(data_dir, cls, seq, 'img')
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Missing folder {img_path}")
            continue

        frame_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_frames = len(frame_files)

        if cls not in counts:
            counts[cls] = {'sequences': 0, 'frames': 0}

        counts[cls]['sequences'] += 1
        counts[cls]['frames'] += num_frames

    return counts


# ====== 🧠 USAGE ======
DATA_DIR = os.path.expanduser('~/assignment_3/SeqTrack/data/lasot')
SPLIT_FILE = os.path.expanduser('~/assignment_3/SeqTrack/lib/train/data_specs/lasot_train_split.txt')
TARGET_CLASSES = ['coin', 'kite']

train_split = load_train_split(SPLIT_FILE)
sample_counts = count_frames_in_split(DATA_DIR, train_split, TARGET_CLASSES)

# ====== 📊 RESULTS ======
print("\n📁 Dataset Counts (from train split):")
for cls, info in sample_counts.items():
    print(f"  {cls}: {info['sequences']} sequences, {info['frames']} frames")

total_sequences = sum(info['sequences'] for info in sample_counts.values())
total_frames = sum(info['frames'] for info in sample_counts.values())
print(f"\n✅ Total sequences: {total_sequences}")
print(f"✅ Total frames: {total_frames}")
