import os
import pandas as pd
import numpy as np

root = "data/lasot"

for cls in os.listdir(root):
    class_path = os.path.join(root, cls)
    if not os.path.isdir(class_path):
        continue

    for seq in os.listdir(class_path):
        seq_path = os.path.join(class_path, seq)
        bb_file = os.path.join(seq_path, "groundtruth.txt")

        if os.path.exists(bb_file):
            try:
                df = pd.read_csv(bb_file, header=None)
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                df.to_csv(bb_file, index=False, header=False)
                print(f"✅ Cleaned {bb_file}")
            except Exception as e:
                print(f"⚠️ Error cleaning {bb_file}: {e}")
