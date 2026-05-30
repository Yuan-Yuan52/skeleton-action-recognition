"""
augment_flip_dataset.py - Generate flipped skeleton npy files for data_original/npy
"""
import os
import numpy as np
import pandas as pd

# MediaPipe Pose 33 left/right symmetric joint pairs
LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

def flip_skeleton(seq: np.ndarray) -> np.ndarray:
    """Horizontally flip a skeleton sequence (T, 33, 3)."""
    flipped = seq.copy()
    flipped[:, :, 0] = -flipped[:, :, 0]
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        tmp = flipped[:, left_idx, :].copy()
        flipped[:, left_idx, :] = flipped[:, right_idx, :]
        flipped[:, right_idx, :] = tmp
    return flipped


def main():
    skeleton_dir = "data_original/npy"

    npy_files = [f for f in os.listdir(skeleton_dir)
                 if f.endswith(".npy") and "_flip" not in f]
    print(f"[INFO] Found {len(npy_files)} original .npy files in {skeleton_dir}")

    created = skipped = 0
    for fname in sorted(npy_files):
        src_path = os.path.join(skeleton_dir, fname)
        dst_path = os.path.join(skeleton_dir, fname.replace(".npy", "_flip.npy"))
        if os.path.exists(dst_path):
            print(f"  [SKIP] {dst_path} already exists.")
            skipped += 1
            continue
        seq = np.load(src_path)
        flipped = flip_skeleton(seq)
        np.save(dst_path, flipped)
        print(f"  [OK] {dst_path}  shape={flipped.shape}")
        created += 1

    print(f"\n[INFO] Done. Created={created}, Skipped={skipped}")


if __name__ == "__main__":
    main()
