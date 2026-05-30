import os
import numpy as np
import pandas as pd
import random

EPISODES_CSV = "analysis/episodes_from_json_all.csv"
SKELETON_DIR = "data_original/npy"

SEG_LEN = 16
POS_OVERLAP = 0.75
NEG_OVERLAP = 0.05


def compute_overlap(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0


print("==== Loading episodes ====")
df = pd.read_csv(EPISODES_CSV)
print("Total episodes:", len(df))

print("\n==== Checking episode validity ====")

invalid_count = 0

for i, row in df.iterrows():
    if row.start >= row.end:
        print(f"[INVALID] video {row.video_id} start={row.start} end={row.end}")
        invalid_count += 1

print("Invalid episodes:", invalid_count)

print("\n==== Checking skeleton length ====")

for vid in df.video_id.unique():
    sk_path = os.path.join(SKELETON_DIR, f"{vid}.npy")
    if not os.path.exists(sk_path):
        print(f"[MISSING] skeleton {vid}")
        continue

    T = np.load(sk_path).shape[0]
    eps = df[df.video_id == vid]

    for _, row in eps.iterrows():
        if row.end > T:
            print(f"[OUT OF RANGE] video {vid} end={row.end} T={T}")

print("Skeleton range check done.")


print("\n==== Sampling windows for sanity check ====")

sample_checks = 20

for vid in random.sample(list(df.video_id.unique()), 
                         min(5, len(df.video_id.unique()))):

    sk_path = os.path.join(SKELETON_DIR, f"{vid}.npy")
    if not os.path.exists(sk_path):
        continue

    sk = np.load(sk_path)
    T = sk.shape[0]
    eps = df[df.video_id == vid]

    print(f"\n--- Video {vid} ---")

    for _ in range(sample_checks):
        start = random.randint(0, T - SEG_LEN - 1)
        end = start + SEG_LEN

        overlaps = []
        for _, row in eps.iterrows():
            overlaps.append(
                compute_overlap(start, end, row.start, row.end)
            )

        max_overlap = max(overlaps) if overlaps else 0

        label = 1 if max_overlap >= POS_OVERLAP else 0

        print(
            f"Window [{start},{end}] max_overlap={max_overlap:.3f} label={label}"
        )