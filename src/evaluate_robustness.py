"""
evaluate_robustness.py
======================
Evaluate the robustness of Transformer, GRU, and ST-GCN models under:
1. Gaussian Noise (Jitter): std = 0.0, 0.005, 0.01, 0.02, 0.05, 0.1
2. Random Joint Occlusion: num_joints = 0, 1, 2, 3, 4, 5 (from core joints)
3. Specific Joint Occlusion:
   - Wrists (15, 16)
   - Knees (25, 26)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)
sys.path.append(os.path.abspath(os.path.join(src_dir, '../NTU/stgcn')))

from models_transformer import SpatialTemporalTransformer
from models_skeleton import GRUClassifier
from net.st_gcn import Model as STGCN
from train_transformer import build_phase_table, split_by_video, normalize_skeleton, resample_seq
from utils import seed_everything

EPISODES_CSV = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR = "data_original/npy"
TARGET_COL   = "class_id"
NUM_CLASSES  = 5
SEED         = 42
OUT_DIR      = "thesis_materials"
CORE_JOINTS  = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Custom dataset that allows applying noise/occlusion dynamically at evaluation
class RobustnessEvalDataset(Dataset):
    def __init__(self, df, skeleton_dir, out_len=64, noise_std=0.0, occluded_joints=None):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.noise_std = noise_std
        self.occluded_joints = occluded_joints if occluded_joints is not None else []
        self._skeleton_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_skeleton(self, video_id):
        if video_id in self._skeleton_cache:
            return self._skeleton_cache[video_id]
        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"skeleton npy not found: {npy_path}")
        seq = np.load(npy_path).astype(np.float32)
        self._skeleton_cache[video_id] = seq
        return seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        s = int(row["start_frame_skel"])
        e = int(row["end_frame_skel"])
        label = int(row["label"])

        full_seq = self._load_skeleton(video_id)
        T = full_seq.shape[0]

        s = max(0, min(s, T - 1))
        e = max(0, min(e, T - 1))
        if e < s:
            e = s

        seg = full_seq[s:e + 1].copy()

        # 1. Resample to fixed T=64
        seg = resample_seq(seg, self.out_len)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Apply Noise (Jitter)
        if self.noise_std > 0.0:
            noise = np.random.normal(0, self.noise_std, size=seg.shape).astype(np.float32)
            seg += noise

        # 3. Apply Joint Occlusion (zero out joints)
        for j in self.occluded_joints:
            seg[:, j, :] = 0.0

        return torch.from_numpy(seg).float(), label

def evaluate_models(val_df, device, noise_std=0.0, occluded_joints=None):
    dataset = RobustnessEvalDataset(val_df, SKELETON_DIR, out_len=64, noise_std=noise_std, occluded_joints=occluded_joints)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 1. Load Transformer
    transformer_model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)
    tx_ckpt = torch.load("checkpoints/best_S2T2_noNorm_flip/best.pth", map_location=device, weights_only=False)
    transformer_model.load_state_dict(tx_ckpt["state_dict"])
    transformer_model.eval()

    # 2. Load GRU (33 joints)
    gru_model = GRUClassifier(num_joints=33, in_channels=6, num_classes=NUM_CLASSES).to(device)
    gru_ckpt = torch.load("checkpoints/phase_newway_class4_gru/lift_class_id/best.pth", map_location=device, weights_only=False)
    gru_model.load_state_dict(gru_ckpt["state_dict"])
    gru_model.eval()

    # 2b. Load GRU (13 joints)
    gru13_model = GRUClassifier(num_joints=13, in_channels=6, num_classes=NUM_CLASSES).to(device)
    gru13_ckpt = torch.load("checkpoints/gru_13joints/best.pth", map_location=device, weights_only=False)
    gru13_model.load_state_dict(gru13_ckpt["state_dict"])
    gru13_model.eval()

    # 3. Load ST-GCN
    stgcn_model = STGCN(
        in_channels=3, num_class=NUM_CLASSES,
        graph_args={"layout": "coco", "strategy": "spatial"},
        edge_importance_weighting=True
    ).to(device)
    stgcn_ckpt = torch.load("checkpoints/stgcn_v2/lift_class_id/best.pth", map_location=device, weights_only=False)
    stgcn_model.load_state_dict(stgcn_ckpt["state_dict"])
    stgcn_model.eval()

    tx_correct, gru_correct, gru13_correct, stgcn_correct = 0, 0, 0, 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            total += y.size(0)

            # --- Eval Transformer ---
            # No Norm, select 13 core joints
            x_tx = x[:, :, CORE_JOINTS, :].to(device)
            tx_logits = transformer_model(x_tx)
            tx_correct += (tx_logits.argmax(dim=1) == y).sum().item()

            # --- Eval GRU & ST-GCN ---
            # Both need normalized skeletons
            x_norm = torch.stack([torch.from_numpy(normalize_skeleton(item.numpy())).float() for item in x]).to(device)

            # GRU with velocity features (33 joints)
            velocity = torch.zeros_like(x_norm)
            velocity[:, 1:, :, :] = x_norm[:, 1:, :, :] - x_norm[:, :-1, :, :]
            x_gru = torch.cat([x_norm, velocity], dim=-1)
            gru_logits = gru_model(x_gru)
            gru_correct += (gru_logits.argmax(dim=1) == y).sum().item()

            # GRU (13 joints)
            x_norm_13 = x_norm[:, :, CORE_JOINTS, :]
            velocity_13 = torch.zeros_like(x_norm_13)
            velocity_13[:, 1:, :, :] = x_norm_13[:, 1:, :, :] - x_norm_13[:, :-1, :, :]
            x_gru_13 = torch.cat([x_norm_13, velocity_13], dim=-1)
            gru13_logits = gru13_model(x_gru_13)
            gru13_correct += (gru13_logits.argmax(dim=1) == y).sum().item()

            # ST-GCN with coco mapping
            coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            x_stgcn = x_norm[:, :, coco_mapping, :].permute(0, 3, 1, 2).unsqueeze(-1)
            stgcn_logits = stgcn_model(x_stgcn)
            stgcn_correct += (stgcn_logits.argmax(dim=1) == y).sum().item()

    return tx_correct / total, gru_correct / total, gru13_correct / total, stgcn_correct / total

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness evaluation on device: {device}")

    # Build validation split
    df_orig = pd.read_csv(EPISODES_CSV)
    df_orig = df_orig.dropna(subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df_orig = df_orig[df_orig[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()
    
    def norm_vid(v):
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    phase_orig = pd.DataFrame({
        "video_id":         df_orig["video_id"].apply(norm_vid),
        "start_frame_skel": df_orig["lift_start_frame"].astype(int) // 2,
        "end_frame_skel":   df_orig["lift_end_frame"].astype(int) // 2,
        "label":            df_orig[TARGET_COL].astype(int)
    })
    
    video_ids = sorted(phase_orig["video_id"].unique())
    random.Random(SEED).shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * 0.2))
    val_vids = set(video_ids[:n_val])
    val_df = phase_orig[phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)
    print(f"Validation episodes count: {len(val_df)}")

    robustness_data = []

    # 1. Evaluate Gaussian Noise Jitter
    print("\n>>> Evaluating Gaussian Noise...")
    noise_stds = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    for std in noise_stds:
        tx_acc, gru_acc, gru13_acc, stgcn_acc = evaluate_models(val_df, device, noise_std=std)
        print(f"Noise std={std:.3f} | Tx: {tx_acc*100:.2f}% | GRU (33J): {gru_acc*100:.2f}% | GRU (13J): {gru13_acc*100:.2f}% | ST-GCN: {stgcn_acc*100:.2f}%")
        robustness_data.append({
            "Perturbation": "Gaussian Noise",
            "Intensity": std,
            "Transformer": tx_acc * 100,
            "GRU_33J": gru_acc * 100,
            "GRU_13J": gru13_acc * 100,
            "ST_GCN": stgcn_acc * 100
        })

    # 2. Evaluate Random Joint Occlusion
    print("\n>>> Evaluating Random Joint Occlusion...")
    occlude_counts = [0, 1, 2, 3, 4, 5]
    for count in occlude_counts:
        # Pick random joints from CORE_JOINTS
        # To make it deterministic for evaluation, we fix seed
        random.seed(SEED)
        occluded = random.sample(CORE_JOINTS, count) if count > 0 else []
        tx_acc, gru_acc, gru13_acc, stgcn_acc = evaluate_models(val_df, device, occluded_joints=occluded)
        print(f"Occluded joints count={count} {occluded} | Tx: {tx_acc*100:.2f}% | GRU (33J): {gru_acc*100:.2f}% | GRU (13J): {gru13_acc*100:.2f}% | ST-GCN: {stgcn_acc*100:.2f}%")
        robustness_data.append({
            "Perturbation": "Random Joint Occlusion",
            "Intensity": count,
            "Transformer": tx_acc * 100,
            "GRU_33J": gru_acc * 100,
            "GRU_13J": gru13_acc * 100,
            "ST_GCN": stgcn_acc * 100
        })

    # 3. Evaluate Specific Occlusion (Wrists vs. Knees)
    print("\n>>> Evaluating Specific Joint Occlusions...")
    # Wrists (15, 16)
    tx_acc_w, gru_acc_w, gru13_acc_w, stgcn_acc_w = evaluate_models(val_df, device, occluded_joints=[15, 16])
    print(f"Occluded Wrists (15, 16) | Tx: {tx_acc_w*100:.2f}% | GRU (33J): {gru_acc_w*100:.2f}% | GRU (13J): {gru13_acc_w*100:.2f}% | ST-GCN: {stgcn_acc_w*100:.2f}%")
    robustness_data.append({
        "Perturbation": "Wrist Occlusion",
        "Intensity": 2,
        "Transformer": tx_acc_w * 100,
        "GRU_33J": gru_acc_w * 100,
        "GRU_13J": gru13_acc_w * 100,
        "ST_GCN": stgcn_acc_w * 100
    })

    # Knees (25, 26)
    tx_acc_k, gru_acc_k, gru13_acc_k, stgcn_acc_k = evaluate_models(val_df, device, occluded_joints=[25, 26])
    print(f"Occluded Knees (25, 26) | Tx: {tx_acc_k*100:.2f}% | GRU (33J): {gru_acc_k*100:.2f}% | GRU (13J): {gru13_acc_k*100:.2f}% | ST-GCN: {stgcn_acc_k*100:.2f}%")
    robustness_data.append({
        "Perturbation": "Knee Occlusion",
        "Intensity": 2,
        "Transformer": tx_acc_k * 100,
        "GRU_33J": gru_acc_k * 100,
        "GRU_13J": gru13_acc_k * 100,
        "ST_GCN": stgcn_acc_k * 100
    })

    # Save to CSV
    df_robust = pd.DataFrame(robustness_data)
    df_robust.to_csv(os.path.join(OUT_DIR, "robustness_results.csv"), index=False)
    print(f"\nSaved robustness results to {os.path.join(OUT_DIR, 'robustness_results.csv')}")

    # 4. Plot comparative figures
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Plot 1: Gaussian Noise
    df_noise = df_robust[df_robust["Perturbation"] == "Gaussian Noise"]
    axes[0].plot(df_noise["Intensity"], df_noise["Transformer"], marker='o', label="ST-Transformer (Ours, 13J)", color='dodgerblue', linewidth=2.5)
    axes[0].plot(df_noise["Intensity"], df_noise["GRU_33J"], marker='s', label="GRU Baseline (33J)", color='forestgreen', linewidth=2)
    axes[0].plot(df_noise["Intensity"], df_noise["GRU_13J"], marker='d', label="GRU Baseline (13J)", color='mediumpurple', linewidth=2)
    axes[0].plot(df_noise["Intensity"], df_noise["ST_GCN"], marker='^', label="ST-GCN Baseline (17J)", color='crimson', linewidth=2)
    axes[0].set_title("Robustness: Gaussian Noise Jitter", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Noise Std (sigma)", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_ylim(40, 102)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)

    # Plot 2: Random Joint Occlusion
    df_occ = df_robust[df_robust["Perturbation"] == "Random Joint Occlusion"]
    axes[1].plot(df_occ["Intensity"], df_occ["Transformer"], marker='o', label="ST-Transformer (Ours, 13J)", color='dodgerblue', linewidth=2.5)
    axes[1].plot(df_occ["Intensity"], df_occ["GRU_33J"], marker='s', label="GRU Baseline (33J)", color='forestgreen', linewidth=2)
    axes[1].plot(df_occ["Intensity"], df_occ["GRU_13J"], marker='d', label="GRU Baseline (13J)", color='mediumpurple', linewidth=2)
    axes[1].plot(df_occ["Intensity"], df_occ["ST_GCN"], marker='^', label="ST-GCN Baseline (17J)", color='crimson', linewidth=2)
    axes[1].set_title("Robustness: Random Joint Occlusion", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Number of Occluded Joints", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_ylim(40, 102)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "robustness_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness comparison plot to {plot_path}")

    # Plot 3: Specific Joint Occlusion Bar Chart
    plt.figure(figsize=(9, 5.5))
    df_spec = df_robust[df_robust["Perturbation"].isin(["Wrist Occlusion", "Knee Occlusion"])]
    
    # We pivot for bar plotting
    df_pivot = df_spec.melt(id_vars=["Perturbation"], value_vars=["Transformer", "GRU_33J", "GRU_13J", "ST_GCN"], var_name="Model", value_name="Accuracy")
    sns.barplot(x="Perturbation", y="Accuracy", hue="Model", data=df_pivot, palette=["dodgerblue", "forestgreen", "mediumpurple", "crimson"])
    plt.title("Robustness under Specific Key Joint Occlusions", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)")
    plt.ylim(50, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0:
            plt.gca().annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5),
                               ha='center', va='center', xytext=(0, 3), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    spec_plot_path = os.path.join(OUT_DIR, "robustness_specific_occlusion.png")
    plt.savefig(spec_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved specific occlusion plot to {spec_plot_path}")
    print("[ALL DONE] Robustness evaluation completed successfully.")

if __name__ == "__main__":
    main()
