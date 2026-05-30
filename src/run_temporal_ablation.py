"""
run_temporal_ablation.py
========================
Temporal Ablation Experiments for Spatial-Temporal Transformer:
1. Sequence Length (T) Ablation:
   - T=32, T=64 (Baseline), T=128
   - Train and evaluate with S=2, T=2, No Norm, Flip
2. Frame Rate Stride Ablation:
   - Stride=1 (15 FPS - Baseline), Stride=2 (7.5 FPS), Stride=4 (3.75 FPS)
   - Zero-shot evaluation using the best T=64 model
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

from models_transformer import SpatialTemporalTransformer
from train_transformer import build_phase_table, split_by_video, normalize_skeleton, resample_seq
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

EPISODES_CSV = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR = "data_original/npy"
TARGET_COL   = "class_id"
NUM_CLASSES  = 5
EPOCHS       = 80
SEED         = 42
CLASS_NAMES  = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
OUT_DIR      = "thesis_materials"

class TemporalAblationDataset(Dataset):
    def __init__(self, df, skeleton_dir, out_len=64, normalize=False, use_aug=True, mode="train", stride_factor=1):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.normalize = normalize
        self.use_aug = use_aug
        self.mode = mode
        self.stride_factor = stride_factor
        self.labels = self.df["label"].astype(int).tolist()
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

        seg = full_seq[s:e + 1]

        if self.normalize:
            seg = normalize_skeleton(seg)

        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply stride downsampling to simulate lower frame rate
        if self.stride_factor > 1:
            seg = seg[::self.stride_factor]
            if len(seg) == 0:
                seg = full_seq[s:e + 1]

        seg = resample_seq(seg, self.out_len)
        
        CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        seg = seg[:, CORE_JOINTS, :]

        if self.mode == "train" and self.use_aug:
            # 1. Random Rotation
            if random.random() < 0.5:
                angle = random.uniform(-15.0, 15.0) * np.pi / 180.0
                c = np.cos(angle)
                s_a = np.sin(angle)
                old_x = seg[..., 0].copy()
                old_z = seg[..., 2].copy()
                seg[..., 0] = old_x * c - old_z * s_a
                seg[..., 2] = old_x * s_a + old_z * c
            
            # 2. Gaussian Jitter
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.005, size=seg.shape)
                seg += noise

        return torch.from_numpy(seg).float(), label

def build_fair_flip_split():
    def norm_vid(v):
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    df_orig = pd.read_csv(EPISODES_CSV)
    df_orig = df_orig.dropna(subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df_orig = df_orig[df_orig[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()
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

    val_df     = phase_orig[phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)
    train_orig = phase_orig[~phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)

    flip_csv = "analysis/episodes_from_json_all_zero_flip.csv"
    df_flip = pd.read_csv(flip_csv)
    df_flip = df_flip.dropna(subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df_flip = df_flip[df_flip[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()
    phase_flip = pd.DataFrame({
        "video_id":         df_flip["video_id"].apply(norm_vid),
        "start_frame_skel": df_flip["lift_start_frame"].astype(int) // 2,
        "end_frame_skel":   df_flip["lift_end_frame"].astype(int) // 2,
        "label":            df_flip[TARGET_COL].astype(int)
    })
    flip_only = phase_flip[phase_flip["video_id"].str.endswith("_flip")].reset_index(drop=True)
    train_df  = pd.concat([train_orig, flip_only], ignore_index=True)
    return train_df, val_df

def train_one_config(train_df, val_df, out_len, ckpt_dir, tag):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] {tag} | T={out_len} | device={device}")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_ds = TemporalAblationDataset(train_df, SKELETON_DIR, out_len=out_len, normalize=False, use_aug=True, mode="train")
    val_ds   = TemporalAblationDataset(val_df,   SKELETON_DIR, out_len=out_len, normalize=False, use_aug=False, mode="val")
    
    sampler = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        scheduler.step()
        val_acc, cm, _ = evaluate(model, val_loader, device, NUM_CLASSES)
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            
        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_acc": best_acc},
            is_best=is_best, ckpt_dir=ckpt_dir, filename=f"epoch_{epoch}.pth"
        )
    print(f"[DONE] {tag} | Best Val Acc = {best_acc*100:.2f}%")
    return best_acc

def eval_stride(val_df, stride_factor, ckpt_path):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_ds = TemporalAblationDataset(val_df, SKELETON_DIR, out_len=64, normalize=False, use_aug=False, mode="val", stride_factor=stride_factor)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    val_acc, cm, _ = evaluate(model, val_loader, device, NUM_CLASSES)
    print(f"[STRIDE EVAL] Stride Factor={stride_factor} | Val Acc = {val_acc*100:.2f}%")
    return val_acc, cm

def main():
    print(">>> Building fair flip split...")
    train_df, val_df = build_fair_flip_split()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Sequence Length Ablation (T=32, T=64, T=128)
    # -------------------------------------------------------------
    print("\n>>> Running Sequence Length Ablation...")
    results_len = {}
    
    # T=32
    acc_32 = train_one_config(
        train_df, val_df, out_len=32,
        ckpt_dir="checkpoints/ablation_temporal_T32",
        tag="T=32 Ablation"
    )
    results_len["T=32"] = acc_32
    
    # T=64 (Load existing or train if missing)
    best_T64_ckpt = "checkpoints/best_S2T2_noNorm_flip/best.pth"
    if os.path.exists(best_T64_ckpt):
        print(f"Loading existing T=64 checkpoint: {best_T64_ckpt}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(best_T64_ckpt, map_location=device)
        acc_64 = checkpoint["best_acc"]
        print(f"T=64 (Baseline) loaded accuracy: {acc_64*100:.2f}%")
    else:
        acc_64 = train_one_config(
            train_df, val_df, out_len=64,
            ckpt_dir="checkpoints/best_S2T2_noNorm_flip",
            tag="T=64 Baseline"
        )
    results_len["T=64 (Baseline)"] = acc_64
    
    # T=128
    acc_128 = train_one_config(
        train_df, val_df, out_len=128,
        ckpt_dir="checkpoints/ablation_temporal_T128",
        tag="T=128 Ablation"
    )
    results_len["T=128"] = acc_128
    
    # Save Sequence Length Ablation Results
    df_len = pd.DataFrame([
        {"Sequence Length (T)": k, "Val Accuracy (%)": f"{v*100:.2f}"}
        for k, v in results_len.items()
    ])
    df_len.to_csv(os.path.join(OUT_DIR, "ablation_temporal_len.csv"), index=False)
    print(f"Saved: {os.path.join(OUT_DIR, 'ablation_temporal_len.csv')}")
    
    # -------------------------------------------------------------
    # 2. Stride Ablation (Stride=1 (15 FPS), Stride=2 (7.5 FPS), Stride=4 (3.75 FPS))
    # -------------------------------------------------------------
    print("\n>>> Running Frame Rate Stride Ablation...")
    results_stride = {}
    
    # Best T=64 checkpoint
    ckpt_path = "checkpoints/best_S2T2_noNorm_flip/best.pth"
    
    # Stride=1 -> 15 FPS (Baseline)
    acc_s1, _ = eval_stride(val_df, stride_factor=1, ckpt_path=ckpt_path)
    results_stride["Stride=1 (15 FPS)"] = acc_s1
    
    # Stride=2 -> 7.5 FPS
    acc_s2, _ = eval_stride(val_df, stride_factor=2, ckpt_path=ckpt_path)
    results_stride["Stride=2 (7.5 FPS)"] = acc_s2
    
    # Stride=4 -> 3.75 FPS
    acc_s4, _ = eval_stride(val_df, stride_factor=4, ckpt_path=ckpt_path)
    results_stride["Stride=4 (3.75 FPS)"] = acc_s4
    
    # Save Stride Ablation Results
    df_stride = pd.DataFrame([
        {"Frame Rate Settings": k, "Val Accuracy (%)": f"{v*100:.2f}"}
        for k, v in results_stride.items()
    ])
    df_stride.to_csv(os.path.join(OUT_DIR, "ablation_temporal_stride.csv"), index=False)
    print(f"Saved: {os.path.join(OUT_DIR, 'ablation_temporal_stride.csv')}")
    
    # -------------------------------------------------------------
    # 3. Plot Comparison Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Sequence Length
    sns.barplot(x="Sequence Length (T)", y=df_len["Val Accuracy (%)"].astype(float), data=df_len, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Ablation: Sequence Length (T)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[0].set_ylim(85, 100)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
        
    # Plot Stride
    sns.lineplot(x="Frame Rate Settings", y=df_stride["Val Accuracy (%)"].astype(float), data=df_stride, marker='o', ax=axes[1], color='darkorange', linewidth=2)
    axes[1].set_title("Ablation: Frame Rate Stride", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[1].set_ylim(80, 100)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    for x_idx, y_val in enumerate(df_stride["Val Accuracy (%)"].astype(float)):
        axes[1].annotate(f"{y_val:.2f}%", (x_idx, y_val + 0.5), ha='center', fontsize=9, fontweight='semibold')
        
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "temporal_ablation_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    print("[ALL DONE] Temporal ablation completed successfully.")

if __name__ == "__main__":
    main()
