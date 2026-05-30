"""
run_best_configs.py
===================
Run 2 final candidate configurations on all_zero dataset:
  - Config A: S=2, T=1 + No Norm + With Flip
  - Config B: S=2, T=2 + No Norm + With Flip

Train: original (non-val) + ALL _flip versions
Val:   original-only (same val videos as baseline)
Epochs: 80
"""

import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import EpisodePhaseDataset, build_phase_table, split_by_video
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

EPISODES_CSV = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR = "data_original/npy"
TARGET_COL   = "class_id"
NUM_CLASSES  = 5
EPOCHS       = 80
SEED         = 42
CLASS_NAMES  = ['Other', 'Squat', 'Hip-Hinge', 'Asymmetric', 'Upright']


def build_fair_flip_split():
    def norm_vid(v):
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    # Step1: 用原始 CSV 決定相同的 val_vids
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

    # Step2: 所有 _flip 版本全加進 train
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

    print(f"[Split] Val vids: {sorted(val_vids)}")
    print(f"[Split] Train={len(train_df)} (orig+flip), Val={len(val_df)} (orig only)")
    print("[Split] Train label dist:\n", train_df["label"].value_counts().sort_index())
    print("[Split] Val label dist:\n", val_df["label"].value_counts().sort_index())
    return train_df, val_df


def train_one(spatial, temporal, ckpt_dir, train_df, val_df, tag):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[TRAIN] {tag} | device={device}")
    print(f"{'='*60}")

    os.makedirs(ckpt_dir, exist_ok=True)

    # normalize=False (No Norm), use_aug=True (keep online aug)
    train_ds = EpisodePhaseDataset(train_df, skeleton_dir=SKELETON_DIR,
                                   out_len=64, normalize=False, use_aug=True, mode="train")
    val_ds   = EpisodePhaseDataset(val_df,   skeleton_dir=SKELETON_DIR,
                                   out_len=64, normalize=False, use_aug=False, mode="val")

    sampler      = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=spatial, num_temporal_layers=temporal,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    best_acc = 0.0
    best_cm  = None
    best_report = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any(): continue
            optimizer.zero_grad()
            logits = model(x)
            if torch.isnan(logits).any() or torch.isinf(logits).any(): continue
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader.dataset))
        val_acc, cm, report = evaluate(model, val_loader, device, NUM_CLASSES)
        is_best = val_acc > best_acc
        if is_best:
            best_acc    = val_acc
            best_cm     = cm.copy()
            best_report = report

        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "best_acc": best_acc},
            is_best=is_best, ckpt_dir=ckpt_dir, filename=f"epoch_{epoch}.pth"
        )
        print(f"Epoch {epoch}/{EPOCHS} Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")

    print(f"\n[DONE] {tag} → Best Val Acc = {best_acc*100:.2f}%")
    print("Best CM:\n", best_cm)
    print("Best Report:\n", best_report)
    return best_acc, best_cm


def norm_cm(cm):
    cm = cm.astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs * 100


def plot_side_by_side(results, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Best Config Comparison: No Norm + Offline Flip\n(all_zero dataset, same Val Set, 80 Epochs)",
        fontsize=14, fontweight='bold'
    )
    for ax, (label, (acc, cm)) in zip(axes, results.items()):
        sns.heatmap(norm_cm(cm), annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES[:cm.shape[0]],
                    yticklabels=CLASS_NAMES[:cm.shape[0]],
                    ax=ax, linewidths=0.5, linecolor='gray', vmin=0, vmax=100)
        ax.set_title(f"{label}\nVal Acc = {acc*100:.2f}%", fontsize=13, pad=10)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")


if __name__ == "__main__":
    print(">>> Building fair flip split...")
    train_df, val_df = build_fair_flip_split()

    results = {}

    # Config A: S=2, T=1
    acc_a, cm_a = train_one(
        spatial=2, temporal=1,
        ckpt_dir="checkpoints/best_S2T1_noNorm_flip",
        train_df=train_df, val_df=val_df,
        tag="S=2, T=1 | No Norm | With Flip"
    )
    results["S=2, T=1\n(No Norm + Flip)"] = (acc_a, cm_a)

    # Config B: S=2, T=2
    acc_b, cm_b = train_one(
        spatial=2, temporal=2,
        ckpt_dir="checkpoints/best_S2T2_noNorm_flip",
        train_df=train_df, val_df=val_df,
        tag="S=2, T=2 | No Norm | With Flip"
    )
    results["S=2, T=2\n(No Norm + Flip)"] = (acc_b, cm_b)

    # 輸出比較圖
    plot_side_by_side(results, "confusion_matrix_best_configs.png")

    print("\n\n" + "="*60)
    print("  FINAL COMPARISON SUMMARY")
    print("="*60)
    for name, (acc, _) in results.items():
        print(f"  {name.replace(chr(10), ' ')}: {acc*100:.2f}%")

    # 儲存 CSV
    pd.DataFrame([
        {"Config": k.replace('\n', ' '), "Best Val Acc (%)": f"{v[0]*100:.2f}"}
        for k, v in results.items()
    ]).to_csv("best_configs_results.csv", index=False)
    print("[SAVED] best_configs_results.csv")
    print("[ALL DONE]")
