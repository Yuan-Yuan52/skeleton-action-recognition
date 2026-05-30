"""
run_all_experiments.py
======================
一鍵執行所有消融實驗（正確資料集版本）：
  資料集: analysis/episodes_from_json_all_zero.csv
  骨架:   data_original/npy
  Epochs: 80
  Num Classes: 5 (0=Other, 1=Squat, 2=Hip-Hinge, 3=Asymmetric, 4=Upright)

實驗一：層數消融 (Layer Depth Ablation)
  1+1, 1+2, 2+1, 2+2, 3+4

實驗二：方法消融 (Method Ablation)
  Full / No-Norm / No-Aug / No-Both  (固定 2+2 架構)

實驗三：公平翻轉消融 (Fair Flip Augmentation)
  Baseline vs With-Flip  (固定 2+2 架構, 相同 Val Set)
"""

import os, sys, subprocess, random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import (
    EpisodePhaseDataset, build_phase_table, split_by_video,
    normalize_skeleton, resample_seq
)
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

# ========== 全域設定 ==========
EPISODES_CSV   = "analysis/episodes_from_json_all_zero.csv"
FLIP_CSV       = "analysis/episodes_from_json_all_zero_flip.csv"
SKELETON_DIR   = "data_original/npy"
TARGET_COL     = "class_id"
NUM_CLASSES    = 5
EPOCHS         = 80
SEED           = 42
CLASS_NAMES    = ['Other', 'Squat', 'Hip-Hinge', 'Asymmetric', 'Upright']


# ========== 通用訓練函式 ==========
def train_one(
    spatial, temporal, ckpt_dir,
    episodes_csv=EPISODES_CSV, skeleton_dir=SKELETON_DIR,
    normalize=True, use_aug=True,
    train_df_override=None, val_df_override=None,
    tag=""
):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[TRAIN] {tag} | S={spatial} T={temporal} | Epochs={EPOCHS}")
    print(f"{'='*60}")

    os.makedirs(ckpt_dir, exist_ok=True)

    # --- 建立 train/val split ---
    if train_df_override is not None and val_df_override is not None:
        train_df = train_df_override
        val_df   = val_df_override
    else:
        phase_df = build_phase_table(episodes_csv, "lift", TARGET_COL, sample_stride=2)
        train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)

    train_ds = EpisodePhaseDataset(train_df, skeleton_dir=skeleton_dir,
                                   out_len=64, normalize=normalize,
                                   use_aug=use_aug, mode="train")
    val_ds   = EpisodePhaseDataset(val_df,   skeleton_dir=skeleton_dir,
                                   out_len=64, normalize=normalize,
                                   use_aug=False, mode="val")

    sampler = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=spatial, num_temporal_layers=temporal,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.0)

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

    print(f"[DONE] {tag} → Best Val Acc = {best_acc*100:.2f}%")
    print("Best CM:\n", best_cm)
    return best_acc, best_cm


# ========== 繪圖工具 ==========
def norm_cm(cm):
    cm = cm.astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs * 100


def plot_cm_grid(cms_dict, title, outpath, cols=3):
    """cms_dict: {'label': (acc, cm)}"""
    items = list(cms_dict.items())
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if rows == 1: axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, (label, (acc, cm)) in enumerate(items):
        ax = axes_flat[idx]
        labels_to_show = CLASS_NAMES[:cm.shape[0]]
        sns.heatmap(norm_cm(cm), annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels_to_show, yticklabels=labels_to_show,
                    ax=ax, linewidths=0.5, linecolor='gray', vmin=0, vmax=100)
        ax.set_title(f"{label}\nVal Acc={acc*100:.2f}%", fontsize=12, pad=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')

    for idx in range(len(items), len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")


# ========== 實驗一：層數消融 ==========
def run_depth_ablation():
    configs = [(1,1), (1,2), (2,1), (2,2), (3,4)]
    results = {}
    for s, t in configs:
        tag  = f"S{s}T{t}"
        ckpt = f"checkpoints/ablation_zero/{tag}"
        acc, cm = train_one(s, t, ckpt, tag=tag)
        results[f"S={s} T={t}"] = (acc, cm)

    # 儲存結果 CSV
    rows = [{"Config": k, "Spatial": k.split()[0][2:], "Temporal": k.split()[1][2:],
             "Best Val Acc (%)": f"{v[0]*100:.2f}"} for k, v in results.items()]
    pd.DataFrame(rows).to_csv("ablation_depth_zero_results.csv", index=False)
    print("[SAVED] ablation_depth_zero_results.csv")

    plot_cm_grid(results, "Layer Depth Ablation (episodes_from_json_all_zero)",
                 "confusion_matrix_depth_ablation_zero.png", cols=3)
    return results


# ========== 實驗二：方法消融 ==========
def run_method_ablation():
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)

    configs = [
        ("Full (Norm+Aug)",     True,  True),
        ("No Norm",             False, True),
        ("No Aug",              True,  False),
        ("No Norm & No Aug",    False, False),
    ]
    results = {}
    for name, norm, aug in configs:
        tag  = name.replace(" ", "_").replace("&", "and")
        ckpt = f"checkpoints/method_zero/{tag}"
        acc, cm = train_one(2, 2, ckpt,
                            normalize=norm, use_aug=aug,
                            train_df_override=train_df,
                            val_df_override=val_df,
                            tag=name)
        results[name] = (acc, cm)

    rows = [{"Method": k, "Best Val Acc (%)": f"{v[0]*100:.2f}"} for k, v in results.items()]
    pd.DataFrame(rows).to_csv("ablation_methods_zero_results.csv", index=False)
    print("[SAVED] ablation_methods_zero_results.csv")

    plot_cm_grid(results, "Method Ablation (2+2 Architecture, all_zero dataset)",
                 "confusion_matrix_method_ablation_zero.png", cols=2)
    return results


# ========== 實驗三：公平翻轉消融 ==========
def build_fair_flip_split(seed=SEED):
    """Train: orig (non-val) + ALL _flip; Val: orig only (same as baseline)."""
    import re

    def norm_vid(v):
        if pd.isna(v): return str(v)
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    # Step1: 用原始 CSV 決定 val_vids（與 baseline 相同）
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
    random.Random(seed).shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * 0.2))
    val_vids = set(video_ids[:n_val])

    val_df       = phase_orig[phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)
    train_orig   = phase_orig[~phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)

    # Step2: 加入全部 _flip
    df_flip = pd.read_csv(FLIP_CSV)
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

    print(f"[Flip] Val vids: {sorted(val_vids)}")
    print(f"[Flip] Train={len(train_df)} (orig+flip), Val={len(val_df)} (orig only)")
    return train_df, val_df, val_df  # val_df used for both


def run_flip_ablation():
    # ---- Baseline ----
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)

    acc_base, cm_base = train_one(
        2, 2, "checkpoints/flip_zero_baseline",
        train_df_override=train_df, val_df_override=val_df,
        tag="Baseline (No Flip)"
    )

    # ---- Fair Flip ----
    flip_train_df, flip_val_df, _ = build_fair_flip_split()
    acc_flip, cm_flip = train_one(
        2, 2, "checkpoints/flip_zero_fair",
        train_df_override=flip_train_df, val_df_override=flip_val_df,
        tag="With Offline Flip (Fair)"
    )

    results = {
        f"No Flip (Baseline)": (acc_base, cm_base),
        f"With Offline Flip":  (acc_flip, cm_flip),
    }
    plot_cm_grid(results, "Flip Augmentation Ablation (all_zero, same Val Set)",
                 "confusion_matrix_flip_ablation_zero.png", cols=2)

    pd.DataFrame([
        {"Method": "No Flip (Baseline)", "Train N": len(train_df), "Val N": len(val_df),  "Best Val Acc (%)": f"{acc_base*100:.2f}"},
        {"Method": "With Offline Flip",  "Train N": len(flip_train_df), "Val N": len(flip_val_df), "Best Val Acc (%)": f"{acc_flip*100:.2f}"},
    ]).to_csv("ablation_flip_zero_results.csv", index=False)
    print("[SAVED] ablation_flip_zero_results.csv")
    return results


# ========== 主程式 ==========
if __name__ == "__main__":
    print(">>> Running ALL ablation experiments with all_zero dataset <<<")

    print("\n\n" + "="*60)
    print("  EXPERIMENT 1: Layer Depth Ablation")
    print("="*60)
    depth_results = run_depth_ablation()

    print("\n\n" + "="*60)
    print("  EXPERIMENT 2: Method Ablation")
    print("="*60)
    method_results = run_method_ablation()

    print("\n\n" + "="*60)
    print("  EXPERIMENT 3: Fair Flip Augmentation")
    print("="*60)
    flip_results = run_flip_ablation()

    # ======== 最終結果彙整 ========
    print("\n\n" + "="*60)
    print("  ALL RESULTS SUMMARY")
    print("="*60)
    print("\n[Layer Depth]")
    for k, (a, _) in depth_results.items():
        print(f"  {k}: {a*100:.2f}%")
    print("\n[Method]")
    for k, (a, _) in method_results.items():
        print(f"  {k}: {a*100:.2f}%")
    print("\n[Flip]")
    for k, (a, _) in flip_results.items():
        print(f"  {k}: {a*100:.2f}%")
    print("\n[ALL DONE]")
