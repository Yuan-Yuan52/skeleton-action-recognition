"""
run_flip_fair.py
================
?¬å¹³?ç¿»è½æ´å¢å¯¦é©ï?
  - Train: ?å?å½±ç?ï¼é? valï¼?+ ???_flip ?æ¬
  - Val:   ??Baseline å®å¨ä¸æ¨???å? 6 ?¯å½±?ï?93 ç­ï?

?æ¨£ç¢ºä??©åå¯¦é©ç?é©è??å??¨ç¸?ï??¯ä»¥ç´ç²¹æ¯è?
?ç¿»è½è??æ?æ²æ?å¹«å©æ¨¡å??¨ç¸?æ¸¬è©¦é?ä¸è¡¨?¾æ´å¥½ãã?"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import pandas as pd
from models_transformer import SpatialTemporalTransformer
from train_transformer import (
    EpisodePhaseDataset,
    build_phase_table,
    normalize_skeleton,
    resample_seq
)
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

def build_fair_split(episodes_csv_flip, episodes_csv_orig, val_ratio=0.2, seed=42, sample_stride=2):
    """
    ?¨å?å§?CSV æ±ºå? train/val splitï¼?    ?¶å???flip ?æ¬?¨é¨å¡é?train set??    """
    # ---- Step 1: ?¨å?å§?CSV ??train/val splitï¼è? Baseline ?¸å?ï¼?---
    df_orig = pd.read_csv(episodes_csv_orig)
    if "use_for_train" in df_orig.columns:
        df_orig = df_orig[df_orig["use_for_train"] == 1].copy()

    # ??build_phase_table ?é?è¼?    phase = "lift"
    target = "start_class4"
    start_col, end_col = "lift_start_frame", "lift_end_frame"

    df_orig = df_orig.dropna(subset=["video_id", start_col, end_col, target]).copy()
    df_orig = df_orig[df_orig[target].astype(int).between(0, 4)].copy()

    def norm_vid(v):
        if pd.isna(v): return v
        if isinstance(v, (int, np.integer)): return str(int(v))
        if isinstance(v, (float, np.floating)): return str(int(v)) if float(v).is_integer() else str(v)
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    phase_df_orig = pd.DataFrame({
        "video_id": df_orig["video_id"].apply(norm_vid),
        "start_frame_skel": df_orig[start_col].astype(int) // sample_stride,
        "end_frame_skel":   df_orig[end_col].astype(int) // sample_stride,
        "label": df_orig[target].astype(int)
    })

    # ??Baseline ?¸å???split seed
    video_ids = sorted(phase_df_orig["video_id"].unique())
    random.Random(seed).shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * val_ratio))
    val_vids = set(video_ids[:n_val])

    print(f"[INFO] Val videos (same as Baseline): {sorted(val_vids)}")
    print(f"[INFO] Train videos (original): {sorted(set(video_ids) - val_vids)}")

    val_df   = phase_df_orig[phase_df_orig["video_id"].isin(val_vids)].reset_index(drop=True)
    train_orig_df = phase_df_orig[~phase_df_orig["video_id"].isin(val_vids)].reset_index(drop=True)

    # ---- Step 2: ? å¥?¨é¨ _flip ?æ¬??Train ----
    df_flip = pd.read_csv(episodes_csv_flip)
    if "use_for_train" in df_flip.columns:
        df_flip = df_flip[df_flip["use_for_train"] == 1].copy()

    df_flip = df_flip.dropna(subset=["video_id", start_col, end_col, target]).copy()
    df_flip = df_flip[df_flip[target].astype(int).between(0, 4)].copy()

    phase_df_flip = pd.DataFrame({
        "video_id": df_flip["video_id"].apply(norm_vid),
        "start_frame_skel": df_flip[start_col].astype(int) // sample_stride,
        "end_frame_skel":   df_flip[end_col].astype(int) // sample_stride,
        "label": df_flip[target].astype(int)
    })

    # ?ªä???_flip ?æ¬?è??å?
    flip_only_df = phase_df_flip[phase_df_flip["video_id"].str.endswith("_flip")].reset_index(drop=True)
    print(f"[INFO] Flip episodes added to train: {len(flip_only_df)}")

    # ?ä½µ train
    train_df = pd.concat([train_orig_df, flip_only_df], ignore_index=True)

    print(f"[INFO] Final train size: {len(train_df)}, Val size: {len(val_df)}")
    print("[INFO] Val label distribution:")
    print(val_df["label"].value_counts().sort_index())
    print("[INFO] Train label distribution:")
    print(train_df["label"].value_counts().sort_index())

    return train_df, val_df


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    epochs = 80
    num_classes = 5
    skeleton_dir = "data_original/npy"
    ckpt_dir = "checkpoints/flip_fair_80ep/lift_start_class4"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- å»ºç??¬å¹³??Train / Val Split ----
    train_df, val_df = build_fair_split(
        episodes_csv_flip = "analysis/episodes_from_json_all_more_flip.csv",
        episodes_csv_orig = "analysis/episodes_from_json_all_more.csv",
        val_ratio = 0.2,
        seed = 42
    )

    train_ds = EpisodePhaseDataset(train_df, skeleton_dir=skeleton_dir, out_len=64, normalize=True, use_aug=True, mode="train")
    val_ds   = EpisodePhaseDataset(val_df,   skeleton_dir=skeleton_dir, out_len=64, normalize=True, use_aug=False, mode="val")

    sampler = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=num_classes, dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
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
        val_acc, cm, report = evaluate(model, val_loader, device, num_classes)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "best_acc": best_acc},
            is_best=is_best, ckpt_dir=ckpt_dir, filename=f"epoch_{epoch}.pth"
        )

        print(f"Epoch {epoch}/{epochs} Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)

    print(f"\n[DONE] Fair Flip 80ep ??Best Val Acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
