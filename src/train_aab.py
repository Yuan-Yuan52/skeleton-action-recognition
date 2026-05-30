# -*- coding: utf-8 -*-
"""
train_aab.py
============
訓練並比較兩組模型（消融設計）：
  - Ours-AAB  : ST-Transformer S=2,T=2 + Anatomical Attention Bias (use_aab=True)
  - Ours-Base : ST-Transformer S=2,T=2，無 AAB                     (use_aab=False)

資料設定與 run_best_configs.py 完全相同（同一 val split、No Norm、Fair Flip），
確保兩組實驗的差異只來自 AAB 本身。

執行（從專案根目錄）：
    python src/train_aab.py
"""

import os, sys, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer_aab import SpatialTemporalTransformerAAB
from train_transformer import EpisodePhaseDataset
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

# ── 設定（與 run_best_configs.py 完全一致）────────────────────────────────────
EPISODES_CSV  = "analysis/episodes_from_json_all_zero.csv"
FLIP_CSV      = "analysis/episodes_from_json_all_zero_flip.csv"
SKELETON_DIR  = "data_original/npy"
TARGET_COL    = "class_id"
NUM_CLASSES   = 5
EPOCHS        = 80
SEED          = 42
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 5e-4

CLASS_NAMES = ['Other', 'Squat', 'Hip-Hinge', 'Asymmetric', 'Upright']
OUT_DIR     = "thesis_materials"
CKPT_BASE   = "checkpoints"


# ── 資料分割（與 run_best_configs.py 相同的 Fair Flip 設計）──────────────────
def build_fair_flip_split():
    def norm_vid(v):
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    df_orig = pd.read_csv(EPISODES_CSV)
    df_orig = df_orig.dropna(
        subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df_orig = df_orig[df_orig[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()

    phase_orig = pd.DataFrame({
        "video_id":         df_orig["video_id"].apply(norm_vid),
        "start_frame_skel": df_orig["lift_start_frame"].astype(int) // 2,
        "end_frame_skel":   df_orig["lift_end_frame"].astype(int) // 2,
        "label":            df_orig[TARGET_COL].astype(int),
    })

    video_ids = sorted(phase_orig["video_id"].unique())
    random.Random(SEED).shuffle(video_ids)
    n_val    = max(1, int(len(video_ids) * 0.2))
    val_vids = set(video_ids[:n_val])

    val_df     = phase_orig[phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)
    train_orig = phase_orig[~phase_orig["video_id"].isin(val_vids)].reset_index(drop=True)

    df_flip = pd.read_csv(FLIP_CSV)
    df_flip = df_flip.dropna(
        subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df_flip = df_flip[df_flip[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()
    phase_flip = pd.DataFrame({
        "video_id":         df_flip["video_id"].apply(norm_vid),
        "start_frame_skel": df_flip["lift_start_frame"].astype(int) // 2,
        "end_frame_skel":   df_flip["lift_end_frame"].astype(int) // 2,
        "label":            df_flip[TARGET_COL].astype(int),
    })
    flip_only = phase_flip[
        phase_flip["video_id"].str.endswith("_flip")].reset_index(drop=True)
    train_df = pd.concat([train_orig, flip_only], ignore_index=True)

    print(f"[Split] Val vids  : {sorted(val_vids)}")
    print(f"[Split] Train={len(train_df)} (orig+flip), Val={len(val_df)} (orig only)")
    return train_df, val_df


# ── Silhouette Score（特徵空間聚類評估）──────────────────────────────────────
@torch.no_grad()
def compute_silhouette(model, val_loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in val_loader:
        x = x.to(device)
        # 取 Global Pooling 前的特徵（temporal transformer 輸出的 mean）
        B, T, J, C = x.shape
        h = model.joint_embed(x) + model.spatial_pos_embed
        h = h.view(B * T, J, -1)
        attn_mask = model.aab.get(B * T) if model.use_aab else None
        for layer in model.spatial_layers:
            h = layer(h, attn_mask=attn_mask)
        h = h.reshape(B, T, J * model.joint_embed.out_features)
        h = model.spatial_proj(h)
        h = h + model.temporal_pos_embed[:, :T, :]
        h = model.temporal_transformer(h)
        feat = h.mean(dim=1).cpu().numpy()   # (B, d_model)
        feats.append(feat)
        labels.append(y.numpy())
    feats  = np.concatenate(feats,  axis=0)
    labels = np.concatenate(labels, axis=0)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(feats, labels, sample_size=min(len(feats), 500)))


# ── 混淆矩陣圖 ────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, title, save_path):
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ── 單組訓練 ──────────────────────────────────────────────────────────────────
def train_one(use_aab, train_df, val_df):
    tag      = "AAB" if use_aab else "NoAAB"
    ckpt_dir = os.path.join(CKPT_BASE, f"aab_{tag.lower()}")
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[TRAIN] use_aab={use_aab}  tag={tag}  device={device}")
    print(f"{'='*60}")

    train_ds = EpisodePhaseDataset(train_df, skeleton_dir=SKELETON_DIR,
                                   out_len=64, normalize=False,
                                   use_aug=True, mode="train")
    val_ds   = EpisodePhaseDataset(val_df,   skeleton_dir=SKELETON_DIR,
                                   out_len=64, normalize=False,
                                   use_aug=False, mode="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=make_weighted_sampler(train_ds.labels),
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model = SpatialTemporalTransformerAAB(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3,
        use_aab=use_aab
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,} ({n_params/1e6:.3f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    best_acc   = 0.0
    best_cm    = None
    best_report = ""
    log_rows   = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            is_best=is_best, ckpt_dir=ckpt_dir,
            filename=f"epoch_{epoch}.pth"
        )

        log_rows.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_acc:.4f}")

    # 計算最佳 checkpoint 的 Silhouette Score
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt.get("state_dict", best_ckpt))
    sil = compute_silhouette(model, val_loader, device)
    print(f"\n[{tag}] Best Val Acc = {best_acc:.4f}  Silhouette = {sil:.4f}")
    print(f"[{tag}] Classification Report:\n{best_report}")

    # 儲存混淆矩陣
    title = (f"ST-Transformer + AAB (use_aab={use_aab})\n"
             f"Val Acc={best_acc:.4f}  Silhouette={sil:.4f}")
    cm_path = os.path.join(OUT_DIR, f"confusion_matrix_aab_{tag.lower()}.png")
    save_confusion_matrix(best_cm, title, cm_path)

    return {"tag": tag, "best_acc": best_acc, "silhouette": sil,
            "cm": best_cm, "log": log_rows}


# ── 對比圖：AAB vs NoAAB 學習曲線 ────────────────────────────────────────────
def plot_learning_curves(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'AAB': '#e74c3c', 'NoAAB': '#3498db'}
    for r in results:
        epochs = [row["epoch"] for row in r["log"]]
        accs   = [row["val_acc"] for row in r["log"]]
        label  = f"{r['tag']}  (best={r['best_acc']:.4f}, sil={r['silhouette']:.4f})"
        ax.plot(epochs, accs, color=colors[r['tag']], linewidth=2, label=label)
        ax.axhline(r['best_acc'], color=colors[r['tag']],
                   linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("AAB vs No-AAB: Validation Accuracy Curve", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0.90, 1.01)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "aab_ablation_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


# ── 結果 CSV ──────────────────────────────────────────────────────────────────
def save_results_csv(results):
    rows = [{"Config": r["tag"],
             "Val_Acc": f"{r['best_acc']:.4f}",
             "Silhouette": f"{r['silhouette']:.4f}"}
            for r in results]
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "aab_ablation_results.csv")
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")
    print(df.to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("[STEP 1] Building data split...")
    train_df, val_df = build_fair_flip_split()

    results = []
    # 先跑 AAB，再跑 NoAAB（消融對照）
    for use_aab in [True, False]:
        r = train_one(use_aab, train_df, val_df)
        results.append(r)

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    for r in results:
        print(f"  {r['tag']:8s}: Acc={r['best_acc']:.4f}  Silhouette={r['silhouette']:.4f}")

    plot_learning_curves(results)
    save_results_csv(results)


if __name__ == "__main__":
    main()
