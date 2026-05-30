"""
train_comparison_models.py
==========================
訓練 GRU 與 ST-GCN 比較模型（5 類，all_zero 資料集）：
  - 資料集: analysis/episodes_from_json_all_zero.csv
  - 骨架:   data_original/npy/
  - Target: class_id (class 0 ~ class 4)
  - Epochs: 80
  - Val Split: Video ['13', '19', '2', '24', '35', '4'] (與 Transformer 一致)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Add NTU/stgcn path for STGCN
sys.path.append(os.path.abspath(os.path.join(src_dir, '../NTU/stgcn')))
from net.st_gcn import Model as STGCN
from models_skeleton import GRUClassifier
from train_transformer import (
    build_phase_table, split_by_video, normalize_skeleton, resample_seq
)

class EpisodePhaseDataset(Dataset):
    """
    EpisodePhaseDataset that returns the full 33 joints.
    """
    def __init__(self, df, skeleton_dir, out_len=64, normalize=True, use_aug=True, mode="train"):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.normalize = normalize
        self.use_aug = use_aug
        self.mode = mode
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

        seq = np.load(npy_path).astype(np.float32)  # (T, 33, 3)
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
        if T <= 0:
            raise ValueError(f"Empty skeleton sequence: video_id={video_id}, s={s}, e={e}, T={T}")

        s = max(0, min(s, T - 1))
        e = max(0, min(e, T - 1))
        if e < s:
            e = s

        if (e - s + 1) <= 0:
            raise ValueError(f"Invalid segment after clamp: video_id={video_id}, s={s}, e={e}, T={T}")

        seg = full_seq[s:e + 1]  # (Te, 33, 3)

        if self.normalize:
            seg = normalize_skeleton(seg)

        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        seg = resample_seq(seg, self.out_len)  # (out_len, 33, 3)

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
                noise = np.random.normal(loc=0.0, scale=0.015, size=seg.shape).astype(np.float32)
                seg += noise

        x = torch.from_numpy(seg).float()  # (T, 33, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

from utils import seed_everything, save_checkpoint, make_weighted_sampler

EPISODES_CSV   = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR   = "data_original/npy"
TARGET_COL     = "class_id"
NUM_CLASSES    = 5
EPOCHS         = 80
SEED           = 42
CLASS_NAMES    = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
OUT_DIR        = "thesis_materials"


# ========== Focal Loss for GRU ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ========== Evaluation Helper ==========
def evaluate_model(model, loader, device, model_type):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if model_type == "gru":
                # Concat velocity features
                velocity = torch.zeros_like(x)
                velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
                x_in = torch.cat([x, velocity], dim=-1)
                logits = model(x_in)
            elif model_type == "stgcn":
                # Select 17 COCO joints and shape to (N, C, T, V, M)
                coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_in = x[:, :, coco_mapping, :]
                x_in = x_in.permute(0, 3, 1, 2).unsqueeze(-1)
                logits = model(x_in)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return acc, cm, all_preds, all_labels


# ========== Plotting Helpers ==========
def plot_confusion_matrix(labels, preds, title, outpath):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=100, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j + 0.5, i + 0.75, f"(n={cm[i,j]})",
                        ha='center', va='center', fontsize=7, color='gray')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")


def extract_features_and_plot_tsne(model, loader, device, model_type, best_acc, outpath):
    model.eval()
    features_list = []
    labels_list = []
    
    def hook(module, input):
        if model_type == "stgcn":
            feat = input[0] # (N, 256, T, V)
            feat = feat.mean(dim=(2, 3)) # (N, 256)
            features_list.append(feat.detach().cpu().numpy())
        else:
            features_list.append(input[0].detach().cpu().numpy())
            
    if model_type == "stgcn":
        handle = model.fcn.register_forward_pre_hook(hook)
    else:
        handle = model.fc[0].register_forward_pre_hook(hook)
        
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if model_type == "gru":
                velocity = torch.zeros_like(x)
                velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
                x_in = torch.cat([x, velocity], dim=-1)
                _ = model(x_in)
            elif model_type == "stgcn":
                coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_in = x[:, :, coco_mapping, :]
                x_in = x_in.permute(0, 3, 1, 2).unsqueeze(-1)
                _ = model(x_in)
            labels_list.append(y.numpy())
            
    handle.remove()
    feats = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"[t-SNE] Running for: {model_type.upper()}")
    perp = min(30, len(feats) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, max_iter=1000)
    emb = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("Set1", n_colors=NUM_CLASSES)
    for i in range(NUM_CLASSES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[palette[i]], label=CLASS_NAMES[i],
                   alpha=0.7, s=40, edgecolors='w', linewidths=0.3)
    ax.set_title(f"t-SNE: {model_type.upper()} Model\nVal Acc={best_acc*100:.2f}%", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(title="Action Class", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")


# ========== Main Training Process ==========
def train_model(model_type):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n==================== Training {model_type.upper()} Model ====================")
    
    # 1. Split data
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)
    
    # Both use normalize=True as baseline settings for GRU and ST-GCN
    train_ds = EpisodePhaseDataset(train_df, skeleton_dir=SKELETON_DIR, out_len=64, normalize=True, use_aug=True, mode="train")
    val_ds   = EpisodePhaseDataset(val_df,   skeleton_dir=SKELETON_DIR, out_len=64, normalize=True, use_aug=False, mode="val")
    
    sampler = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # 2. Build model
    if model_type == "gru":
        model = GRUClassifier(num_joints=33, in_channels=6, num_classes=NUM_CLASSES).to(device)
        criterion = FocalLoss(gamma=2.0)
        ckpt_dir = "checkpoints/phase_newway_class4_gru/lift_class_id"
    elif model_type == "stgcn":
        model = STGCN(in_channels=3, num_class=NUM_CLASSES, graph_args={"layout": "coco", "strategy": "spatial"}, edge_importance_weighting=True).to(device)
        criterion = nn.CrossEntropyLoss()
        ckpt_dir = "checkpoints/stgcn_v2/lift_class_id"
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if model_type == "gru":
                velocity = torch.zeros_like(x)
                velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
                x_combined = torch.cat([x, velocity], dim=-1)
            elif model_type == "stgcn":
                coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_combined = x[:, :, coco_mapping, :]
                x_combined = x_combined.permute(0, 3, 1, 2).unsqueeze(-1)
                
            optimizer.zero_grad()
            logits = model(x_combined)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        val_acc, cm, _, _ = evaluate_model(model, val_loader, device, model_type)
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            
        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "best_acc": best_acc},
            is_best=is_best, ckpt_dir=ckpt_dir, filename=f"epoch_{epoch}.pth"
        )
        print(f"Epoch {epoch}/{EPOCHS} Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")
        
    print(f"\n[DONE] {model_type.upper()} Best Val Acc = {best_acc*100:.2f}%")
    
    # 4. Generate Figures from the best model
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    
    _, _, preds, labels = evaluate_model(model, val_loader, device, model_type)
    
    # Save Confusion Matrix
    cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{model_type}_zero.png")
    plot_confusion_matrix(labels, preds, f"Confusion Matrix: {model_type.upper()} Baseline\nVal Acc={best_acc*100:.2f}%", cm_path)
    
    # Save t-SNE Plot
    tsne_path = os.path.join(OUT_DIR, f"tsne_{model_type}_zero.png")
    extract_features_and_plot_tsne(model, val_loader, device, model_type, best_acc, tsne_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Train both models
    train_model("gru")
    train_model("stgcn")
    print("\n[ALL TRAINING AND FIGURES COMPLETED]")


if __name__ == "__main__":
    main()
