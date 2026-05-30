"""
generate_new_figures.py
=======================
用正確資料集重新生成所有 t-SNE + CM 圖表：
  - Input CSV:  analysis/episodes_from_json_all_zero.csv（無 flip 驗證集）
  - Input NPY:  data_original/npy/
  - Target:     class_id
  - Best Model: checkpoints/best_S2T2_noNorm_flip/best.pth (S=2,T=2, No Norm)
"""

import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import build_phase_table, split_by_video, EpisodePhaseDataset
from utils import seed_everything

EPISODES_CSV  = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR  = "data_original/npy"
TARGET_COL    = "class_id"
NUM_CLASSES   = 5
SEED          = 42
CLASS_NAMES   = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
BEST_CKPT     = "checkpoints/best_S2T2_noNorm_flip/best.pth"
OUT_DIR       = "thesis_materials"


def get_val_loader():
    """Build val DataLoader using the original (non-flip) val set."""
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)
    val_ds = EpisodePhaseDataset(
        val_df, skeleton_dir=SKELETON_DIR, out_len=64,
        normalize=False,   # No Norm (matches best model training)
        use_aug=False, mode="val"
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"[INFO] Val set: {len(val_ds)} samples")
    return val_loader


def build_model(spatial=2, temporal=2):
    return SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=spatial, num_temporal_layers=temporal,
        num_classes=NUM_CLASSES, dropout=0.3
    )


def extract_features_and_preds(model, loader, device):
    """Forward hook on fc[0] to capture pre-classification features."""
    features_list = []
    def hook(module, input):
        features_list.append(input[0].detach().cpu().numpy())
    handle = model.fc[0].register_forward_pre_hook(hook)

    labels_list = []
    preds_list = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.append(preds)
            labels_list.append(y.cpu().numpy())

    handle.remove()
    features = np.concatenate(features_list, axis=0)
    labels   = np.concatenate(labels_list, axis=0)
    preds    = np.concatenate(preds_list, axis=0)
    return features, labels, preds


def plot_tsne(features, labels, title, outpath):
    """Run t-SNE and plot scatter."""
    print(f"[t-SNE] Running for: {title}")
    perp = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, max_iter=1000)
    emb  = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("Set1", n_colors=NUM_CLASSES)
    for i in range(NUM_CLASSES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[palette[i]], label=CLASS_NAMES[i],
                   alpha=0.7, s=40, edgecolors='w', linewidths=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(title="Action Class", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")


def plot_confusion_matrix(labels, preds, title, outpath):
    """Plot recall-normalized confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    # Normalize by row (recall)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=100, ax=ax)

    # Add raw counts in smaller text
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


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_loader = get_val_loader()

    # ========================================================
    # 1. Trained Best Model (S=2, T=2, No Norm, With Flip)
    # ========================================================
    print("\n=== Trained Best Model ===")
    model = build_model(2, 2).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    best_acc = ckpt.get("best_acc", "N/A")
    print(f"Loaded: {BEST_CKPT}, best_acc={best_acc}")

    feats, labels, preds = extract_features_and_preds(model, val_loader, device)

    # t-SNE
    plot_tsne(
        feats, labels,
        title=f"t-SNE: Best Model (S=2,T=2 | No Norm | Flip)\nVal Acc={best_acc*100:.2f}%",
        outpath=os.path.join(OUT_DIR, "tsne_best_model_zero.png")
    )

    # Confusion Matrix
    plot_confusion_matrix(
        labels, preds,
        title=f"Confusion Matrix: Best Model (S=2,T=2 | No Norm | Flip)\nVal Acc={best_acc*100:.2f}%",
        outpath=os.path.join(OUT_DIR, "confusion_matrix_best_model_zero.png")
    )

    # Print classification report
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    print("Classification Report:\n", report)

    # ========================================================
    # 2. Untrained Model (random weights, same architecture)
    # ========================================================
    print("\n=== Untrained Model (Random Weights) ===")
    model_untrained = build_model(2, 2).to(device)  # random init
    feats_u, labels_u, preds_u = extract_features_and_preds(model_untrained, val_loader, device)

    plot_tsne(
        feats_u, labels_u,
        title="t-SNE: Untrained Model (S=2,T=2 | Random Weights)",
        outpath=os.path.join(OUT_DIR, "tsne_untrained_zero.png")
    )

    # ========================================================
    # 3. Side-by-side t-SNE comparison
    # ========================================================
    print("\n=== Side-by-side t-SNE ===")
    perp = min(30, len(feats) - 1)
    emb_trained   = TSNE(n_components=2, perplexity=perp, random_state=SEED).fit_transform(feats)
    emb_untrained = TSNE(n_components=2, perplexity=perp, random_state=SEED).fit_transform(feats_u)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("t-SNE Feature Visualization: Untrained vs. Trained Best Model\n"
                 "(Input: episodes_from_json_all_zero.csv | data_original/npy)",
                 fontsize=14, fontweight='bold')
    palette = sns.color_palette("Set1", n_colors=NUM_CLASSES)

    for ax, emb, title_sub in zip(axes,
        [emb_untrained, emb_trained],
        ["Untrained (Random Weights)", f"Trained Best (Val Acc={best_acc*100:.2f}%)"]):
        for i in range(NUM_CLASSES):
            mask = labels == i
            if mask.sum() == 0: continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[palette[i]], label=CLASS_NAMES[i],
                       alpha=0.7, s=35, edgecolors='w', linewidths=0.3)
        ax.set_title(title_sub, fontsize=13, pad=8)
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(title="Class", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, "tsne_comparison_zero.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {outpath}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
