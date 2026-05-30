"""
generate_flip_cm.py
===================
å¾å·²è¨ç·´?å©?æ¨¡?ï?baseline / with_flipï¼å?è¼å¥ best.pthï¼??¶å??å¥?¨å??ªç? val set ä¸è?ç®æ··æ·ç©???è¼¸åºæ¯è??ã?"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import build_phase_table, split_by_video, EpisodePhaseDataset
from utils import evaluate, seed_everything
from torch.utils.data import DataLoader

# ---- ä½ ç? 4 ?å§¿??class ?ç¨± ----
CLASS_NAMES = ["Squat", "Hip-Hinge", "Asymmetric", "Upright", "Other"]
NUM_CLASSES  = 5


def load_model_and_evaluate(ckpt_path, episodes_csv, skeleton_dir, device):
    seed_everything(42)

    phase_df = build_phase_table(episodes_csv, "lift", "start_class4", sample_stride=2)
    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=42)

    val_ds = EpisodePhaseDataset(
        val_df,
        skeleton_dir=skeleton_dir,
        out_len=64,
        normalize=True,
        use_aug=False,
        mode="val"
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Loaded: {ckpt_path}  (saved at epoch {ckpt.get('epoch','?')})")

    val_acc, cm, report = evaluate(model, val_loader, device, NUM_CLASSES)
    print(f"  Val Acc = {val_acc*100:.2f}%")
    print("  Report:\n", report)
    print("  CM:\n", cm)
    return val_acc, cm


def plot_side_by_side(cm_before, cm_after, acc_before, acc_after, class_names, save_path):
    """Before / After æ··æ??©é£ä¸¦æ?æ¯è??ï?Recall % ?æ¬ï¼?""

    def norm_cm(cm):
        cm = cm.astype(float)
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return cm / rs * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix Comparison: Offline Flip Augmentation", fontsize=16, fontweight='bold')

    titles = [
        f"Without Flip (Baseline)\nVal Acc = {acc_before*100:.2f}%",
        f"With Offline Flip Augmentation\nVal Acc = {acc_after*100:.2f}%",
    ]
    cms = [norm_cm(cm_before), norm_cm(cm_after)]

    for ax, cm_norm, title in zip(axes, cms, titles):
        sns.heatmap(
            cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, linecolor='gray',
            vmin=0, vmax=100
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skeleton_dir = "data_original/npy"

    configs = [
        {
            "label": "baseline",
            "ckpt":  "checkpoints/flip_ablation_baseline/lift_start_class4/best.pth",
            "csv":   "analysis/episodes_from_json_all_more.csv",
        },
        {
            "label": "with_flip",
            "ckpt":  "checkpoints/flip_ablation_with_flip/lift_start_class4/best.pth",
            "csv":   "analysis/episodes_from_json_all_more_flip.csv",
        },
    ]

    results = {}
    for cfg in configs:
        print(f"\n[{cfg['label'].upper()}]")
        acc, cm = load_model_and_evaluate(cfg["ckpt"], cfg["csv"], skeleton_dir, device)
        results[cfg["label"]] = {"acc": acc, "cm": cm}

    # ä¸¦æ?æ··æ??©é£
    plot_side_by_side(
        cm_before  = results["baseline"]["cm"],
        cm_after   = results["with_flip"]["cm"],
        acc_before = results["baseline"]["acc"],
        acc_after  = results["with_flip"]["acc"],
        class_names= CLASS_NAMES,
        save_path  = "confusion_matrix_flip_comparison.png"
    )

    print("\n====== FINAL RESULT ======")
    print(f"  Baseline (No Flip) : {results['baseline']['acc']*100:.2f}%")
    print(f"  With Flip          : {results['with_flip']['acc']*100:.2f}%")


if __name__ == "__main__":
    main()
