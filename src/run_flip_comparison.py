"""
run_flip_comparison.py
======================
æ¯è??æ??¡é¢ç·å·¦?³ç¿»è½è??æ´å¢ãå?æ¨¡å?æºç¢º?è???Class ?é?è¡¨ç¾?å½±?¿ã?
?ºå?ä½¿ç¨ï¼?  - ?ä½³æ¶æ§ï?Spatial=2, Temporal=2
  - è¨ç·´è¨­å?ï¼Norm + Aug, 40 epochs

è·å©?æ¨¡?ï?
  1. Baselineï¼ä½¿?¨å?å§?episodes CSV
  2. With Flipï¼ä½¿?¨å«ç¿»è??æ¬??episodes CSV

?çµç¢?ºï?
  - ablation_flip_results.csvï¼æ?ç¢ºç?æ¯è?è¡?  - confusion_matrix_before_flip.pngï¼æª? ç¿»è½ç?æ··æ??©é£
  - confusion_matrix_after_flip.pngï¼å??¥ç¿»è½å??æ··æ·ç©??"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path so we can import training modules
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import (
    EpisodePhaseDataset,
    build_phase_table,
    split_by_video,
    normalize_skeleton,
    resample_seq
)
from utils import evaluate, seed_everything
from torch.utils.data import DataLoader


CLASS_NAMES = ["Squat", "Hip-Hinge", "Asymmetric", "Upright", "Other"]


def plot_confusion_matrix(cm, class_names, title, save_path):
    """ç¹ªè£½æ··æ??©é£ä¸¦å²å­?""
    # Normalize to percentage
    cm_normalized = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    cm_normalized = cm_normalized / row_sums * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
    
    # Left: raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, linecolor='gray')
    axes[0].set_title('Count', fontsize=13)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Right: normalized %
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, linecolor='gray',
                vmin=0, vmax=100)
    axes[1].set_title('Recall % (row-wise)', fontsize=13)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix: {save_path}")


def train_and_evaluate(config_name, episodes_csv, skeleton_dir, ckpt_dir, num_classes=5):
    """?·è?è¨ç·´ä¸¦å??³æ?ä½³æ¨¡?ç?æºç¢º?è?æ··æ??©é£"""
    
    cmd = [
        "python", "src/train_transformer.py",
        "--episodes_csv", episodes_csv,
        "--skeleton_dir", skeleton_dir,
        "--num_spatial_layers", "2",
        "--num_temporal_layers", "2",
        "--epochs", "40",
        "--ckpt_dir", ckpt_dir,
        "--wandb_name", f"flip_ablation_{config_name}"
    ]
    
    print(f"\n{'='*55}")
    print(f"[TRAINING] {config_name}")
    print(f"{'='*55}")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    best_acc = 0.0
    for line in process.stdout:
        print(line, end='')
        if "BestAcc=" in line:
            try:
                best_acc_str = line.split("BestAcc=")[1].strip().split()[0]
                best_acc = float(best_acc_str)
            except Exception:
                pass
    
    process.wait()
    print(f"\n[DONE] {config_name} ??Best Val Acc = {best_acc*100:.2f}%")
    return best_acc


def get_confusion_matrix(episodes_csv, skeleton_dir, ckpt_dir, num_classes=5, phase="lift", target="start_class4"):
    """å¾å·²è¨ç·´?æ¨¡?è??¥æ?ä½³æ??ï?è¨ç?æ··æ??©é£"""
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build val dataset (same val split as training)
    phase_df = build_phase_table(episodes_csv, phase, target, sample_stride=2)
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
        num_classes=num_classes, dropout=0.3
    )
    model.to(device)
    
    # Load best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"{phase}_{target}", "model_best.pth")
    if not os.path.exists(ckpt_path):
        # Try alternative naming
        ckpt_path_alt = os.path.join(ckpt_dir, "lift_start_class4", "model_best.pth")
        if os.path.exists(ckpt_path_alt):
            ckpt_path = ckpt_path_alt
        else:
            print(f"[WARN] No checkpoint found at {ckpt_path}, skipping CM.")
            return None
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    val_acc, cm, report = evaluate(model, val_loader, device, num_classes)
    print(f"[INFO] Loaded checkpoint ??Val Acc = {val_acc*100:.2f}%")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)
    return cm, val_acc


def main():
    skeleton_dir = "data_original/npy"
    
    configs = [
        {
            "name": "baseline",
            "display_name": "Without Flip (Baseline)",
            "episodes_csv": "analysis/episodes_from_json_all_more.csv",
            "ckpt_dir": "checkpoints/flip_ablation_baseline"
        },
        {
            "name": "with_flip",
            "display_name": "With Offline Flip Augmentation",
            "episodes_csv": "analysis/episodes_from_json_all_more_flip.csv",
            "ckpt_dir": "checkpoints/flip_ablation_with_flip"
        }
    ]
    
    results = []
    confusion_matrices = {}
    
    for cfg in configs:
        best_acc = train_and_evaluate(
            config_name=cfg["name"],
            episodes_csv=cfg["episodes_csv"],
            skeleton_dir=skeleton_dir,
            ckpt_dir=cfg["ckpt_dir"]
        )
        
        # Get confusion matrix from best model
        result = get_confusion_matrix(
            episodes_csv=cfg["episodes_csv"],
            skeleton_dir=skeleton_dir,
            ckpt_dir=cfg["ckpt_dir"]
        )
        
        if result is not None:
            cm, loaded_acc = result
            confusion_matrices[cfg["name"]] = cm
        
        results.append({
            "Method": cfg["display_name"],
            "Best Val Acc (%)": f"{best_acc*100:.2f}"
        })
        
        # Save intermediate
        pd.DataFrame(results).to_csv("ablation_flip_results.csv", index=False)
    
    # ===== ç¹ªè£½æ··æ??©é£æ¯è???=====
    for cfg in configs:
        name = cfg["name"]
        if name in confusion_matrices:
            save_path = f"confusion_matrix_{name}.png"
            plot_confusion_matrix(
                confusion_matrices[name],
                CLASS_NAMES,
                title=cfg["display_name"],
                save_path=save_path
            )
    
    # ===== ?çµç???=====
    print("\n" + "="*55)
    print("FINAL COMPARISON: Offline Flip Augmentation")
    print("="*55)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_flip_results.csv", index=False)
    print("\n[DONE] Results saved to ablation_flip_results.csv")
    print("[DONE] Confusion matrices saved as PNG files.")


if __name__ == "__main__":
    main()
