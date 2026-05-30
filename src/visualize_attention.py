# -*- coding: utf-8 -*-
"""
visualize_attention.py
======================
從最佳模型 (best_S2T2_noNorm_flip) 提取空間注意力權重，
生成三張論文用圖：

圖 1: per_class_spatial_attention.png
      各類別平均空間注意力熱圖 (5 個子圖)

圖 2: class3_vs_class4_attention_diff.png
      Class 3 (屈膝) vs Class 4 (直膝) 差異熱圖
      → 論文最有說服力的圖，說明模型區分危險姿態的依據

圖 3: attention_top_pairs_per_class.png
      每個類別前 5 名最強注意力關節對比較圖

執行方式（從專案根目錄）：
    python src/visualize_attention.py
"""

import os, sys, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from models_transformer import SpatialTemporalTransformer
from train_transformer import EpisodePhaseDataset, build_phase_table, split_by_video

# ── 設定 ──────────────────────────────────────────────────────────────────────
CKPT_PATH    = "checkpoints/best_S2T2_noNorm_flip/best.pth"
EPISODES_CSV = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR = "data_original/npy"
TARGET_COL   = "class_id"
NUM_CLASSES  = 5
SEED         = 42
OUT_DIR      = "thesis_materials"

CLASS_NAMES = {
    0: "Class 0\nOther",
    1: "Class 1\nSquat",
    2: "Class 2\nHip-Hinge",
    3: "Class 3\nAsymmetric",
    4: "Class 4\nUpright",
}
CLASS_COLORS = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# 13 核心關節名稱（依照 CORE_JOINTS 順序）
# CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
JOINT_NAMES = [
    "Nose",       # 0
    "L.Shoulder", # 11
    "R.Shoulder", # 12
    "L.Elbow",    # 13
    "R.Elbow",    # 14
    "L.Wrist",    # 15
    "R.Wrist",    # 16
    "L.Hip",      # 23
    "R.Hip",      # 24
    "L.Knee",     # 25
    "R.Knee",     # 26
    "L.Ankle",    # 27
    "R.Ankle",    # 28
]
NUM_JOINTS = len(JOINT_NAMES)  # 13


# ── 資料載入（與 run_best_configs.py 相同的 val split）────────────────────────
def load_val_dataset():
    def norm_vid(v):
        s = str(v)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

    df = pd.read_csv(EPISODES_CSV)
    df = df.dropna(subset=["video_id", "lift_start_frame", "lift_end_frame", TARGET_COL]).copy()
    df = df[df[TARGET_COL].astype(int).between(0, NUM_CLASSES - 1)].copy()
    phase_df = pd.DataFrame({
        "video_id":         df["video_id"].apply(norm_vid),
        "start_frame_skel": df["lift_start_frame"].astype(int) // 2,
        "end_frame_skel":   df["lift_end_frame"].astype(int) // 2,
        "label":            df[TARGET_COL].astype(int)
    })
    video_ids = sorted(phase_df["video_id"].unique())
    random.Random(SEED).shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * 0.2))
    val_vids = set(video_ids[:n_val])
    val_df = phase_df[phase_df["video_id"].isin(val_vids)].reset_index(drop=True)
    print(f"[INFO] Val videos: {sorted(val_vids)}")
    print(f"[INFO] Val samples: {len(val_df)}")
    print(val_df["label"].value_counts().sort_index())
    return EpisodePhaseDataset(val_df, skeleton_dir=SKELETON_DIR,
                               out_len=64, normalize=False, use_aug=False, mode="val")


# ── 模型載入 ──────────────────────────────────────────────────────────────────
def load_model(device):
    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=2, num_temporal_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3
    ).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")
    return model


# ── 手動 Spatial Encoder Forward（繞過 PyTorch 硬編的 need_weights=False）────
@torch.no_grad()
def run_spatial_layer_with_attn(layer, x):
    """
    手動重現 TransformerEncoderLayer.forward()，
    但以 need_weights=True 呼叫 self_attn，取得注意力矩陣。

    x   : (B*T, J, d_model)
    回傳: x_out (B*T, J, d_model),  attn (J, J)
    """
    # PyTorch TransformerEncoderLayer (norm_first=False, 預設):
    #   x = x + dropout(self_attn(x, x, x)[0])
    #   x = norm1(x)
    #   x = x + dropout(linear2(activation(linear1(x))))
    #   x = norm2(x)
    attn_out, attn_weights = layer.self_attn(
        x, x, x,
        need_weights=True,
        average_attn_weights=True   # 平均所有 head → (B*T, J, J)
    )
    # attn_weights: (B*T, J, J)
    attn_mean = attn_weights.mean(dim=0).cpu().numpy()  # (J, J)

    x = x + layer.dropout1(attn_out)
    x = layer.norm1(x)
    x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = x + layer.dropout2(x2)
    x = layer.norm2(x)
    return x, attn_mean


# ── 主程式：收集各類別注意力 ──────────────────────────────────────────────────
@torch.no_grad()
def collect_attention_by_class(model, dataset, device):
    """
    跑完整個 val set，按類別分組收集平均注意力矩陣。
    回傳: class_attn dict {class_id: (num_layers, J, J) ndarray}
    """
    num_layers = len(model.spatial_transformer.layers)
    # class_layers_list[class_id][layer_idx] = list of (J,J)
    class_layers_list = {c: [[] for _ in range(num_layers)] for c in range(NUM_CLASSES)}

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        label = int(y.item())
        B, T, J, C = 1, x.shape[0], x.shape[1], x.shape[2]
        x_in = x.unsqueeze(0).to(device)  # (1, T, J, C)

        # ── Spatial 前處理（與 model.forward 一致）──
        h = model.joint_embed(x_in)            # (1, T, J, d)
        h = h + model.spatial_pos_embed        # broadcast
        h = h.view(B * T, J, -1)              # (B*T, J, d)

        # ── 逐層跑 spatial encoder，每層都擷取注意力 ──
        for li, layer in enumerate(model.spatial_transformer.layers):
            h, attn_map = run_spatial_layer_with_attn(layer, h)
            class_layers_list[label][li].append(attn_map)

    # 對每個類別、每一層做平均
    result = {}
    for cls_id in range(NUM_CLASSES):
        per_layer = []
        for li in range(num_layers):
            arr_list = class_layers_list[cls_id][li]
            if arr_list:
                per_layer.append(np.stack(arr_list, axis=0).mean(axis=0))
            else:
                per_layer.append(np.zeros((NUM_JOINTS, NUM_JOINTS)))
        result[cls_id] = np.stack(per_layer, axis=0)  # (num_layers, J, J)
        print(f"  Class {cls_id}: {len(class_layers_list[cls_id][0])} samples collected")

    return result


# ── 圖 1：各類別平均空間注意力熱圖 ────────────────────────────────────────────
def plot_per_class_heatmap(class_attn, layer_idx=0, save_path=None):
    """
    layer_idx: 要視覺化的 spatial encoder 層（0=第一層，通常最有可解釋性）
    """
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(22, 4.5))
    fig.suptitle(
        f"Spatial Self-Attention Weights per Action Class\n"
        f"(Spatial Encoder Layer {layer_idx+1}, Averaged over Validation Set)",
        fontsize=13, fontweight='bold', y=1.02
    )

    vmin, vmax = 0.0, 0.0
    for cls_id in range(NUM_CLASSES):
        attn = class_attn[cls_id][layer_idx]
        vmax = max(vmax, attn.max())

    for cls_id in range(NUM_CLASSES):
        ax = axes[cls_id]
        attn = class_attn[cls_id][layer_idx]
        sns.heatmap(
            attn,
            ax=ax,
            xticklabels=JOINT_NAMES,
            yticklabels=JOINT_NAMES,
            cmap='YlOrRd',
            vmin=vmin, vmax=vmax,
            square=True,
            cbar=(cls_id == NUM_CLASSES - 1),
            annot=False,
            linewidths=0.3,
            linecolor='#eeeeee'
        )
        ax.set_title(CLASS_NAMES[cls_id], fontsize=11, fontweight='bold',
                     color=CLASS_COLORS[cls_id], pad=6)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)
        ax.set_xlabel("Key Joint", fontsize=8)
        ax.set_ylabel("Query Joint" if cls_id == 0 else "", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.close()


# ── 圖 2：Class 3 vs Class 4 差異熱圖（論文最重要的圖）──────────────────────
def plot_class3_vs_class4_diff(class_attn, layer_idx=0, save_path=None):
    """
    差值圖：Attn(Class4) - Attn(Class3)
    正值（紅）→ Class 4（直膝彎腰）特有的高注意力關節對
    負值（藍）→ Class 3（屈膝）特有的高注意力關節對
    """
    attn3 = class_attn[3][layer_idx]  # (J, J)
    attn4 = class_attn[4][layer_idx]  # (J, J)
    diff  = attn4 - attn3             # 正值=Class4更強，負值=Class3更強

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Class 3 (Asymmetric/Safe Lift) vs Class 4 (Upright/Risky Lift)\n"
        "Spatial Attention Comparison — The Key Discriminative Signal",
        fontsize=13, fontweight='bold'
    )

    # 左：Class 3
    sns.heatmap(attn3, ax=axes[0],
                xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES,
                cmap='YlOrRd', square=True, cbar=True, annot=False,
                linewidths=0.3, linecolor='#eeeeee')
    axes[0].set_title("Class 3 — Asymmetric (Safe)\nAttn avg over val set",
                       fontsize=11, color=CLASS_COLORS[3])
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    axes[0].tick_params(axis='y', rotation=0, labelsize=7)

    # 中：Class 4
    sns.heatmap(attn4, ax=axes[1],
                xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES,
                cmap='YlOrRd', square=True, cbar=True, annot=False,
                linewidths=0.3, linecolor='#eeeeee')
    axes[1].set_title("Class 4 — Upright (Risky)\nAttn avg over val set",
                       fontsize=11, color=CLASS_COLORS[4])
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)
    axes[1].tick_params(axis='y', rotation=0, labelsize=7)

    # 右：差值
    abs_max = np.abs(diff).max()
    sns.heatmap(diff, ax=axes[2],
                xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES,
                cmap='RdBu_r', center=0,
                vmin=-abs_max, vmax=abs_max,
                square=True, cbar=True, annot=False,
                linewidths=0.3, linecolor='#eeeeee')
    axes[2].set_title("Difference (Class4 − Class3)\nRed=Class4 dominant, Blue=Class3 dominant",
                       fontsize=11, color='#2c3e50')
    axes[2].tick_params(axis='x', rotation=45, labelsize=7)
    axes[2].tick_params(axis='y', rotation=0, labelsize=7)

    # 標出差異最大的前 3 個關節對
    flat_idx = np.argsort(np.abs(diff).flatten())[::-1][:3]
    for rank, fi in enumerate(flat_idx):
        qi, ki = divmod(fi, NUM_JOINTS)
        val = diff[qi, ki]
        color = '#c0392b' if val > 0 else '#2980b9'
        axes[2].add_patch(
            mpatches.Rectangle((ki, qi), 1, 1,
                                fill=False, edgecolor=color, lw=2.5)
        )
        print(f"  Top-{rank+1} diff pair: {JOINT_NAMES[qi]} ← {JOINT_NAMES[ki]}  "
              f"diff={val:+.4f}  ({'Class4 > Class3' if val > 0 else 'Class3 > Class4'})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.close()


# ── 圖 3：各類別 Top-5 最強注意力關節對 ──────────────────────────────────────
def plot_top_pairs_per_class(class_attn, layer_idx=0, top_k=5, save_path=None):
    """
    橫向條形圖：每個類別注意力最強的前 5 個 (Query, Key) 關節對
    """
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(22, 5))
    fig.suptitle(
        f"Top-{top_k} Strongest Spatial Attention Pairs per Class\n"
        f"(Spatial Encoder Layer {layer_idx+1})",
        fontsize=13, fontweight='bold'
    )

    for cls_id in range(NUM_CLASSES):
        ax = axes[cls_id]
        attn = class_attn[cls_id][layer_idx].copy()
        # 去除對角線（自注意力）
        np.fill_diagonal(attn, 0)

        flat = attn.flatten()
        top_idx = np.argsort(flat)[::-1][:top_k]
        scores = flat[top_idx]
        labels = []
        for fi in top_idx:
            qi, ki = divmod(fi, NUM_JOINTS)
            labels.append(f"{JOINT_NAMES[qi]}\n← {JOINT_NAMES[ki]}")

        colors = [CLASS_COLORS[cls_id]] * top_k
        bars = ax.barh(range(top_k)[::-1], scores, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_yticks(range(top_k)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Avg Attn Weight", fontsize=9)
        ax.set_title(CLASS_NAMES[cls_id], fontsize=11, fontweight='bold',
                     color=CLASS_COLORS[cls_id])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=8)

        # 在 bar 右側標數值
        for bar, score in zip(bars, scores[::-1]):
            ax.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va='center', fontsize=7.5, color='#2c3e50')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.close()


# ── 額外：列印各類別的 top attention pair 供論文文字引用 ──────────────────────
def print_summary(class_attn, layer_idx=0):
    print("\n" + "=" * 60)
    print(f"  Attention Pattern Summary — Layer {layer_idx + 1}")
    print("=" * 60)
    for cls_id in range(NUM_CLASSES):
        attn = class_attn[cls_id][layer_idx].copy()
        np.fill_diagonal(attn, 0)
        flat = attn.flatten()
        top3 = np.argsort(flat)[::-1][:3]
        print(f"\n{CLASS_NAMES[cls_id].replace(chr(10), ' ')}:")
        for fi in top3:
            qi, ki = divmod(fi, NUM_JOINTS)
            print(f"  {JOINT_NAMES[qi]:12s} ← {JOINT_NAMES[ki]:12s}  {flat[fi]:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[STEP 1] Loading validation dataset...")
    val_ds = load_val_dataset()

    print("[STEP 2] Loading model...")
    model = load_model(device)

    print("[STEP 3] Collecting attention weights (this takes ~1-2 min)...")
    class_attn = collect_attention_by_class(model, val_ds, device)

    print("[STEP 4] Generating figures...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 圖 1
    plot_per_class_heatmap(
        class_attn, layer_idx=0,
        save_path=os.path.join(OUT_DIR, "attention_per_class_layer1.png")
    )
    plot_per_class_heatmap(
        class_attn, layer_idx=1,
        save_path=os.path.join(OUT_DIR, "attention_per_class_layer2.png")
    )

    # 圖 2（論文最重要的圖）
    print("\n[Class 3 vs Class 4 差異分析]")
    plot_class3_vs_class4_diff(
        class_attn, layer_idx=0,
        save_path=os.path.join(OUT_DIR, "attention_class3_vs_class4_diff.png")
    )

    # 圖 3
    plot_top_pairs_per_class(
        class_attn, layer_idx=0,
        save_path=os.path.join(OUT_DIR, "attention_top_pairs_per_class.png")
    )

    # 文字摘要
    print_summary(class_attn, layer_idx=0)

    print("\n[DONE] All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
