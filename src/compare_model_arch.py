"""
compare_model_arch.py
比較三個模型的架構參數：
  1. ST-GCN 原生版本 (3 class，前人設定)
  2. ST-GCN 本研究版本 (5 class，自訓練)
  3. Robust Transformer (5 class，本研究)
"""

import os, sys, time, warnings
import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '../NTU/stgcn'))

from models_transformer import SpatialTemporalTransformer
from models_skeleton import GRUClassifier
from net.st_gcn import Model as STGCN

# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def count_params(model):
    total    = sum(p.numel() for p in model.parameters())
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_layers(model):
    """計算 Linear + Conv 層的數量（等效「可學習層」數）"""
    return sum(1 for m in model.modules()
               if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)))

def benchmark_latency(model, dummy_input, device, iterations=200, warmup=50):
    model.eval()
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end = time.time()
    avg_ms = (end - start) / iterations * 1000
    fps    = 1000.0 / avg_ms
    return avg_ms, fps

def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / 1024 / 1024

def describe_stgcn(num_class, layout, in_channels=3):
    """回傳 ST-GCN 的層次描述"""
    return {
        "spatial_block" : "ST-GCN Block × 9 (Graph Conv + BN + ReLU + Residual)",
        "temporal_block": "包含在每個 ST-GCN Block 內（1D Temporal Conv）",
        "output_layer"  : f"Global Average Pooling → FCN (256→{num_class})",
        "graph_layout"  : layout,
        "in_channels"   : in_channels,
        "temporal_window": "T=16 (前人設定，8 FPS)" if num_class == 3 else "T=64 (本研究，30 FPS)",
    }

def describe_transformer():
    return {
        "spatial_encoder" : "Spatial Transformer Block × 3 (Multi-Head Attention + FFN + LayerNorm)",
        "temporal_encoder": "Temporal Transformer Block × 4 (Multi-Head Attention + FFN + LayerNorm)",
        "attention_heads" : "8 heads, d_model=256",
        "output_layer"    : "Global Average Pooling → MLP (256→5)",
        "joint_embedding" : "Linear projection (3 → 256) × 13 joints",
        "positional_enc"  : "Spatial + Temporal Positional Encoding",
        "temporal_window" : "T=64 (本研究，30 FPS)",
    }

# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*65}")
    print(f"  模型架構比較報告 | Device: {device}")
    print(f"{'='*65}\n")

    results = []

    # ── 1. ST-GCN 原生 (3 class，COCO 17 joints，T=16) ──────────────
    print("[1/3] 建立 ST-GCN (原生 3-class)...")
    stgcn_orig = STGCN(
        in_channels=3, num_class=3,
        graph_args={"layout": "coco", "strategy": "spatial"},
        edge_importance_weighting=True
    ).to(device)
    dummy_orig = torch.randn(1, 3, 16, 17, 1)  # N,C,T,V,M
    total_p, train_p = count_params(stgcn_orig)
    n_layers         = count_layers(stgcn_orig)
    size_mb          = model_size_mb(stgcn_orig)
    avg_ms, fps      = benchmark_latency(stgcn_orig, dummy_orig, device)
    results.append({
        "名稱"         : "ST-GCN (原生, 前人)",
        "分類數"       : 3,
        "輸入關節數"   : "17 (COCO)",
        "時間視窗 T"   : 16,
        "總參數量"     : total_p,
        "可訓練參數"   : train_p,
        "可學習層數"   : n_layers,
        "模型大小 (MB)": f"{size_mb:.2f}",
        "推論延遲 (ms)": f"{avg_ms:.2f}",
        "極限 FPS"     : f"{fps:.1f}",
    })

    # ── 2. ST-GCN 本研究 (5 class，COCO 17 joints，T=64) ─────────────
    print("[2/3] 建立 ST-GCN (5-class，本研究設定)...")
    stgcn_5cls = STGCN(
        in_channels=3, num_class=5,
        graph_args={"layout": "coco", "strategy": "spatial"},
        edge_importance_weighting=True
    ).to(device)
    dummy_5cls = torch.randn(1, 3, 64, 17, 1)  # T=64
    total_p, train_p = count_params(stgcn_5cls)
    n_layers         = count_layers(stgcn_5cls)
    size_mb          = model_size_mb(stgcn_5cls)
    avg_ms, fps      = benchmark_latency(stgcn_5cls, dummy_5cls, device)
    results.append({
        "名稱"         : "ST-GCN (5-class, 本研究)",
        "分類數"       : 5,
        "輸入關節數"   : "17 (COCO)",
        "時間視窗 T"   : 64,
        "總參數量"     : total_p,
        "可訓練參數"   : train_p,
        "可學習層數"   : n_layers,
        "模型大小 (MB)": f"{size_mb:.2f}",
        "推論延遲 (ms)": f"{avg_ms:.2f}",
        "極限 FPS"     : f"{fps:.1f}",
    })

    # ── 3. Robust Transformer (5 class，13 joints，T=64) ──────────────
    print("[3/3] 建立 Robust Transformer (5-class，本研究)...")
    transformer = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=3, num_temporal_layers=4,
        num_classes=5, dropout=0.0
    ).to(device)
    dummy_tr = torch.randn(1, 64, 13, 3)  # N,T,V,C
    total_p, train_p = count_params(transformer)
    n_layers         = count_layers(transformer)
    size_mb          = model_size_mb(transformer)
    avg_ms, fps      = benchmark_latency(transformer, dummy_tr, device)
    results.append({
        "名稱"         : "Robust Transformer (本研究)",
        "分類數"       : 5,
        "輸入關節數"   : "13 (核心, 輕量化)",
        "時間視窗 T"   : 64,
        "總參數量"     : total_p,
        "可訓練參數"   : train_p,
        "可學習層數"   : n_layers,
        "模型大小 (MB)": f"{size_mb:.2f}",
        "推論延遲 (ms)": f"{avg_ms:.2f}",
        "極限 FPS"     : f"{fps:.1f}",
    })

    # ── 輸出比較表 ────────────────────────────────────────────────────
    keys = ["名稱", "分類數", "輸入關節數", "時間視窗 T",
            "總參數量", "可訓練參數", "可學習層數",
            "模型大小 (MB)", "推論延遲 (ms)", "極限 FPS"]

    print(f"\n{'='*65}")
    print("  架構比較總表 (CPU 實測)")
    print(f"{'='*65}")
    for key in keys:
        row = f"  {key:<18}: "
        for r in results:
            val = str(r[key])
            if key == "總參數量" or key == "可訓練參數":
                val = f"{int(r[key]):,}"
            row += f"{val:<28}"
        print(row)

    print(f"\n{'='*65}")
    print("  ST-GCN 架構說明")
    print(f"{'='*65}")
    desc = describe_stgcn(3, "COCO 17 joints")
    for k, v in desc.items():
        print(f"  {k:<20}: {v}")

    print(f"\n{'='*65}")
    print("  Robust Transformer 架構說明")
    print(f"{'='*65}")
    desc = describe_transformer()
    for k, v in desc.items():
        print(f"  {k:<20}: {v}")

if __name__ == "__main__":
    main()
