# -*- coding: utf-8 -*-
"""
models_transformer_aab.py
=========================
Spatio-Temporal Transformer with Anatomical Attention Bias (AAB).

核心創新：
  - 標準 ST-Transformer：所有關節對在空間注意力中完全平等對待
  - ST-GCN：固定物理拓撲鄰接矩陣，無法學習非相鄰關聯
  - AAB（本研究）：以人體骨架圖距離初始化注意力偏置，保留解剖先驗
                  同時讓所有權重完全可學習，突破 GCN 固定拓撲束縛

13 核心關節索引對應（CORE_JOINTS = [0,11,12,13,14,15,16,23,24,25,26,27,28]）：
  0: Nose        1: L.Shoulder  2: R.Shoulder
  3: L.Elbow     4: R.Elbow     5: L.Wrist     6: R.Wrist
  7: L.Hip       8: R.Hip       9: L.Knee     10: R.Knee
 11: L.Ankle    12: R.Ankle
"""

import torch
import torch.nn as nn
import numpy as np

# ── 骨架圖邊定義（解剖相鄰關係，13 關節重新映射後的索引）──────────────────────
CORE_EDGES = [
    (0, 1), (0, 2),     # Nose → L/R Shoulder
    (1, 2),             # L.Shoulder ↔ R.Shoulder
    (1, 3), (3, 5),     # L.Shoulder → L.Elbow → L.Wrist
    (2, 4), (4, 6),     # R.Shoulder → R.Elbow → R.Wrist
    (1, 7), (2, 8),     # Shoulder → Hip（軀幹左右側）
    (7, 8),             # L.Hip ↔ R.Hip
    (7, 9),  (9, 11),   # L.Hip → L.Knee → L.Ankle
    (8, 10), (10, 12),  # R.Hip → R.Knee → R.Ankle
]


def build_graph_distance(num_joints=13, edges=CORE_EDGES):
    """BFS Floyd-Warshall：計算所有關節對的骨架圖最短距離。"""
    dist = np.full((num_joints, num_joints), float(num_joints), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for i, j in edges:
        dist[i, j] = 1.0
        dist[j, i] = 1.0
    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


# ── Anatomical Attention Bias 模組 ────────────────────────────────────────────
class AnatomicalAttentionBias(nn.Module):
    """
    可學習的解剖注意力偏置。

    初始化策略：
      distance=1 (直接相鄰)   → bias ≈ +0.5   (鼓勵關注)
      distance=2              → bias ≈ 0
      distance≥3 (遠端關節)   → bias ≈ negative (初始壓制，但可學習)

    偏置以 attn_mask 形式注入 MultiheadAttention：
      attention = softmax( QK^T/√dk + anatomical_bias )
    """
    def __init__(self, num_joints=13, num_heads=8):
        super().__init__()
        dist = build_graph_distance(num_joints)
        d_max = dist.max()

        # 線性映射：distance=0 → +1, distance=max → -1
        init_bias = 1.0 - 2.0 * dist / d_max   # 範圍 [-1, +1]

        # 每個 head 獨立一份，初始值相同，訓練後各自分化
        init_tensor = torch.tensor(init_bias, dtype=torch.float32)
        init_tensor = init_tensor.unsqueeze(0).expand(num_heads, -1, -1).contiguous()

        # 乘以小係數：初始影響力小，讓模型先學姿態再微調偏置
        self.bias = nn.Parameter(init_tensor * 0.1)
        self.num_heads = num_heads

    def get(self, batch_size):
        """
        回傳展開後的 attn_mask 供 MultiheadAttention 使用。
        Shape: (batch_size * num_heads, J, J)
        """
        # self.bias: (H, J, J)
        return (self.bias
                .unsqueeze(0)                           # (1, H, J, J)
                .expand(batch_size, -1, -1, -1)         # (B, H, J, J)
                .reshape(batch_size * self.num_heads,   # (B*H, J, J)
                         self.bias.shape[1],
                         self.bias.shape[2]))


# ── 含 AAB 的空間 Encoder 層 ──────────────────────────────────────────────────
class SpatialEncoderLayerAAB(nn.Module):
    """
    標準 TransformerEncoderLayer + 解剖注意力偏置注入。
    結構與 nn.TransformerEncoderLayer(batch_first=True) 完全相同，
    唯一差異：self_attn 時傳入 attn_mask=anatomical_bias。
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None):
        # x: (B*T, J, d_model)
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x


# ── 主模型：ST-Transformer with AAB ──────────────────────────────────────────
class SpatialTemporalTransformerAAB(nn.Module):
    """
    與原始 SpatialTemporalTransformer 完全相同的 I/O 介面，
    空間編碼器改用 SpatialEncoderLayerAAB + AnatomicalAttentionBias。

    use_aab=False 時退化為標準 ST-Transformer（用於消融對照）。
    """
    def __init__(self, num_joints=13, in_channels=3,
                 d_model=256, nhead=8,
                 num_spatial_layers=2, num_temporal_layers=2,
                 num_classes=5, dropout=0.3,
                 use_aab=True):
        super().__init__()
        self.use_aab = use_aab

        # 1. Joint Embedding + 空間位置編碼
        self.joint_embed       = nn.Linear(in_channels, d_model)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, num_joints, d_model) * 0.02)

        # 2. Spatial Encoder（AAB 版）
        self.spatial_layers = nn.ModuleList([
            SpatialEncoderLayerAAB(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_spatial_layers)
        ])

        # 3. Anatomical Attention Bias（可開關）
        if use_aab:
            self.aab = AnatomicalAttentionBias(num_joints, nhead)
        else:
            self.aab = None

        # 4. Spatial Projection：展平後線性降維
        self.spatial_proj = nn.Linear(num_joints * d_model, d_model)

        # 5. Temporal Encoder（標準，不動）
        self.max_len = 128
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.max_len, d_model) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=num_temporal_layers)

        # 6. 分類頭
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, J, C)
        B, T, J, C = x.shape

        # Spatial 前處理
        h = self.joint_embed(x)             # (B, T, J, d_model)
        h = h + self.spatial_pos_embed      # broadcast 位置編碼
        h = h.view(B * T, J, -1)           # (B*T, J, d_model)

        # AAB mask（只計算一次，所有層共享）
        attn_mask = self.aab.get(B * T) if self.use_aab else None

        # Spatial Encoder
        for layer in self.spatial_layers:
            h = layer(h, attn_mask=attn_mask)

        # Spatial Projection
        h = h.reshape(B, T, J * self.joint_embed.out_features)  # (B, T, J*d)
        h = self.spatial_proj(h)                                  # (B, T, d)

        # Temporal Encoder
        h = h + self.temporal_pos_embed[:, :T, :]
        h = self.temporal_transformer(h)                          # (B, T, d)

        # Global Pooling + 分類
        return self.fc(h.mean(dim=1))                             # (B, num_classes)


# ── 快速驗證 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    for use_aab in [True, False]:
        m = SpatialTemporalTransformerAAB(use_aab=use_aab)
        m.eval()
        x = torch.randn(2, 64, 13, 3)
        with torch.no_grad():
            out = m(x)
        n = sum(p.numel() for p in m.parameters())
        tag = "AAB" if use_aab else "No-AAB"
        print(f"{tag}: output={tuple(out.shape)}  params={n:,} ({n/1e6:.3f}M)")

    # 顯示解剖距離矩陣
    dist = build_graph_distance()
    joints = ["Nose","LSh","RSh","LEl","REl","LWr","RWr",
              "LHip","RHip","LKn","RKn","LAn","RAn"]
    print("\nAnatomical Distance Matrix (first 5 joints):")
    header = f"{'':8s}" + "".join(f"{j:6s}" for j in joints[:5])
    print(header)
    for i in range(5):
        row = f"{joints[i]:8s}" + "".join(f"{dist[i,j]:6.0f}" for j in range(5))
        print(row)
