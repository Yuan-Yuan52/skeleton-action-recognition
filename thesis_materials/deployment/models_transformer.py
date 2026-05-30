import torch
import torch.nn as nn

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, num_joints=33, in_channels=3,
                 d_model=256, nhead=8, num_spatial_layers=3, num_temporal_layers=4,
                 num_classes=5, dropout=0.3):
        super().__init__()
        
        # 1. 將 (X,Y,Vis) 投影到高維度空間
        self.joint_embed = nn.Linear(in_channels, d_model)
        
        # 2. 空間位置編碼 (區分 33 個不同的關節)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 1, num_joints, d_model))
        
        # 3. 空間 Transformer (學習同一影格內，各個關節的連動關係)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_spatial_layers)
        
        # 4. 投影層：把攤平的 33 個關節特徵壓回 d_model 維度，避免後續參數爆炸
        self.spatial_proj = nn.Linear(num_joints * d_model, d_model)
        
        # 4.5. 時間位置編碼 (用來區分影格的先後順序)
        self.max_len = 128
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.max_len, d_model))
        
        # 5. 時間 Transformer (學習不同影格之間的動作連續性)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_temporal_layers)
        
        # 6. 最終分類層
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Joints, Channels) = (B, T, 33, 3)
        B, T, J, C = x.shape
        
        # --- Spatial Module (空間處理) ---
        x = self.joint_embed(x) # (B, T, J, d_model)
        x = x + self.spatial_pos_embed
        
        x = x.view(B * T, J, -1) # (B*T, 33, d_model)
        x = self.spatial_transformer(x)
        
        # ★ 攤平 33 個關節保留所有細節，然後用 Linear 層降維
        x = x.view(B, T, J * self.joint_embed.out_features) # (B, T, 33 * d_model)
        x = self.spatial_proj(x) # (B, T, d_model)
        
        # --- Temporal Module (時間處理) ---
        x = x + self.temporal_pos_embed[:, :T, :]
        x = self.temporal_transformer(x) # (B, T, d_model)
        
        # 用 Global Average Pooling 代表整段動作的總結
        x_global = x.mean(dim=1) # (B, d_model)
        
        logits = self.fc(x_global)
        return logits