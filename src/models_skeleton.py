# 模型定義檔案
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, num_joints=33, in_channels=3,
                 hidden_size=256, num_layers=2,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.input_size = num_joints * in_channels
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B, T, K, C)
        B, T, K, C = x.shape
        x = x.view(B, T, K * C)
        out, _ = self.gru(x)       # (B, T, 2H)
        last = out[:, -1, :]
        logits = self.fc(last)     # (B, num_classes)
        return logits
