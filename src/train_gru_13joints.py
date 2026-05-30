import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models_skeleton import GRUClassifier
from train_transformer import build_phase_table, split_by_video
from train_comparison_models import EpisodePhaseDataset, FocalLoss, evaluate_model, EPISODES_CSV, SKELETON_DIR, TARGET_COL, NUM_CLASSES, EPOCHS, SEED

class GRU13JointsDataset(EpisodePhaseDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # x is (64, 33, 3). Select 13 core joints:
        CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        x = x[:, CORE_JOINTS, :]
        return x, y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training GRU model with 13 core joints (same as Transformer)...")
    
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)
    
    train_ds = GRU13JointsDataset(train_df, skeleton_dir=SKELETON_DIR, out_len=64, normalize=True, use_aug=True, mode="train")
    val_ds   = GRU13JointsDataset(val_df,   skeleton_dir=SKELETON_DIR, out_len=64, normalize=True, use_aug=False, mode="val")
    
    from utils import make_weighted_sampler, seed_everything
    seed_everything(SEED)
    sampler = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    model = GRUClassifier(num_joints=13, in_channels=6, num_classes=NUM_CLASSES).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Concat velocity features
            velocity = torch.zeros_like(x)
            velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
            x_combined = torch.cat([x, velocity], dim=-1)
            
            optimizer.zero_grad()
            logits = model(x_combined)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        val_acc, _, _, _ = evaluate_model(model, val_loader, device, "gru")
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = "checkpoints/gru_13joints"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc
            }, os.path.join(ckpt_dir, "best.pth"))
            
    print(f"\n[DONE] GRU (13 joints) Best Val Acc = {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
