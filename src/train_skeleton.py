# 主訓練模型程式
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OpenMP 重複載入的錯誤

import argparse
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from dataset_skeleton import SkeletonDataset
from models_skeleton import GRUClassifier
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_dir", type=str, default="data/skeleton_segments_npy")
    parser.add_argument("--train_csv", type=str, default="data/split/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/split/val.csv")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/skeleton_gru")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary", action="store_true",help="Train binary classification: 0=移動, 1=抬起/放下")
    parser.add_argument("--use_direction", action="store_true",help="Add hand vertical direction feature (dy) as extra input")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", type=str,
                        default="video_skeleton_gru",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    if args.binary:
        args.num_classes = 2
    else:
        args.num_classes = 3

    seed_everything(args.seed)
    # ---- W&B init (optional) ----
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "binary": args.binary,
                "use_direction": args.use_direction,
                "window_size": args.window_size,
                "stride": args.stride,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "num_classes": args.num_classes,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> CUDA device:", torch.cuda.get_device_name(0))


    # 建 dataset / dataloader
    train_ds = SkeletonDataset(
        skeleton_dir=args.skeleton_dir,
        labels_csv=args.train_csv,
        window_size=args.window_size,
        stride=args.stride,
        binary_mode=args.binary,
        use_direction=args.use_direction,
    )
    val_ds = SkeletonDataset(
        skeleton_dir=args.skeleton_dir,
        labels_csv=args.val_csv,
        window_size=args.window_size,
        stride=args.window_size,   # val: non-overlap
        binary_mode=args.binary,
        use_direction=args.use_direction,
    )


    sampler = make_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,   # 在 CPU 上其實不需要 pin_memory
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # 建 model
    model = GRUClassifier(num_joints=33, in_channels=3,
                          num_classes=args.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # (B, T, K, C)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)

        val_acc, cm, report = evaluate(model, val_loader, device, args.num_classes)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
            })
            
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            is_best=is_best,
            ckpt_dir=args.ckpt_dir,
            filename=f"epoch_{epoch}.pth",
        )

        print(f"Epoch {epoch}/{args.epochs} "
              f"Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
