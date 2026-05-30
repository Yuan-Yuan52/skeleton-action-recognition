# -*- coding: utf-8 -*-
# Binary event training from episodes CSV with sliding windows.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import wandb
except Exception:
    wandb = None
from torch.utils.data import Dataset, DataLoader

from models_skeleton import GRUClassifier
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

# Mediapipe Pose 33 indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def collect_val_probs(model, val_loader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    if len(probs) == 0:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)


def sweep_thresholds(p1, y, thr_list=None, min_precision=None):
    if thr_list is None:
        thr_list = np.arange(0.01, 1.0, 0.01)
    best_f1 = {"thr": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_prec = {"thr": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for thr in thr_list:
        pred = (p1 >= thr).astype(np.int32)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        precision = tp / max(1e-8, (tp + fp))
        recall = tp / max(1e-8, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-8, (precision + recall))

        if f1 > best_f1["f1"]:
            best_f1 = {"thr": float(thr), "precision": precision, "recall": recall, "f1": f1}

        if min_precision is not None and precision >= min_precision:
            if recall > best_prec["recall"]:
                best_prec = {"thr": float(thr), "precision": precision, "recall": recall, "f1": f1}

    return best_f1, best_prec


def normalize_skeleton(seq, eps=1e-6):
    """
    Normalize skeleton to be translation/scale-invariant.
    seq: (T, 33, 3)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, K, C = seq.shape
    coords = seq.copy()

    for t in range(T):
        frame = coords[t]
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

        ls = frame[LEFT_SHOULDER, :2]
        rs = frame[RIGHT_SHOULDER, :2]
        lh = frame[LEFT_HIP, :2]
        rh = frame[RIGHT_HIP, :2]

        center = (ls + rs + lh + rh) / 4.0
        frame[:, :2] -= center

        shoulder_w = np.linalg.norm(ls - rs)
        hip_w = np.linalg.norm(lh - rh)
        scale = shoulder_w + hip_w
        if scale < eps:
            scale = 1.0
        frame[:, :2] /= scale

        coords[t] = frame
    return coords


class BinaryEventWindowDataset(Dataset):
    """
    Build sliding windows from full skeleton sequence.
    Label = 1 if window center is inside [event_start, event_end],
    excluding ignore zones near boundaries.
    """

    def __init__(
        self,
        df,
        skeleton_dir,
        seg_len=16,
        seg_stride=1,
        val_seg_stride=None,
        sample_stride=2,
        ignore_margin=2,
        pos_overlap=0.5,
        neg_overlap=0.1,
        csv_frame_unit="auto",
        normalize=True,
    ):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.val_seg_stride = val_seg_stride
        self.sample_stride = sample_stride
        self.ignore_margin = ignore_margin
        self.pos_overlap = pos_overlap
        self.neg_overlap = neg_overlap
        self.csv_frame_unit = csv_frame_unit
        self.normalize = normalize

        self._skeleton_cache = {}
        self._index = []
        self._labels = []

        self._build_index()

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self._index)

    def _load_skeleton(self, video_id):
        if video_id in self._skeleton_cache:
            return self._skeleton_cache[video_id]
        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Skeleton npy not found: {npy_path}")
        seq = np.load(npy_path).astype(np.float32)  # (T, 33, 3)
        self._skeleton_cache[video_id] = seq
        return seq

    def _build_index(self):
        total_videos = len(self.df)
        print(f"[INFO] Building windows from {total_videos} episodes...")
        for i, row in enumerate(self.df.iterrows(), start=1):
            row = row[1]
            video_id = str(row["video_id"])
            start_frame = int(row["lift_start_frame"])
            end_frame = int(row["lower_end_frame"])

            seq = self._load_skeleton(video_id)
            T = seq.shape[0]
            if T <= 0:
                continue

            # All CSV frames are treated as raw frames.
            unit_mode = "raw"
            start_frame = start_frame // self.sample_stride
            end_frame = end_frame // self.sample_stride
            print(f"[INFO] video_id={video_id} frame_unit={unit_mode} "
                  f"raw_start={row['lift_start_frame']} raw_end={row['lower_end_frame']} "
                  f"used_start={start_frame} used_end={end_frame} T={T}")

            start_frame = max(0, min(start_frame, T - 1))
            end_frame = max(0, min(end_frame, T - 1))
            if end_frame < start_frame:
                end_frame = start_frame

            seg_stride = self.seg_stride if self.val_seg_stride is None else self.val_seg_stride
            t = 0
            while t + self.seg_len <= T:
                win_start = t
                win_end = t + self.seg_len - 1
                overlap_len = max(0, min(win_end, end_frame) - max(win_start, start_frame) + 1)
                overlap_ratio = overlap_len / float(self.seg_len)

                if overlap_ratio >= self.pos_overlap:
                    label = 1
                elif overlap_ratio <= self.neg_overlap:
                    label = 0
                else:
                    t += seg_stride
                    continue

                self._index.append((video_id, t))
                self._labels.append(label)
                t += seg_stride

            if i % 50 == 0 or i == total_videos:
                print(f"[INFO] Processed {i}/{total_videos} episodes. "
                      f"Windows so far: {len(self._index)}")

    def __getitem__(self, idx):
        video_id, t = self._index[idx]
        seq = self._load_skeleton(video_id)
        window = seq[t:t + self.seg_len]

        if self.normalize:
            window = normalize_skeleton(window)

        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(window)  # (T, K, C)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y


def split_by_video(df, val_ratio=0.2, seed=42):
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    video_ids = sorted(df["video_id"].unique())
    random.Random(seed).shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * val_ratio))
    val_vids = set(video_ids[:n_val])

    train_df = df[~df["video_id"].isin(val_vids)].reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_vids)].reset_index(drop=True)
    print(f"[INFO] Split by video_id: total_videos={len(video_ids)} "
          f"train_episodes={len(train_df)} val_episodes={len(val_df)}")
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_csv", type=str,
                        default="analysis/episodes_from_json_all.csv")
    parser.add_argument("--skeleton_dir", type=str, default="data_original/npy")
    parser.add_argument("--seg_len", type=int, default=16)
    parser.add_argument("--seg_stride", type=int, default=2)
    parser.add_argument("--val_seg_stride", type=int, default=None)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--ignore_margin", type=int, default=2)
    parser.add_argument("--pos_overlap", type=float, default=0.5)
    parser.add_argument("--neg_overlap", type=float, default=0.1)
    parser.add_argument("--csv_frame_unit", type=str, default="auto",
                        choices=["auto", "raw", "skeleton"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/skeleton_gru_new")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true",
                        help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", type=str,
                        default="video_skeleton_gru",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    args.num_classes = 2

    seed_everything(args.seed)

    # ---- W&B init (optional) ----
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed. Please install wandb or disable --use_wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "episodes_csv": args.episodes_csv,
                "seg_len": args.seg_len,
                "seg_stride": args.seg_stride,
                "val_seg_stride": args.val_seg_stride,
                "sample_stride": args.sample_stride,
                "ignore_margin": args.ignore_margin,
                "pos_overlap": args.pos_overlap,
                "neg_overlap": args.neg_overlap,
                "csv_frame_unit": args.csv_frame_unit,
                "val_ratio": args.val_ratio,
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

    df = pd.read_csv(args.episodes_csv)
    if "use_for_train" in df.columns:
        before_cnt = len(df)
        df = df[df["use_for_train"] == 1].copy()
        after_cnt = len(df)
        print(f"[INFO] use_for_train filter: {before_cnt} -> {after_cnt}")
    required = ["video_id", "lift_start_frame", "lower_end_frame"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is required in episodes_csv.")
    df = df.dropna(subset=required).copy()
    train_df, val_df = split_by_video(df, args.val_ratio, args.seed)

    train_ds = BinaryEventWindowDataset(
        train_df,
        skeleton_dir=args.skeleton_dir,
        seg_len=args.seg_len,
        seg_stride=args.seg_stride,
        val_seg_stride=None,
        sample_stride=args.sample_stride,
        ignore_margin=args.ignore_margin,
        pos_overlap=args.pos_overlap,
        neg_overlap=args.neg_overlap,
        csv_frame_unit=args.csv_frame_unit,
        normalize=True,
    )
    val_ds = BinaryEventWindowDataset(
        val_df,
        skeleton_dir=args.skeleton_dir,
        seg_len=args.seg_len,
        seg_stride=args.seg_stride,
        val_seg_stride=args.val_seg_stride if args.val_seg_stride is not None else args.seg_len,
        sample_stride=args.sample_stride,
        ignore_margin=args.ignore_margin,
        pos_overlap=args.pos_overlap,
        neg_overlap=args.neg_overlap,
        csv_frame_unit=args.csv_frame_unit,
        normalize=True,
    )
    print("[INFO] train windows:", len(train_ds), "pos:", sum(train_ds.labels),
          "neg:", len(train_ds.labels) - sum(train_ds.labels))
    print("[INFO] val windows:", len(val_ds), "pos:", sum(val_ds.labels),
          "neg:", len(val_ds.labels) - sum(val_ds.labels))

    sampler = make_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
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
    if len(val_ds) == 0:
        print("[WARN] Validation dataset is empty. Skipping validation.")

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
    print("[INFO] Using WeightedSampler only (no class weight).")

    best_acc = 0.0
    best_f1 = 0.0

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

        if len(val_ds) == 0:
            val_acc, cm, report = 0.0, None, "N/A (empty val set)"
            f1_pos = 0.0
            is_best = False
        else:
            val_acc, cm, report = evaluate(model, val_loader, device, args.num_classes)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            f1_pos = (2 * tp) / max(1e-8, (2 * tp + fp + fn))
            is_best = f1_pos > best_f1
            best_f1 = max(best_f1, f1_pos)
            p1, y = collect_val_probs(model, val_loader, device)
            best_f1_thr, best_prec_thr = sweep_thresholds(
                p1, y, min_precision=0.80
            )
            if best_f1_thr["thr"] is not None:
                print("[VAL] Best F1 thr="
                      f"{best_f1_thr['thr']:.2f} "
                      f"P={best_f1_thr['precision']:.3f} "
                      f"R={best_f1_thr['recall']:.3f} "
                      f"F1={best_f1_thr['f1']:.3f}")
            if best_prec_thr["thr"] is not None:
                print("[VAL] Precision>=0.80 thr="
                      f"{best_prec_thr['thr']:.2f} "
                      f"P={best_prec_thr['precision']:.3f} "
                      f"R={best_prec_thr['recall']:.3f} "
                      f"F1={best_prec_thr['f1']:.3f}")

        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": val_acc,
                "val_f1_pos": f1_pos,
                "best_f1": best_f1,
                "lr": optimizer.param_groups[0]["lr"],
            })

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
            },
            is_best=is_best,
            ckpt_dir=args.ckpt_dir,
            filename=f"epoch_{epoch}.pth",
        )

        print(f"Epoch {epoch}/{args.epochs} "
              f"Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestF1={best_f1:.4f} F1Pos={f1_pos:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
