# src/train_phase_class4.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # å£æ? OMP ?è?è¼å¥è­¦å?

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    import wandb
except Exception:
    wandb = None
from models_transformer import SpatialTemporalTransformer
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

# Mediapipe Pose 33 é»ç´¢å¼ï?è·ä?ä¹å?ä¸æ¨??
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


# -----------------------------
# ?±ç¨?å?å·¥å·
# -----------------------------
def normalize_skeleton(seq, eps=1e-6):
    """
    è·?auto_episode_from_binary / train_skeleton ä¸æ¨?? 2D æ­???ï?
    - ä»¥è©+é«å¹³?ç¶ä¸­å? (0,0)
    - ä»¥è©å¯?é«å¯¬??scale ?ç¸®??
    seq: (T, 33, 3)
    ?å³: (T, 33, 3)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, K, C = seq.shape
    coords = seq.copy()

    for t in range(T):
        frame = coords[t]

        # ==== ?°å?ï¼å???NaN/Inf æ¸æ? ====
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


def resample_seq(seq, out_len):
    """
    ?ä?æ®µé·åº?T ?å??ï?è®æ??ºå??·åº¦ out_len??
    - å¦æ? T > out_len: è¦å? downsample
    - å¦æ? T < out_len: ?è??å¾ä?å¹è£å° out_len
    seq: (T, 33, 3)  ->  (out_len, 33, 3)
    """
    T = seq.shape[0]
    if T == out_len:
        return seq
    if T > out_len:
        idxs = np.linspace(0, T - 1, num=out_len, dtype=int)
        return seq[idxs]
    else:
        pad_len = out_len - T
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        return np.concatenate([seq, pad], axis=0)


# -----------------------------
# Datasetï¼ä?ç­?= ä¸??episode phaseï¼lift ??lowerï¼?
# -----------------------------
class EpisodePhaseDataset(Dataset):
    """
    ä¸??sample = 1 æ®µéª¨?¶ç?æ®?(?ºå??·åº¦ out_len) + 1 ??label (0~3)

    df ?è¦æ?ä½ï?
      - video_id: å°æ? skeleton_npy/{video_id}.npy
      - start_frame_skel, end_frame_skel
      - label: 0..3 ï¼ç´?¥ä???start_class4 / end_class4ï¼?
    """

    def __init__(self, df, skeleton_dir, out_len=64, normalize=True, use_aug=True, mode="train"):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.normalize = normalize
        self.use_aug = use_aug
        self.mode = mode

        # ?ºä? WeightedRandomSampler
        self.labels = self.df["label"].astype(int).tolist()

        # cache æ¯æ¯å½±ç???skeletonï¼é¿?é?è¤è?æª?
        self._skeleton_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_skeleton(self, video_id):
        if video_id in self._skeleton_cache:
            return self._skeleton_cache[video_id]

        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"skeleton npy not found: {npy_path}")

        seq = np.load(npy_path).astype(np.float32)  # (T, 33, 3)
        self._skeleton_cache[video_id] = seq
        return seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        s = int(row["start_frame_skel"])
        e = int(row["end_frame_skel"])
        label = int(row["label"])  # 0..3

        full_seq = self._load_skeleton(video_id)
        T = full_seq.shape[0]
        if T <= 0:
            raise ValueError(
                f"Empty skeleton sequence: video_id={video_id}, s={s}, e={e}, T={T}"
            )

        # ==== ?²å?ï¼clamp ç¯å?ï¼é¿?è???or è² æ¸ ====
        s = max(0, min(s, T - 1))
        e = max(0, min(e, T - 1))
        if e < s:
            e = s

        if (e - s + 1) <= 0:
            raise ValueError(
                f"Invalid segment after clamp: video_id={video_id}, s={s}, e={e}, T={T}"
            )

        seg = full_seq[s:e + 1]  # (Te, 33, 3)

        if self.normalize:
            seg = normalize_skeleton(seg)

        # ==== ?ä??ªä?æ¬¡ï?æ¸é¤ NaN / Inf ====
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        seg = resample_seq(seg, self.out_len)  # (out_len, 33, 3)
        
        # ==== ?ªä???13 ?æ ¸å¿é?ç¯ (é¼»å? + è»å¹¹å??? ====
        CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        seg = seg[:, CORE_JOINTS, :]  # (out_len, 13, 3)

        # ==== è¨ç·´?é²è?è³æ?å¾®æ¾??Data Augmentation ====
        if self.mode == "train" and self.use_aug:
            # 1. ?¨æ??è? (Random Rotation)ï¼æ¨¡?¬ä??æ??è?åº?
            if random.random() < 0.5:
                angle = random.uniform(-15.0, 15.0) * np.pi / 180.0
                c = np.cos(angle)
                s_a = np.sin(angle)
                old_x = seg[..., 0].copy()
                old_z = seg[..., 2].copy()
                seg[..., 0] = old_x * c - old_z * s_a
                seg[..., 2] = old_x * s_a + old_z * c
            
            # 2. ?¨æ??è??å? (Gaussian Jitter)ï¼æ¨¡?¬éª¨?¶è¾¨è­ä?ç©©å?
            if random.random() < 0.5:
                noise = np.random.normal(loc=0.0, scale=0.015, size=seg.shape).astype(np.float32)
                seg += noise

        x = torch.from_numpy(seg).float()  # (T, 13, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -----------------------------
# å¾?episodes_from_json_all_more.csv ?¢ç? phase è¡?
# -----------------------------
def build_phase_table(csv_path, phase, target, sample_stride=2):
    """
    episodes_csv ?è¦æ?ä½ï?
      - video_id
      - lift_start_frame, lift_end_frame
      - lower_start_frame, lower_end_frame
      - start_class4, end_class4  ï¼?..3ï¼?
      - use_for_train (0/1)      ï¼å??æ??å°±?¨ç¨ï¼?

    phase:
      - "lift"  -> ??lift_* + start_class4
      - "lower" -> ??lower_* + end_class4

    target:
      - "start_class4" or "end_class4"
    """
    df = pd.read_csv(csv_path)

    # ?é?æ¿?use_for_train == 1ï¼å??æ?ï¼?
    if "use_for_train" in df.columns:
        df = df[df["use_for_train"] == 1].copy()

    if phase == "lift":
        start_col = "lift_start_frame"
        end_col = "lift_end_frame"
    elif phase == "lower":
        start_col = "lower_start_frame"
        end_col = "lower_end_frame"
    else:
        raise ValueError(f"Unknown phase: {phase}, must be 'lift' or 'lower'")

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in CSV.")

    required = ["video_id", start_col, end_col, target]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is required in CSV.")

    # ?ªæ?æ²æ?æ¨å¥½ frame ??label ?å?
    df = df.dropna(subset=required).copy()

    # Filter invalid labels outside [0, 4]
    df = df[df[target].astype(int).between(0, 4)].copy()

    # å»ºä??ä¹¾æ·¨ç?è¡¨ï?video_id, start_frame_skel, end_frame_skel, label (0..3)
    start_frame_skel = df[start_col].astype(int) // sample_stride
    end_frame_skel = df[end_col].astype(int) // sample_stride

    def _normalize_video_id(v):
        if pd.isna(v):
            return v
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else str(v)
        s = str(v)
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s

    phase_df = pd.DataFrame({
        "video_id": df["video_id"].apply(_normalize_video_id),
        "start_frame_skel": start_frame_skel,
        "end_frame_skel": end_frame_skel,
        "label": df[target].astype(int)   # å·²ç???0..3
    })

    print(f"\n[INFO] Phase='{phase}', target='{target}'")
    print(f"[INFO] Total usable episodes = {len(phase_df)}")
    print("[INFO] Label distribution (0..3):")
    print(phase_df["label"].value_counts().sort_index())

    return phase_df


def split_by_video(phase_df, val_ratio=0.2, seed=42):
    """
    ä»?video_id ?ºå®ä½å? train/val splitï¼é¿?å?ä¸?¯å½±?å??åº?¾å¨ train & val??
    """
    video_ids = sorted(phase_df["video_id"].unique())
    random.Random(seed).shuffle(video_ids)

    n_val = max(1, int(len(video_ids) * val_ratio))
    val_vids = set(video_ids[:n_val])

    train_df = phase_df[~phase_df["video_id"].isin(val_vids)].reset_index(drop=True)
    val_df = phase_df[phase_df["video_id"].isin(val_vids)].reset_index(drop=True)

    print("\n[INFO] Video split:")
    print("  Train videos:", sorted(set(train_df["video_id"].unique())))
    print("  Val videos  :", sorted(set(val_df["video_id"].unique())))
    print(f"[INFO] Train episodes = {len(train_df)}, Val episodes = {len(val_df)}")
    return train_df, val_df


# -----------------------------
# main: è¨ç·´ GRU 4 é¡å?é¡å¨
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes_csv",
        type=str,
        default="analysis/episodes_from_json_all_zero.csv",
        help="?å«???episode æ¨è¨»??CSV"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="lift",
        choices=["lift", "lower"],
        help="è¦è?ç·´åªä¸æ®µï?lift(?¬èµ·) ??lower(?¾ä?)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="class_id",
        help="è¦ç¶ label ?æ?ä½ï?start_class4 ??end_class4"
    )
    parser.add_argument(
        "--skeleton_dir",
        type=str,
        default="data_original/npy",
        help="Dir containing {video_id}.npy"
    )
    parser.add_argument("--out_len", type=int, default=64,
                        help="Resampled sequence length for GRU input")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_spatial_layers", type=int, default=3, help="Number of spatial transformer layers")
    parser.add_argument("--num_temporal_layers", type=int, default=4, help="Number of temporal transformer layers")
    parser.add_argument("--disable_norm", action="store_true", help="Disable Center-Scale Normalization")
    parser.add_argument("--disable_aug", action="store_true", help="Disable Data Augmentation")

    # ==== æ¯è?ä¿å??è¨­å®ï??é? lr, ?æ? label_smoothing ====
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=2,
                        help="Skeleton sampling stride for frame alignment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/phase_newway_class4_gru")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="action_class4")
    parser.add_argument("--wandb_entity", type=str, default="r13941031-national-taiwan-university")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed. Please install wandb or disable --use_wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    # 1. å¾?episodes CSV ?¢ç??å?phase ?è?ç·´è¡¨
    phase_df = build_phase_table(args.episodes_csv, args.phase, args.target,
                                 sample_stride=args.sample_stride)

    # 2. ä»?video_id ??train/val split
    train_df, val_df = split_by_video(phase_df, args.val_ratio, args.seed)

    # 3. Dataset / DataLoader
    train_ds = EpisodePhaseDataset(
        train_df,
        skeleton_dir=args.skeleton_dir,
        out_len=args.out_len,
        normalize=not args.disable_norm,
        use_aug=not args.disable_aug,
        mode="train"
    )
    val_ds = EpisodePhaseDataset(
        val_df,
        skeleton_dir=args.skeleton_dir,
        out_len=args.out_len,
        normalize=not args.disable_norm,
        use_aug=not args.disable_aug,
        mode="val"
    )

    sampler = make_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # 4. GRU æ¨¡å?ï¼? é¡ï?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    num_classes = 5
    model = SpatialTemporalTransformer(
        num_joints=13, 
        in_channels=3,  
        d_model=256, 
        nhead=8, 
        num_spatial_layers=args.num_spatial_layers, 
        num_temporal_layers=args.num_temporal_layers,
        num_classes=num_classes, 
        dropout=0.3
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ckpt è·¯å?? ä? phase / target æ¯è?æ¸æ?
    ckpt_dir = os.path.join(args.ckpt_dir, f"{args.phase}_{args.target}")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_acc = 0.0

    # 5. è¨ç·´ loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)  # (B, T, K, C)
            y = y.to(device)  # (B,)

            # ==== Debugï¼æª¢??input ?¯å¦??NaN / Inf ====
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[WARN] NaN/Inf in input at epoch {epoch}, step {step}")
                continue

            optimizer.zero_grad()
            logits = model(x)  # (B, num_classes)

            # ==== Debugï¼æª¢??logits ====
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[WARN] NaN/Inf in logits at epoch {epoch}, step {step}")
                print("  logits stats:", logits.min().item(), logits.max().item())
                continue

            loss = criterion(logits, y)

            # ==== Debugï¼æª¢??loss ====
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf loss at epoch {epoch}, step {step}")
                print("  x mean/std:", x.mean().item(), x.std().item())
                print("  logits mean/std:", logits.mean().item(), logits.std().item())
                continue

            loss.backward()

            # ==== ? ä? gradient clippingï¼é¿??GRU ?ç¸ ====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader.dataset))

        # é©è?
        val_acc, cm, report = evaluate(model, val_loader, device, num_classes)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            is_best=is_best,
            ckpt_dir=ckpt_dir,
            filename=f"epoch_{epoch}.pth",
        )

        print(f"Epoch {epoch}/{args.epochs} "
              f"Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
