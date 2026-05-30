# src/train_start_class4_baseline.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # å£æ? OMP ?è?è¼å¥è­¦å? (Windows)

import argparse
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models_skeleton import GRUClassifier
from utils import seed_everything, save_checkpoint, evaluate, make_weighted_sampler

# Mediapipe Pose 33 é»ç´¢å¼?
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


# =========================
# Feature helpers (feat5)
# =========================
def _safe_norm(v, eps=1e-6):
    return np.sqrt(np.sum(v * v, axis=-1) + eps)


def _angle_between(u, v, eps=1e-6):
    """
    u, v: (..., 2)
    return: angle in radians, shape (...)
    """
    nu = _safe_norm(u, eps)
    nv = _safe_norm(v, eps)
    dot = np.sum(u * v, axis=-1) / (nu * nv + eps)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def _wrap_pi(a):
    """wrap angle to [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_5_features(seq, vis_thr=0.5):
    """
    seq: (T, 33, 3)
      - [:,:,0:2] = (x,y)
      - [:,:,2]   = visibility/conf
    NOTE:
      å»ºè­° seq ?ç???normalize_skeleton (translation+scale)ï¼éæ¨£ hand_dist å°ºåº¦æ¯è?ç©©å?

    returns feat: (T, 5) float32
      [flex, twist, knee, hand_dist, valid_ratio]
      flex/twist/knee in [0,1] (rad/pi)
      hand_dist is roughly in a reasonable normalized scale (we later /5 -> [0,1] approx)
      valid_ratio in [0,1]
    """
    T = seq.shape[0]
    xy = seq[:, :, :2]
    vis = seq[:, :, 2]

    finite_xy = np.isfinite(xy).all(axis=-1)   # (T,33)
    valid_kp = finite_xy & (vis >= vis_thr)
    valid_ratio = valid_kp.mean(axis=1).astype(np.float32)  # (T,)

    def P(i):
        return xy[:, i, :]  # (T,2)

    hip_c = 0.5 * (P(LEFT_HIP) + P(RIGHT_HIP))
    sh_c  = 0.5 * (P(LEFT_SHOULDER) + P(RIGHT_SHOULDER))

    # flex: trunk vs vertical(up)
    trunk = sh_c - hip_c
    up = np.tile(np.array([[0.0, -1.0]], dtype=np.float32), (T, 1))
    flex = _angle_between(trunk, up)
    flex = np.nan_to_num(flex, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    flex = (flex / np.pi).astype(np.float32)

    # twist proxy: shoulder vector angle - hip vector angle
    sh_vec = P(RIGHT_SHOULDER) - P(LEFT_SHOULDER)
    hip_vec = P(RIGHT_HIP) - P(LEFT_HIP)
    sh_ang = np.arctan2(sh_vec[:, 1], sh_vec[:, 0])
    hip_ang = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    twist = np.abs(_wrap_pi(sh_ang - hip_ang))  # [0,pi]
    twist = np.nan_to_num(twist, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    twist = (twist / np.pi).astype(np.float32)

    # knee angle: angle at knee between (hip-knee) and (ankle-knee), take max(L/R)
    lk = P(LEFT_KNEE);  rk = P(RIGHT_KNEE)
    lh = P(LEFT_HIP);   rh = P(RIGHT_HIP)
    la = P(LEFT_ANKLE); ra = P(RIGHT_ANKLE)

    v1L = lh - lk; v2L = la - lk
    v1R = rh - rk; v2R = ra - rk
    kneeL = _angle_between(v1L, v2L)
    kneeR = _angle_between(v1R, v2R)
    knee = np.maximum(kneeL, kneeR)
    knee = np.nan_to_num(knee, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    knee = (knee / np.pi).astype(np.float32)

    # hand_dist: wrist center to hip center (already in normalized coordinates)
    w_c = 0.5 * (P(LEFT_WRIST) + P(RIGHT_WRIST))
    hand_dist = _safe_norm(w_c - hip_c).astype(np.float32)
    hand_dist = np.nan_to_num(hand_dist, nan=0.0, posinf=0.0, neginf=0.0)
    hand_dist = np.clip(hand_dist, 0.0, 5.0)  # avoid extreme outliers

    feat = np.stack([flex, twist, knee, hand_dist, valid_ratio], axis=1).astype(np.float32)
    return feat


# =========================
# Skeleton preprocess
# =========================
def normalize_skeleton(seq, eps=1e-6):
    """
    2D æ­???ï?
    - ä»¥è©+é«å¹³?ç¶ä¸­å? (0,0)
    - ä»¥è©å¯?é«å¯¬??scale ?ç¸®??
    seq: (T, 33, 3)
    """
    seq = np.asarray(seq, dtype=np.float32)
    coords = seq.copy()
    T = coords.shape[0]

    for t in range(T):
        frame = coords[t]

        ls = frame[LEFT_SHOULDER, :2]
        rs = frame[RIGHT_SHOULDER, :2]
        lh = frame[LEFT_HIP, :2]
        rh = frame[RIGHT_HIP, :2]

        center = (ls + rs + lh + rh) / 4.0
        frame[:, :2] -= center

        shoulder_w = np.linalg.norm(ls - rs)
        hip_w = np.linalg.norm(lh - rh)
        scale = shoulder_w + hip_w
        if (not np.isfinite(scale)) or (scale < eps):
            scale = 1.0

        frame[:, :2] /= scale
        coords[t] = frame

    return coords


def resample_seq(seq, out_len):
    """
    seq: (T, J, C) -> (out_len, J, C)
    - T > out_len: linspace ?æ¨£
    - T < out_len: ?è??å¾ä?å¹ padding
    """
    T = seq.shape[0]
    if T == out_len:
        return seq
    if T > out_len:
        idxs = np.linspace(0, T - 1, num=out_len, dtype=int)
        return seq[idxs]
    pad_len = out_len - T
    pad = np.tile(seq[-1:], (pad_len, 1, 1))
    return np.concatenate([seq, pad], axis=0)


# =========================
# Dataset
# =========================
class EpisodeSkeletonDataset(Dataset):
    """
    ä¸??sample = 1 ??episode ?éª¨?¶ç?æ®?(?ºå??·åº¦ out_len) + label (0..3)

    df å¿é??æ?ä½?
      - video_id
      - start_frame_skel, end_frame_skel
      - label: 0..3
    """

    def __init__(self, df, skeleton_dir, out_len=64, normalize=True, min_len=2,
                 use_feat5=0, vis_thr=0.5):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.normalize = normalize
        self.min_len = min_len

        # NEW: feature switch
        self.use_feat5 = int(use_feat5)
        self.vis_thr = float(vis_thr)

        self._skeleton_cache = {}
        self._video_len_cache = {}

        # ?é?æ¿¾é¯èª?rowï¼å??ãå¤ª?­ãè??ï?
        self.df = self._filter_invalid_rows(self.df)

        # for WeightedRandomSampler
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def _get_video_len(self, video_id):
        if video_id in self._video_len_cache:
            return self._video_len_cache[video_id]

        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"skeleton npy not found: {npy_path}")

        arr = np.load(npy_path, mmap_mode="r")
        T = int(arr.shape[0])
        self._video_len_cache[video_id] = T
        return T

    def _filter_invalid_rows(self, df):
        before = len(df)

        neg = int((df["end_frame_skel"] < df["start_frame_skel"]).sum())
        short = int(((df["end_frame_skel"] - df["start_frame_skel"] + 1) < self.min_len).sum())

        df = df[df["end_frame_skel"] >= df["start_frame_skel"]].copy()
        df = df[(df["end_frame_skel"] - df["start_frame_skel"] + 1) >= self.min_len].copy()

        oor = 0
        keep_idx = []
        for i, row in df.iterrows():
            vid = str(row["video_id"])
            s = int(row["start_frame_skel"])
            e = int(row["end_frame_skel"])
            T = self._get_video_len(vid)
            if s < 0 or e < 0 or s >= T or e >= T:
                oor += 1
                continue
            keep_idx.append(i)

        df2 = df.loc[keep_idx].reset_index(drop=True)
        after = len(df2)

        print(f"[INFO] Dataset filter: before={before}, after={after}, "
              f"dropped={before-after} (neg={neg}, short={short}, oor={oor})")
        return df2

    def _load_skeleton(self, video_id):
        if video_id in self._skeleton_cache:
            return self._skeleton_cache[video_id]

        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"skeleton npy not found: {npy_path}")

        seq = np.load(npy_path).astype(np.float32)  # (T,33,3)
        self._skeleton_cache[video_id] = seq
        return seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        s = int(row["start_frame_skel"])
        e = int(row["end_frame_skel"])
        label = int(row["label"])

        full_seq = self._load_skeleton(video_id)
        seg = full_seq[s:e + 1]  # inclusive, (T,33,3)

        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        # ??normalizeï¼å? resampleï¼å?ç®?feat5ï¼è?ç©©å?ï¼?
        if self.normalize:
            seg = normalize_skeleton(seg)

        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        seg = resample_seq(seg, self.out_len)  # (out_len,33,3)

        if self.use_feat5 == 1:
            feat5 = compute_5_features(seg, vis_thr=self.vis_thr)  # (T,5)

            feat5_scaled = feat5.copy()
            # hand_dist clip ??[0,5]ï¼ç¸®??[0,1] è¿ä¼¼
            feat5_scaled[:, 3] = feat5_scaled[:, 3] / 5.0

            # pseudo joints: (T,5,3)ï¼æ? feature å­å¨ channel0
            pseudo = np.zeros((seg.shape[0], 5, 3), dtype=np.float32)
            pseudo[:, :, 0] = feat5_scaled

            seg = np.concatenate([seg.astype(np.float32), pseudo], axis=1)  # (T,38,3)

        x = torch.from_numpy(seg)  # (T, 33/38, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# =========================
# Table load / split
# =========================
def load_episode_table_from_json(csv_path, target, sample_stride=1):
    """
    å¾?episodes_from_json_all.csv è®?è??ï?å»ºç?è¨ç·´??all_df??
    - target="start_class4" -> lift_start_frame/lift_end_frame
    - target="end_class4"   -> lower_start_frame/lower_end_frame
    """
    df = pd.read_csv(csv_path)

    if "use_for_train" in df.columns:
        df = df[df["use_for_train"] == 1].copy()

    if target not in df.columns:
        raise ValueError(f"target='{target}' not in columns. available={list(df.columns)}")

    df = df.dropna(subset=[target]).copy()

    if target == "start_class4":
        s_col, e_col = "lift_start_frame", "lift_end_frame"
    elif target == "end_class4":
        s_col, e_col = "lower_start_frame", "lower_end_frame"
    else:
        raise ValueError("target must be 'start_class4' or 'end_class4'")

    for c in ["video_id", s_col, e_col, target]:
        if c not in df.columns:
            raise ValueError(f"required col '{c}' not found in csv")

    df["start_frame_skel"] = (df[s_col].astype(int) // sample_stride).astype(int)
    df["end_frame_skel"] = (df[e_col].astype(int) // sample_stride).astype(int)

    raw = df[target].astype(int)
    mn, mx = int(raw.min()), int(raw.max())
    if mn >= 0 and mx <= 3:
        df["label"] = raw
    elif mn >= 1 and mx <= 4:
        df["label"] = raw - 1
    else:
        raise ValueError(f"Unexpected label range in {target}: min={mn}, max={mx}")

    df = df[["video_id", "start_frame_skel", "end_frame_skel", "label"]].reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    print("[INFO] Label distribution:\n", df["label"].value_counts().sort_index())
    return df


def split_by_video_stratified(all_df, val_ratio=0.2, seed=42):
    """
    group split by video_idï¼ä¸¦?¡é?è®?val ??label ?å??¥è??´é?
    NOTE:
      - çµ¦å??ä?ä»?all_df + ?ä???seedï¼éå?split ?¯ãåºå®å¯?ç¾?ç???
    """
    rng = random.Random(seed)

    by_vid = {}
    for vid, g in all_df.groupby("video_id"):
        cnt = g["label"].value_counts().to_dict()
        by_vid[vid] = np.array([cnt.get(i, 0) for i in range(4)], dtype=np.int64)

    vids = list(by_vid.keys())
    rng.shuffle(vids)

    total = sum(by_vid[v] for v in vids)
    target_val = (total * val_ratio).astype(np.int64)

    val_vids = set()
    val_cnt = np.zeros(4, dtype=np.int64)

    vids_sorted = sorted(vids, key=lambda v: by_vid[v].sum(), reverse=True)

    def score(cnt):
        return np.abs(cnt - target_val).sum()

    n_val_videos = max(1, int(len(vids) * val_ratio))

    for v in vids_sorted:
        if len(val_vids) >= n_val_videos:
            break
        new_cnt = val_cnt + by_vid[v]
        if score(new_cnt) <= score(val_cnt):
            val_vids.add(v)
            val_cnt = new_cnt

    remain = [v for v in vids_sorted if v not in val_vids]
    while len(val_vids) < n_val_videos and remain:
        v = remain.pop(0)
        val_vids.add(v)
        val_cnt += by_vid[v]

    train_df = all_df[~all_df["video_id"].isin(val_vids)].reset_index(drop=True)
    val_df = all_df[all_df["video_id"].isin(val_vids)].reset_index(drop=True)

    print("\n[INFO] Video split (stratified-group approx):")
    print("  Val videos  :", sorted(val_vids))
    print("[INFO] Train label dist:\n", train_df["label"].value_counts().sort_index())
    print("[INFO] Val   label dist:\n", val_df["label"].value_counts().sort_index())
    return train_df, val_df


# =========================
# Train
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="analysis/episodes_from_json_all.csv")
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--skeleton_dir", type=str, default="data_original/npy")
    parser.add_argument("--target", type=str, default="start_class4", help="start_class4 or end_class4")
    parser.add_argument("--out_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # NEW: feature switch
    parser.add_argument("--use_feat5", type=int, default=0, choices=[0, 1],
                        help="0: only skeleton (33 joints). 1: append 5 features as pseudo joints (38 joints).")
    parser.add_argument("--vis_thr", type=float, default=0.5,
                        help="visibility threshold used in feat5 computing (valid_ratio etc.)")

    # Windows: å»ºè­°?ç¨ 0ï¼è??å??é? 2~4
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/start_class4_gru")

    # è¨ç·´ç©©å???
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()
    seed_everything(args.seed)

    # ckpt_dir auto suffix to avoid mixing 33/38 checkpoints
    ckpt_dir = args.ckpt_dir
    if args.use_feat5 == 1 and (not ckpt_dir.endswith("_feat5")):
        ckpt_dir = ckpt_dir + "_feat5"
    if args.use_feat5 == 0 and ckpt_dir.endswith("_feat5"):
        # ä½ å??æ??æ?å®æ? *_feat5 ä½?use_feat5=0ï¼å°±?´æ¥ä¿ç?ä¹æ?å·?
        pass

    # 1) load table
    all_df = load_episode_table_from_json(args.csv_path, args.target, args.sample_stride)

    print("[INFO] segment length stats (skel idx):")
    lens = (all_df["end_frame_skel"] - all_df["start_frame_skel"] + 1)
    print(lens.describe(percentiles=[.5, .9, .95]))

    # 2) split
    train_df, val_df = split_by_video_stratified(all_df, args.val_ratio, args.seed)

    # 3) dataset/dataloader
    train_ds = EpisodeSkeletonDataset(
        train_df,
        skeleton_dir=args.skeleton_dir,
        out_len=args.out_len,
        normalize=True,
        use_feat5=args.use_feat5,
        vis_thr=args.vis_thr,
    )
    val_ds = EpisodeSkeletonDataset(
        val_df,
        skeleton_dir=args.skeleton_dir,
        out_len=args.out_len,
        normalize=True,
        use_feat5=args.use_feat5,
        vis_thr=args.vis_thr,
    )

    sampler = make_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 4) model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    num_joints = 33 + (5 if args.use_feat5 == 1 else 0)
    print(f"[INFO] use_feat5={args.use_feat5} -> num_joints={num_joints} | ckpt_dir={ckpt_dir}")

    model = GRUClassifier(num_joints=num_joints, in_channels=3, num_classes=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_acc = 0.0

    # 5) train loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)  # (B, T, 33/38, 3)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at epoch={epoch}: {loss.item()}")
                break

            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            seen += bs

        scheduler.step()
        avg_loss = total_loss / max(1, seen)

        # 6) val
        val_acc, cm, report = evaluate(model, val_loader, device, 4)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "use_feat5": args.use_feat5,
                "num_joints": num_joints,
                "vis_thr": args.vis_thr,
            },
            is_best=is_best,
            ckpt_dir=ckpt_dir,
            filename=f"epoch_{epoch}.pth",
        )

        print(f"Epoch {epoch}/{args.epochs} Loss={avg_loss:.4f} ValAcc={val_acc:.4f} BestAcc={best_acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
