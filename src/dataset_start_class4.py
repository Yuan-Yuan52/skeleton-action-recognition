# src/dataset_start_class4.py
# Dataset for "start_class4" classification task
# Reads skeleton data and prepares samples for training/validation.
# Uses normalization and padding/cropping to fixed length.
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Mediapipe Pose 33 é»ç´¢å¼?
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def normalize_skeleton(seq, eps=1e-6):
    """
    ?ä??æ¬ pipeline ä¸?´ç?ç°¡å®æ­???ï?
    - æ¯ä?å¹æ¸æ???é«å¹³????å¹³ç§»?°ä¸­å¿?
    - ?¤ä»¥ (?©å¯¬ + é«å¯¬) ???å°ºåº¦æ?æºå?
    seq: (T, 33, 3)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, K, C = seq.shape
    coords = seq.copy()

    for t in range(T):
        frame = coords[t, :, :]
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

        coords[t, :, :] = frame

    return coords


def pad_or_crop(seq, max_len, random_crop=False):
    """
    seq: (T, 33, 3)
    - å¦æ? T > max_len: crop
      - train: random crop
      - val: center crop
    - å¦æ? T < max_len: ?¨æ?å¾ä?å¹?è?è£å° max_len
    """
    T = seq.shape[0]
    if T == max_len:
        return seq

    if T > max_len:
        if random_crop:
            start = np.random.randint(0, T - max_len + 1)
        else:
            start = max(0, (T - max_len) // 2)
        return seq[start:start + max_len]
    else:
        pad_len = max_len - T
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        return np.concatenate([seq, pad], axis=0)


class StartClass4Dataset(Dataset):
    """
    çµ¦ãèµ·å§å§¿?¢å?é¡ãç¨??Dataset??

    ?è? episodes_from_json_all.csv è£¡ç?ï¼?
      - video_id
      - lift_start_frame, lift_end_frame
      - start_class4
    ?å»å°æ? data_original/npy/{video_id}.npyï¼?
    ??lift æ®µæ½?ºä??¶ä???sample??
    """

    def __init__(
        self,
        df: pd.DataFrame,
        skeleton_dir: str = "data_original/npy",
        sample_stride: int = 2,
        max_len: int = 64,
        augment: bool = False,
    ):
        """
        df: å·²ç??æ¿¾å¥½ç? DataFrameï¼åªä¿ç? use_for_train == 1, start_class4 in [0..3]ï¼?
        skeleton_dir: ??{video_id}.npy ?è??å¤¾
        sample_stride: ?¶å? extract_skeleton ?¨ç? strideï¼ä???2ï¼?
        max_len: clip ?ºå??·åº¦ï¼éª¨?¶å?ï¼ï??è¨­ 64
        augment: train=True ?æ???random cropï¼val=False ??center crop
        """
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.sample_stride = sample_stride
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        vid = int(row["video_id"])
        start_f = int(row["lift_start_frame"])
        end_f = int(row["lift_end_frame"])
        label = int(row["start_class4"])

        # å½±ç? frame ??skeleton indexï¼å???skeleton æ¯?sample_stride å¹?ä?æ¬¡ï?
        s_idx = start_f // self.sample_stride
        e_idx = end_f // self.sample_stride

        skel_path = os.path.join(self.skeleton_dir, f"{vid:02d}.npy")
        seq = np.load(skel_path)  # (T_total, 33, 3)

        seq = seq[s_idx:e_idx + 1]          # (T_clip, 33, 3)
        seq = normalize_skeleton(seq)       # å¹³ç§» + å°ºåº¦
        seq = pad_or_crop(
            seq,
            self.max_len,
            random_crop=self.augment
        )                                    # (max_len, 33, 3)

        x = torch.from_numpy(seq).float()    # (T, K, C) ??DataLoader è®?(B,T,K,C)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
