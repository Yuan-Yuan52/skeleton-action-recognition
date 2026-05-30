# src/dataset_skeleton.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RIGHT_WRIST = 16  # Mediapipe Pose: 右手腕 index


class SkeletonDataset(Dataset):
    def __init__(self,
                 skeleton_dir,
                 labels_csv,
                 window_size=32,
                 stride=16,
                 binary_mode=False,
                 use_direction=False):
        """
        skeleton_dir: 存放 .npy (T, 33, 3) 的資料夾
        labels_csv  : 至少包含 ['video_id', 'label'] 兩欄
        window_size : 每個樣本的時間長度 (幀數)
        stride      : 滑動視窗步長
        binary_mode : True 時做 2 類：0=移動, 1=抬起/放下
        use_direction: True 時，第 3 個 channel 改成手的垂直方向 dy
        """
        self.skeleton_dir = skeleton_dir
        self.window_size = window_size
        self.stride = stride
        self.binary_mode = binary_mode
        self.use_direction = use_direction

        # 讀取標籤 CSV
        self.df = pd.read_csv(labels_csv)

        # 這兩個 list 是「真的用來 indexing 的東西」
        self.index_map = []  # [(video_id, start_frame), ...]
        self.labels = []     # 對應到每個 window 的 label（長度要和 dataset 一樣）

        # 逐列掃過每個 segment
        for _, row in self.df.iterrows():
            vid = str(row["video_id"])
            lab = int(row["label"])

            # 二類模式：0 -> 0, 1/2 -> 1
            if self.binary_mode:
                lab = 0 if lab == 0 else 1

            npy_path = os.path.join(self.skeleton_dir, f"{vid}.npy")
            if not os.path.exists(npy_path):
                # 找不到檔案就跳過
                continue

            seq = np.load(npy_path)
            T = seq.shape[0]

            t = 0
            while t + self.window_size <= T:
                # 每一個 window 視為一個樣本
                self.index_map.append((vid, t))
                self.labels.append(lab)
                t += self.stride

        print(f"[INFO] SkeletonDataset: windows={len(self.index_map)}, "
              f"binary_mode={self.binary_mode}, use_direction={self.use_direction}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid, start = self.index_map[idx]
        label = self.labels[idx]

        npy_path = os.path.join(self.skeleton_dir, f"{vid}.npy")
        seq = np.load(npy_path).astype(np.float32)  # (T, 33, 3)
        seq = np.nan_to_num(seq, nan=0.0)

        # 取出時間 window
        window = seq[start:start + self.window_size]  # (W, 33, 3)
        W, K, C = window.shape  # C=3: (x, y, conf)

        if self.use_direction:
            # 使用右手腕的 y 座標計算 dy，當作第 3 個 channel
            ys = window[:, RIGHT_WRIST, 1]              # (W,)
            dy = np.diff(ys, prepend=ys[0])             # (W,)
            max_abs = np.max(np.abs(dy)) + 1e-6
            dy = (dy / max_abs).astype(np.float32)      # 正規化

            # 把第三個 channel 改成 dy（原 conf 不用了）
            window[:, :, 2] = dy[:, None]

        x = torch.from_numpy(window)                   # (W, 33, 3)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
