# src/auto_episode_from_binary.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 壓掉 OMP 重複載入警告

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from models_skeleton import GRUClassifier
from kim_features_V2 import compute_kim_features

# Mediapipe Pose 33 點索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def normalize_skeleton(seq, eps=1e-6):
    """
    簡單版骨架正規化：
    - 以肩/髖平均當中心 (0,0)
    - 以肩寬 + 髖寬作為尺度，做縮放
    這邊的邏輯要盡量跟 train 時一致，如果你之後有改 dataset_skeleton，
    記得也同步這裡。
    """
    seq = np.asarray(seq, dtype=np.float32)  # (T, 33, 3)
    T, K, C = seq.shape
    coords = seq.copy()

    for t in range(T):
        frame = coords[t, :, :]
        ls = frame[LEFT_SHOULDER, :2]
        rs = frame[RIGHT_SHOULDER, :2]
        lh = frame[LEFT_HIP, :2]
        rh = frame[RIGHT_HIP, :2]

        # center: 肩 + 髖的平均
        center = (ls + rs + lh + rh) / 4.0
        frame[:, :2] -= center

        # scale: 肩寬 + 髖寬
        shoulder_w = np.linalg.norm(ls - rs)
        hip_w = np.linalg.norm(lh - rh)
        scale = shoulder_w + hip_w
        if scale < eps:
            scale = 1.0
        frame[:, :2] /= scale

        coords[t, :, :] = frame

    return coords


def windows_from_seq(seq, window_size=32, stride=16):
    """
    把一整段骨架 seq (T, K, C) 切成 sliding windows:
    回傳 windows: (N, window_size, K, C), 以及每個 window 對應的起始 frame index
    """
    T, K, C = seq.shape
    if T < window_size:
        # 太短就直接 pad 到 window_size
        pad_len = window_size - T
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        seq_pad = np.concatenate([seq, pad], axis=0)
        return np.expand_dims(seq_pad, axis=0), [0]

    starts = list(range(0, T - window_size + 1, stride))
    windows = []
    for s in starts:
        e = s + window_size
        windows.append(seq[s:e])
    windows = np.stack(windows, axis=0)  # (N, window_size, K, C)
    return windows, starts


def aggregate_windows_to_frames(pred_windows, window_starts, T, window_size, stride,
                                frame_pos_thr=0.5):
    """
    將每個 window 的 binary 預測 (0/1) 彙整到 frame-level。
    簡單作法：對覆蓋到某一個 frame 的所有 window，取平均，再 threshold 0.5。
    """
    frame_score = np.zeros(T, dtype=np.float32)
    frame_count = np.zeros(T, dtype=np.float32)

    for pw, s in zip(pred_windows, window_starts):
        e = min(s + window_size, T)
        # 如果 pred=1，就對覆蓋到的 frame 加 1
        if pw == 1:
            frame_score[s:e] += 1.0
        frame_count[s:e] += 1.0

    # 避免除以 0
    valid_mask = frame_count > 0
    frame_prob = np.zeros(T, dtype=np.float32)
    frame_prob[valid_mask] = frame_score[valid_mask] / frame_count[valid_mask]

    # threshold at 0.5 → 0/1 state
    frame_state = (frame_prob >= frame_pos_thr).astype(np.int32)
    return frame_state, frame_prob


def find_episodes_from_state(frame_state, min_frames=30):
    """
    給一個 0/1 的 frame_state，找連續的 1-run。
    每個 run 長度 >= min_frames 才算一個 episode。
    回傳 list of (start_idx, end_idx)（含 end）
    """
    T = len(frame_state)
    episodes = []
    in_ep = False
    start = 0

    for i in range(T):
        s = frame_state[i]
        if not in_ep and s == 1:
            in_ep = True
            start = i
        elif in_ep and (s == 0 or i == T - 1):
            end = i - 1 if s == 0 else i
            length = end - start + 1
            if length >= min_frames:
                episodes.append((start, end))
            in_ep = False

    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_path", type=str,
                        default="data/skeleton_npy/16.npy",
                        help="整支影片骨架序列 .npy (T, 33, 3)")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/skeleton_gru_binary/best.pth",
                        help="Binary GRU 模型 checkpoint")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="原始影片的 fps（用來換算時間）")
    parser.add_argument("--sample_stride", type=int, default=2,
                        help="抽 skeleton 時的 stride，例如 extract_skeleton 的 --sample_stride")
    parser.add_argument("--min_frames", type=int, default=30,
                        help="最小 episode 長度 (skeleton frames)")
    parser.add_argument("--out_dir", type=str, default="analysis",
                        help="輸出 CSV 存放目錄")
    parser.add_argument("--win_pos_thr", type=float, default=0.5,
                        help="window 層級判定為【拿起/放下】的機率門檻，預設 0.5")
    parser.add_argument("--frame_pos_thr", type=float, default=0.5,
                        help="frame 層級判定為【拿起/放下】的機率門檻，預設 0.5")


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1. 載入骨架 ----------
    seq = np.load(args.skeleton_path)  # (T, 33, 3)
    T, K, C = seq.shape
    print("========== Auto Episodes from Binary Model ==========")
    print(f"[INFO] skeleton_path = {args.skeleton_path}")
    print(f"[INFO] T_total = {T} skeleton frames, fps(video)={args.fps}, sample_stride={args.sample_stride}")

    seq_norm = normalize_skeleton(seq)  # (T, 33, 3)

    # ---------- 2. 切 window ----------
    windows_np, starts = windows_from_seq(
        seq_norm,
        window_size=args.window_size,
        stride=args.stride
    )
    print(f"[INFO] windows = {windows_np.shape[0]}, window_size={args.window_size}, stride={args.stride}")

    # ---------- 3. 載入 binary GRU 模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    model = GRUClassifier(num_joints=33, in_channels=3, num_classes=2)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---------- 4. 對每個 window 做預測 ----------
    with torch.no_grad():
        x = torch.from_numpy(windows_np).float().to(device)  # (N, T, K, C)
        logits = model(x)  # (N, 2)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pos_probs = probs[:, 1]  # 類別1 = 拿起/放下 的機率
        pred_windows = (pos_probs >= args.win_pos_thr).astype(np.int32)  # True/False→1/0

    # ---------- 5. 彙整到 frame-level state ----------
    frame_state, frame_prob = aggregate_windows_to_frames(
        pred_windows, starts, T, args.window_size, args.stride,
        frame_pos_thr=args.frame_pos_thr
    )
    # ---------- 6. 找 episodes ----------
    episodes = find_episodes_from_state(frame_state, min_frames=args.min_frames)
    print(f"[INFO] Episodes found = {len(episodes)} (min_frames={args.min_frames}, ~{args.min_frames/args.fps:.2f} s，這裡的秒數是以 video fps 粗估)")

    # ---------- 7. 對每個 episode 計算 KIM 特徵 ----------
    rows = []
    for idx, (s, e) in enumerate(episodes, start=1):
        seg = seq[s:e+1]  # 原始骨架 (未正規化)，KIM 用的是 2D 像素位置

        # skeleton frame → 原影片 frame / time
        s_vid = s * args.sample_stride
        e_vid = e * args.sample_stride
        start_sec = s_vid / args.fps
        end_sec = e_vid / args.fps

        feats = compute_kim_features(seg, fps=args.fps)
        # 注意：這裡的 fps 傳的是你填的 args.fps（30），
        # 若 skeleton 是每 2 幀取 1 幀，feats["duration_sec"] 會是「以 skeleton fps=30」計的，
        # 真實影片秒數還是以 start_sec/end_sec 為準。

        print(f"\n--- Episode {idx} ---")
        Te = e - s + 1
        print(f"Skeleton frames : {s} ~ {e} (Te={Te})")
        print(f"Video frames    : {s_vid} ~ {e_vid}")
        print(f"Time (sec)      : {start_sec:.2f} ~ {end_sec:.2f}")
        print(f"twist_p95       : {feats['twist_p95']:.3f}")
        print(f"twist_ratio>20  : {feats['twist_ratio_over_20']:.3f}")
        print(f"flex_p95        : {feats['flex_p95']:.3f}")
        print(f"flex_ratio>60   : {feats['flex_ratio_over_60']:.3f}")

        row = {
            "episode_idx": idx,
            "start_frame_skel": s,
            "end_frame_skel": e,
            "num_frames_skel": Te,
            # 方便對照原影片：
            "start_frame_video": s_vid,
            "end_frame_video": e_vid,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        row.update(feats)
        rows.append(row)

    # ---------- 8. 存成 CSV ----------
    video_id = os.path.splitext(os.path.basename(args.skeleton_path))[0]
    out_csv = os.path.join(args.out_dir, f"episode_kim_auto_{video_id}_v2.csv")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] Saved episode summary CSV to: {out_csv}")
    else:
        print("[WARN] No episodes found. CSV not written.")


if __name__ == "__main__":
    main()
