# src/extract_skeleton.py
import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_pose = mp.solutions.pose

def normalize_skeleton(seq):
    """
    seq: (T, 33, 3) with (x, y, conf) in image coordinates [0,1]
    步驟：
    1. 以左右髖關節平均當 root，做平移
    2. 以肩寬當尺度，做尺度正規化
    """
    T, K, C = seq.shape
    norm_seq = seq.copy()

    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    # 平移：減掉髖關節中心
    hips = (norm_seq[:, LEFT_HIP, :2] + norm_seq[:, RIGHT_HIP, :2]) / 2.0
    norm_seq[:, :, :2] -= hips[:, None, :]

    # 尺度：用肩寬
    shoulder_width = np.linalg.norm(
        norm_seq[:, LEFT_SHOULDER, :2] - norm_seq[:, RIGHT_SHOULDER, :2],
        axis=1,
    )
    if np.any(shoulder_width > 1e-4):
        scale = np.median(shoulder_width[shoulder_width > 1e-4])
    else:
        scale = 1.0
    norm_seq[:, :, :2] /= (scale + 1e-6)

    return norm_seq

def extract_from_video(video_path, out_path, sample_stride=2):
    """
    video_path: 輸入影片路徑
    out_path: 輸出 .npy 路徑
    sample_stride: 每幾幀取一幀 (控制時間長度)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_idx = 0

    with pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 等距抽幀
            if frame_idx % sample_stride != 0:
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks is None:
                # 偵測不到時，用 NaN 填
                kp = np.full((33, 3), np.nan, dtype=np.float32)
            else:
                kp = []
                for lm in result.pose_landmarks.landmark:
                    kp.append([lm.x, lm.y, lm.visibility])
                kp = np.array(kp, dtype=np.float32)

            frames.append(kp)
            frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted: {video_path}")

    seq = np.stack(frames, axis=0)      # (T_raw, 33, 3)
    seq_norm = normalize_skeleton(seq)  # 做平移 + 尺度正規化

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, seq_norm)
    print(f"Saved skeleton to {out_path}, shape={seq_norm.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_stride", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    videos = [
        f for f in os.listdir(args.video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    for v in tqdm(videos, desc="Processing videos"):
        vpath = os.path.join(args.video_dir, v)
        vid = os.path.splitext(v)[0]  # 去掉副檔名
        out_path = os.path.join(args.out_dir, f"{vid}.npy")
        extract_from_video(vpath, out_path, sample_stride=args.sample_stride)

if __name__ == "__main__":
    main()
