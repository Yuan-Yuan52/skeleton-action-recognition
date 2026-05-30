# 把「val 上所有錯誤的 segment 列出來」
# src/dump_misclassified_segments.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OpenMP 重複載入錯誤

import argparse
import numpy as np
import pandas as pd
import torch

from models_skeleton import GRUClassifier


def make_windows(seq, window_size=32, stride=16):
    """
    seq: (T, K, C)
    回傳 windows: (N, window_size, K, C)
    如果 T < window_size，會 pad 成一個 window。
    """
    T, K, C = seq.shape

    if T >= window_size:
        windows = []
        t = 0
        while t + window_size <= T:
            windows.append(seq[t:t + window_size])
            t += stride
        windows = np.stack(windows, axis=0)
    else:
        pad_len = window_size - T
        pad = np.zeros((pad_len, K, C), dtype=seq.dtype)
        window = np.concatenate([seq, pad], axis=0)
        windows = window[None, ...]
    return windows


def load_model(ckpt_path, num_classes=3, device="cpu"):
    model = GRUClassifier(num_joints=33, in_channels=3,
                          num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_dir", type=str, default="data/skeleton_segments_npy",
                        help="存放各段骨架 .npy 的資料夾")
    parser.add_argument("--val_csv", type=str, default="data/split/val.csv",
                        help="validation 段列表（video_id, label）")
    parser.add_argument("--ckpt", type=str, default="checkpoints/skeleton_gru/best.pth",
                        help="訓練好的模型權重")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out_csv", type=str, default="misclassified_val.csv",
                        help="把 misclassified 段存成一個 csv，方便之後查閱")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(">>> Using device:", device)

    # 動作名稱（依照你自己的定義調整）
    id2name = {
        0: "移動",
        1: "拿起",
        2: "放下",
    }

    # 1. 載入 val 列表 & 模型
    df = pd.read_csv(args.val_csv)
    print(f"[INFO] Loaded val csv: {args.val_csv}, segments={len(df)}")

    model = load_model(args.ckpt, num_classes=args.num_classes, device=device)
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")

    mis_list = []
    total = 0
    correct = 0

    for idx, row in df.iterrows():
        video_id = row["video_id"]
        true_label = int(row["label"])

        npy_path = os.path.join(args.skeleton_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            print(f"[WARN] Skeleton not found, skip: {npy_path}")
            continue

        seq = np.load(npy_path)  # (T, 33, 3)
        seq = np.nan_to_num(seq, nan=0.0).astype("float32")

        windows = make_windows(seq,
                               window_size=args.window_size,
                               stride=args.stride)

        with torch.no_grad():
            x = torch.from_numpy(windows).to(device)   # (N, T, K, C)
            logits = model(x)                          # (N, C)
            probs = torch.softmax(logits, dim=1)       # (N, C)

        probs_clip = probs.mean(dim=0)                 # (C,)
        probs_clip_np = probs_clip.cpu().numpy()
        pred_label = int(torch.argmax(probs_clip).item())
        pred_conf = float(probs_clip[pred_label].item())

        total += 1
        if pred_label == true_label:
            correct += 1
        else:
            mis_list.append({
                "video_id": video_id,
                "true_label": true_label,
                "true_name": id2name.get(true_label, f"class_{true_label}"),
                "pred_label": pred_label,
                "pred_name": id2name.get(pred_label, f"class_{pred_label}"),
                "pred_conf": pred_conf,
                "prob_class0": probs_clip_np[0] if args.num_classes > 0 else None,
                "prob_class1": probs_clip_np[1] if args.num_classes > 1 else None,
                "prob_class2": probs_clip_np[2] if args.num_classes > 2 else None,
                "num_windows": windows.shape[0],
                "num_frames": seq.shape[0],
            })

    acc = correct / total if total > 0 else 0.0
    print(f"\n[RESULT] Val segments total = {total}")
    print(f"[RESULT] Correct            = {correct}")
    print(f"[RESULT] Misclassified      = {len(mis_list)}")
    print(f"[RESULT] ValAcc (recomputed)= {acc:.4f}")

    if mis_list:
        out_df = pd.DataFrame(mis_list)
        out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved misclassified list to: {args.out_csv}")
        print("[INFO] Example rows:")
        print(out_df.head())
    else:
        print("[INFO] No misclassified segments in val set! 🎉")


if __name__ == "__main__":
    main()
