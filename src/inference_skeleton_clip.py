#模型推論程式

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import argparse
import numpy as np
import torch

from models_skeleton import GRUClassifier


def make_windows(seq, window_size=32, stride=16):
    """
    seq: (T, K, C)
    回傳 windows: (N, window_size, K, C)
    如果 T < window_size，會做 zero-padding 變成一個 window。
    """
    T, K, C = seq.shape

    if T >= window_size:
        windows = []
        t = 0
        while t + window_size <= T:
            windows.append(seq[t:t + window_size])
            t += stride
        windows = np.stack(windows, axis=0)  # (N, T, K, C)
    else:
        # 太短的影片：pad 到 window_size
        pad_len = window_size - T
        pad = np.zeros((pad_len, K, C), dtype=seq.dtype)
        window = np.concatenate([seq, pad], axis=0)  # (window_size, K, C)
        windows = window[None, ...]  # (1, T, K, C)

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
    parser.add_argument("--video_id", type=str, required=True,
                        help="要推論的段名，例如 00_seg000_act1（不含 .npy）")
    parser.add_argument("--ckpt", type=str, default="checkpoints/skeleton_gru/best.pth",
                        help="訓練好的模型權重路徑")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(">>> Using device:", device)

    # 1. 載入骨架序列
    npy_path = os.path.join(args.skeleton_dir, f"{args.video_id}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Skeleton npy not found: {npy_path}")

    seq = np.load(npy_path)  # (T, 33, 3)
    seq = np.nan_to_num(seq, nan=0.0).astype("float32")
    T, K, C = seq.shape
    print(f"[INFO] Loaded skeleton: {npy_path}, shape={seq.shape}")

    # 2. 切成時間窗口
    windows = make_windows(seq, window_size=args.window_size, stride=args.stride)
    print(f"[INFO] Num windows: {windows.shape[0]}, window_size={args.window_size}")

    # 3. 載入 model
    model = load_model(args.ckpt, num_classes=args.num_classes, device=device)

    # 4. 對每個 window 做推論
    with torch.no_grad():
        x = torch.from_numpy(windows)  # (N, T, K, C)
        x = x.to(device)
        logits = model(x)              # (N, num_classes)
        probs = torch.softmax(logits, dim=1)  # (N, num_classes)

    # 5. 將多個 window 的結果平均，做成 clip-level 預測
    probs_clip = probs.mean(dim=0)          # (num_classes,)
    pred_class = int(torch.argmax(probs_clip).item())
    probs_clip_np = probs_clip.cpu().numpy()

    # 你可以在這裡定義自己的 label 名稱對應（先給一個 placeholder）
    id2name = {
        0: "移動",
        1: "拿起",
        2: "放下",
        # 如果之後有第 3 類、4 類再自己補
    }
    pred_name = id2name.get(pred_class, f"class_{pred_class}")

    print("\n========== Inference Result ==========")
    print(f"Video ID      : {args.video_id}")
    print(f"Frames (T)    : {T}")
    print(f"Num windows   : {windows.shape[0]}")
    print(f"Pred class id : {pred_class}")
    print(f"Pred name     : {pred_name}")
    print("Probabilities :")
    for c in range(args.num_classes):
        name = id2name.get(c, f"class_{c}")
        print(f"  class {c} ({name}): {probs_clip_np[c]:.4f}")

    # 如果你想看每個 window 的預測，也可以印一下：
    window_preds = torch.argmax(probs, dim=1).cpu().numpy()
    print("\nPer-window predictions (first 20 windows):")
    print(window_preds[:20])


if __name__ == "__main__":
    main()
