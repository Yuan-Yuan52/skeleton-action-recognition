#模型訓練/驗證集切割程式
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_csv", type=str, default="data/segments.csv")
    parser.add_argument("--train_csv", type=str, default="data/split/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/split/val.csv")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="驗證集佔原始影片數量的比例，例如 0.2")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.train_csv), exist_ok=True)

    df = pd.read_csv(args.segments_csv)

    # 以 source_video 為單位做 group split
    videos = df["source_video"].unique()
    train_videos, val_videos = train_test_split(
        videos,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    train_df = df[df["source_video"].isin(train_videos)].copy()
    val_df = df[df["source_video"].isin(val_videos)].copy()

    # 只保留 video_id + label 欄位，符合我們 SkeletonDataset 的輸入格式
    train_out = train_df[["video_id", "label"]]
    val_out = val_df[["video_id", "label"]]

    # 儲存
    train_out.to_csv(args.train_csv, index=False)
    val_out.to_csv(args.val_csv, index=False)

    print(f"[INFO] Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
    print(f"[INFO] Train segments: {len(train_out)}, Val segments: {len(val_out)}")
    print(f"[INFO] Saved train csv to: {args.train_csv}")
    print(f"[INFO] Saved val csv to  : {args.val_csv}")

if __name__ == "__main__":
    main()