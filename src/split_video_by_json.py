# 長片段切割程式
import os
import json
import argparse
import csv
import cv2
from glob import glob

def split_video_by_annotations(video_path, ann_path, out_dir, writer_rows):
    """
    video_path: 對應的原始影片，如 data/videos/00.mp4
    ann_path  : 對應的標註 json，如 data/annotations/00_annotations.json
    out_dir   : 輸出分段影片的資料夾
    writer_rows: list，累積寫入 segments.csv 的 row
    """
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]  # "00"
    print(f"Processing {base_video_name} ...")

    # 讀 json
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 這裡假設格式跟你上傳的範例一樣：list 裡只有一個 dict
    actions = data[0]["Actions"]
    # 依 Start Frame 排序（保險）
    actions = sorted(actions, key=lambda x: x["Start Frame"])

    # 開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video FPS={fps}, size=({w}x{h}), segments={len(actions)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 輸出 mp4

    for idx, seg in enumerate(actions):
        start_f = int(seg["Start Frame"])
        end_f = int(seg["End Frame"])
        action_id = int(seg["Action"])

        # 如果你的標註是「1-based frame index」，想要從 0 開始，可以改成：
        # start_f = start_f - 1
        # end_f   = end_f - 1

        # 每一段輸出檔名：00_seg000_act1.mp4
        seg_name = f"{base_video_name}_seg{idx:03d}_act{action_id}"
        out_path = os.path.join(out_dir, seg_name + ".mp4")

        print(f"    Segment {idx:03d}: frames [{start_f}, {end_f}], action={action_id}")

        # 跳到起始 frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        frame_idx = start_f

        while frame_idx <= end_f:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1

        writer.release()

        # 記錄到 csv 行
        writer_rows.append({
            "video_id": seg_name,
            "label": action_id,
            "source_video": base_video_name,
            "start_frame": start_f,
            "end_frame": end_f,
        })

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="data/videos",
                        help="原始長影片所在資料夾，例如 data/videos")
    parser.add_argument("--ann_dir", type=str, required=True,
                        help="json 標註所在資料夾，例如 data/annotations")
    parser.add_argument("--out_dir", type=str, default="data/segments",
                        help="輸出分段影片的資料夾，例如 data/segments")
    parser.add_argument("--video_ext", type=str, default=".mp4",
                        help="原始影片的副檔名，例如 .mp4")
    parser.add_argument("--csv_path", type=str, default="data/segments.csv",
                        help="輸出的 segments 標註 csv 路徑")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    # 找出所有 *_annotations.json
    json_paths = sorted(glob(os.path.join(args.ann_dir, "*_annotations.json")))
    if not json_paths:
        print(f"[ERROR] No *_annotations.json found in {args.ann_dir}")
        return

    rows = []

    for ann_path in json_paths:
        ann_name = os.path.basename(ann_path)          # 00_annotations.json
        base = os.path.splitext(ann_name)[0]           # 00_annotations
        # 把 "_annotations" 拿掉，得到影片的 base name，例如 "00"
        video_base = base.replace("_annotations", "")
        video_path = os.path.join(args.video_dir, video_base + args.video_ext)

        if not os.path.exists(video_path):
            print(f"[WARN] Video not found for {ann_name}: {video_path}")
            continue

        split_video_by_annotations(video_path, ann_path, args.out_dir, rows)

    # 寫出 segments.csv
    if rows:
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video_id", "label", "source_video", "start_frame", "end_frame"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved segments csv to: {args.csv_path}")
    else:
        print("[WARN] No segments written, please check annotations.")

if __name__ == "__main__":
    main()
