# src/episode_kim_demo.py
import os
import numpy as np
import pandas as pd

from kim_features_V2 import compute_kim_features


def load_segments_for_source_from_splits(train_csv, val_csv, source_prefix):
    """
    從 train.csv + val.csv 讀出屬於同一支原始影片 (source_prefix) 的 segments。

    假設 csv 欄位格式：
        video_id, label

    這裡用 video_id 的前綴來判斷來源影片，例如：
        source_prefix = "16_" 對應 "16_seg000_act1", "16_seg001_act0", ...

    回傳：
        僅包含該 source_prefix 的 DataFrame，依 video_id 排序。
    """
    dfs = []
    if os.path.exists(train_csv):
        dfs.append(pd.read_csv(train_csv))
    if os.path.exists(val_csv):
        dfs.append(pd.read_csv(val_csv))

    if not dfs:
        raise FileNotFoundError("No train/val csv found.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    mask = df_all["video_id"].astype(str).str.startswith(source_prefix)
    df_src = df_all[mask].copy()

    df_src = df_src.sort_values("video_id").reset_index(drop=True)
    return df_src


def group_episodes(df_src):
    """
    用 label 序列把 segments group 成 episodes。
    規則（假設 1 = 彎腰/拿起, 0 = 移動, 2 = 放下）：
      - 遇到 label == 1 視為 episode 開始
      - 往後找第一個 label == 2 視為 episode 結束
      - 中間允許有 label == 0

    回傳：
      episodes: list of list，每個元素是一個 episode 內在 df_src 的 row index 列表。
    """
    episodes = []
    i = 0
    n = len(df_src)

    while i < n:
        row = df_src.iloc[i]
        lbl = int(row["label"])
        if lbl == 1:
            start_idx = i
            j = i + 1
            end_idx = None
            while j < n:
                lbl_j = int(df_src.iloc[j]["label"])
                if lbl_j == 2:
                    end_idx = j
                    break
                j += 1
            if end_idx is not None:
                ep_indices = list(range(start_idx, end_idx + 1))
                episodes.append(ep_indices)
                i = end_idx + 1
            else:
                # 之後都沒有放下，就忽略這個未完成 episode
                i += 1
        else:
            i += 1

    return episodes


def load_episode_skeleton(ep_indices, df_src, skeleton_dir):
    """
    把一個 episode 內的多個 segment skeleton 串接起來。

    每個 segment 對應一個 .npy 檔：
        (T_i, 33, 3)
    串接後：
        (sum_i T_i, 33, 3)
    """
    seq_list = []
    for idx in ep_indices:
        row = df_src.iloc[idx]
        vid = row["video_id"]
        npy_path = os.path.join(skeleton_dir, f"{vid}.npy")
        if not os.path.exists(npy_path):
            print(f"[WARN] npy not found, skip segment: {npy_path}")
            continue
        seq = np.load(npy_path)  # (T, 33, 3)
        seq = np.nan_to_num(seq, nan=0.0).astype("float32")
        seq_list.append(seq)

    if not seq_list:
        return None

    return np.concatenate(seq_list, axis=0)  # (T_total, 33, 3)


def main():
    # label 對應說明：1=彎腰/拿起, 0=移動, 2=放下
    id2name = {0: "移動", 1: "拿起", 2: "放下"}

    train_csv = "data/split/train.csv"
    val_csv = "data/split/val.csv"
    skeleton_dir = "data/skeleton_segments_npy"

    # ======= 在這裡設定你要看的原始影片 ID 前綴 =======
    # 例如 segments 為 16_segxxx_act?，就用 "16_"
    source_prefix = "01_"   # TODO: 想看 08_xxx 就改成 "08_"
    # ==================================================

    fps = 30.0

    print(f"[INFO] source_prefix={source_prefix}")
    df_src = load_segments_for_source_from_splits(train_csv, val_csv, source_prefix)
    print(f"[INFO] segments for this source: {len(df_src)}")

    if df_src.empty:
        print("[WARN] No segments found for this source_prefix (請確認前綴是否正確，例如 '16_' 或 '00_').")
        return

    episodes = group_episodes(df_src)
    print(f"[INFO] Found {len(episodes)} episodes for source_prefix={source_prefix!r}")

    if not episodes:
        print("[WARN] No episodes found (可能這支影片沒有完整的「彎腰→放下」序列)")
        return

    # 收集每個 episode 的特徵與簡單解讀，最後輸出成 CSV
    episode_rows = []

    for epi_id, ep_indices in enumerate(episodes, start=1):
        print("\n==============================")
        print(f"Episode {epi_id} (row indices in df_src: {ep_indices})")

        df_ep = df_src.iloc[ep_indices]
        print(df_ep[["video_id", "label"]])

        # 1) 讀取並串接 episode skeleton
        seq_epi = load_episode_skeleton(ep_indices, df_src, skeleton_dir)
        if seq_epi is None:
            print("[WARN] No skeleton for this episode, skip.")
            continue

        T_total = seq_epi.shape[0]
        duration_sec = T_total / fps
        print(f"[INFO] Episode total frames: {T_total}, duration ~ {duration_sec:.2f} s")

        # 2) 計算 KIM-like 特徵
        features = compute_kim_features(seq_epi, fps=fps)
        print("[KIM-like features]")
        for k, v in features.items():
            print(f"  {k:22s}: {v:.3f}")

        # 3) 簡單解讀 trunk flex & twist
        flex_p95 = features.get("flex_p95", 0.0)
        twist_p95 = features.get("twist_p95", 0.0)

        if flex_p95 < 20:
            flex_level = "Low"
        elif flex_p95 < 60:
            flex_level = "Medium"
        else:
            flex_level = "High"

        twist_flag = "High" if twist_p95 > 20 else "Low"

        print("[Simple KIM-style interpretation]")
        print(f"  Trunk flex level   : {flex_level} (p95={flex_p95:.1f}°)")
        print(f"  Twist exposure flag: {twist_flag} (p95={twist_p95:.1f}° > 20°?)")

        # 4) 收集到列表，方便後續用 CSV 分析
        row = {
            "source_prefix": source_prefix,
            "episode_id": epi_id,
            "num_segments": len(ep_indices),
            "segment_indices": ";".join(str(i) for i in ep_indices),
            "first_video_id": df_ep.iloc[0]["video_id"],
            "last_video_id": df_ep.iloc[-1]["video_id"],
            "T_total": T_total,
            "duration_sec": duration_sec,
        }
        # 加入所有 KIM 特徵
        for k, v in features.items():
            row[k] = v

        # 再加上簡單解讀欄位
        row["flex_level"] = flex_level
        row["twist_flag"] = twist_flag

        episode_rows.append(row)

    # 5) 全部 episodes 處理完後，輸出 CSV
    if episode_rows:
        out_dir = "analysis"
        os.makedirs(out_dir, exist_ok=True)
        prefix_clean = source_prefix.strip("_")
        out_csv = os.path.join(out_dir, f"episode_kim_summary_{prefix_clean}.csv")
        df_out = pd.DataFrame(episode_rows)
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] Saved episode summary CSV to: {out_csv}")
    else:
        print("[INFO] No episode rows to save (all episodes skipped or had no skeleton).")


if __name__ == "__main__":
    main()

