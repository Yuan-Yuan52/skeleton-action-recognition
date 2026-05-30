# src/kim_scoring.py
import argparse
import os
import pandas as pd

# 這裡是「開始/結束 4 類」對應的 KIM trunk 分數表
# ※ 現在的數字只是示意，請你根據手上的 KIM 量表調整！
KIM_TRUNK_SCORE = {
    0: {  # start_class4 = 1：從桌子拿起的正確姿勢
        0: 0,  # 1->1：桌→桌
        1: 5,  # 1->2：桌→椅
        2: 9,  # 1->3：桌→地板(站著重度彎)
        3: 7,  # 1->4：桌→地板(蹲下輕微彎)
    },
    1: {  # start_class4 = 2：從椅子拿取的輕微彎取
        0: 3,  # 2->1：椅→桌
        1: 5,  # 2->2：椅→椅
        2: 13,  # 2->3：椅→地板(站著重度彎)
        3: 10,  # 2->4：椅→地板(蹲下輕微彎)
    },
    2: {  # start_class4 = 3：不蹲直接到地板的重度彎
        0: 9,  # 3->1：地板(重度彎)→桌
        1: 13,  # 3->2：地板(重度彎)→椅
        2: 20,  # 3->3：地板(重度彎)→地板(重度彎)
        3: 18,  # 3->4：地板(重度彎)→地板(蹲下)
    },
    3: {  # start_class4 = 4：蹲下放置地板(輕微彎)
        0: 7,  # 4->1：地板(蹲)→桌
        1: 10,  # 4->2：地板(蹲)→椅
        2: 18,  # 4->3：地板(蹲)→地板(重度彎)
        3: 15,  # 4->4：地板(蹲)→地板(蹲)
    },
}


def kim_trunk_score_from_class4(start_class4: int, end_class4: int) -> float:
    """用 (start_class4, end_class4) 查表，回傳 KIM trunk score。"""
    s = int(start_class4)
    e = int(end_class4)

    if s not in KIM_TRUNK_SCORE or e not in KIM_TRUNK_SCORE[s]:
        # 不合法或缺 label 的情況，可以先給 0 或 NaN
        return 0.0

    return float(KIM_TRUNK_SCORE[s][e])

def compute_kim_total_from_row(row, low_thr=12.0, high_thr=18.0):
    """
    給一列 episode（pandas.Series），計算：
      - kim_trunk_score : 左半邊姿勢評級分數（0,3,5,7,9）
      - kim_extra_score : 右半邊額外加分（extra_score，如果沒有就當 0）
      - kim_total_score : 兩者相加
      - kim_risk_band   : Low / Medium / High（三級分）

    low_thr, high_thr 用來切分風險等級，之後你可以依實驗結果調整。
    """
    # 1) 左半邊：開始/結束 4 類對照到 KIM 姿勢評級分數
    trunk = kim_trunk_score_from_class4(row["start_class4"], row["end_class4"])

    # 2) 右半邊：額外加分，直接用 kim_features.py 裡算好的 extra_score
    extra = float(row.get("extra_score", 0.0))

    # 3) 總分
    total = trunk + extra

    # 4) 風險等級（可調）
    if total < low_thr:
        band = "Low"
    elif total < high_thr:
        band = "Medium"
    else:
        band = "High"

    return trunk, extra, total, band



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/episode_kim_auto_16_v2.csv",
        help="episode KIM CSV 路徑（要包含 start_class4 / end_class4 / extra_score）"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="輸出 CSV 路徑 (預設在檔名後面加 _scored)"
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 檢查必備欄位
    for col in ["start_class4", "end_class4"]:
        if col not in df.columns:
            raise ValueError(f"CSV 裡找不到欄位: '{col}'，請確認你已經標好。")

    # 逐列計算分數
    trunk_list = []
    extra_list = []
    total_list = []
    band_list = []

    for _, row in df.iterrows():
        trunk, extra, total, band = compute_kim_total_from_row(row)
        trunk_list.append(trunk)
        extra_list.append(extra)
        total_list.append(total)
        band_list.append(band)

    df["kim_trunk_score"] = trunk_list      # 左半邊 0/3/5/7/9
    df["kim_extra_score"] = extra_list      # 右半邊 extra_score
    df["kim_total_score"] = total_list      # 加總
    df["kim_risk_band"] = band_list         # Low / Medium / High

    # 決定輸出檔名
    if args.out_csv is None:
        root, ext = os.path.splitext(csv_path)
        out_csv = root + "_scored" + ext
    else:
        out_csv = args.out_csv

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved scored CSV to: {out_csv}")

    # 簡單印一下統計
    print("\n[INFO] kim_trunk_score 分佈：")
    print(df["kim_trunk_score"].value_counts().sort_index())

    if "kim_risk_band" in df.columns:
        print("\n[INFO] kim_risk_band 分佈：")
        print(df["kim_risk_band"].value_counts())


if __name__ == "__main__":
    main()