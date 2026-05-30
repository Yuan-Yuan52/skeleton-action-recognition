# src/kim_features.py
import numpy as np

# Mediapipe Pose 33 點索引
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def compute_twist_flex(seq, eps=1e-6):
    """
    計算一段骨架序列的「2D 扭轉角」與「前屈角」。

    參數
    ----
    seq : np.ndarray, shape (T, 33, 3)
        (x, y, conf)，通常已做過平移/尺度正規化。

    回傳
    ----
    twist_deg : np.ndarray, shape (T,)
        每一幀的「肩寬線 vs 髖寬線」夾角（0~180 度）。
        這是在平面上的近似扭轉，真正軸向扭轉要 3D 才最準。
    flex_deg : np.ndarray, shape (T,)
        每一幀的「髖中心→肩中心」與「垂直向上」的夾角（0=直立，越大越前屈/後仰）。
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, K, C = seq.shape
    pts = seq[:, :, :2]  # 只用 x, y

    twist_deg = np.zeros(T, dtype=np.float32)
    flex_deg = np.zeros(T, dtype=np.float32)

    for t in range(T):
        frame = pts[t]  # (33, 2)

        # 取四個關鍵點
        ls = frame[LEFT_SHOULDER]   # 左肩
        rs = frame[RIGHT_SHOULDER]  # 右肩
        lh = frame[LEFT_HIP]        # 左髖
        rh = frame[RIGHT_HIP]       # 右髖

        # -------- 扭轉：肩寬線 vs 髖寬線 --------
        sh_vec = ls - rs   # 肩寬向量
        hip_vec = lh - rh  # 髖寬向量

        sh_norm = np.linalg.norm(sh_vec)
        hip_norm = np.linalg.norm(hip_vec)

        if sh_norm > eps and hip_norm > eps:
            cos_angle = np.dot(sh_vec, hip_vec) / (sh_norm * hip_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))  # 0~180
            angle = min(angle, 180.0 - angle)         # <<< 新增：摺到 0~90
            twist_deg[t] = angle
        else:
            twist_deg[t] = 0.0


        # -------- 前屈：髖中心 → 肩中心 vs 垂直向上 --------
        hip_center = (lh + rh) / 2.0
        sh_center = (ls + rs) / 2.0
        torso_vec = sh_center - hip_center  # (dx, dy)

        torso_norm = np.linalg.norm(torso_vec)
        if torso_norm > eps:
            # 影像座標 y 向下為正，因此「身體直立向上」我們用 (0, -1)
            v_up = np.array([0.0, -1.0], dtype=np.float32)
            cos_flex = np.dot(torso_vec / torso_norm, v_up)
            cos_flex = np.clip(cos_flex, -1.0, 1.0)
            angle_flex = np.degrees(np.arccos(cos_flex))  # 0=直立，越大越彎
            flex_deg[t] = angle_flex
        else:
            flex_deg[t] = 0.0

    return twist_deg, flex_deg


def compute_kim_features(seq, fps=30.0,
                         flex_thr_low=20.0, flex_thr_high=60.0,
                         twist_thr=20.0):
    """
    給一整段骨架，計算一些類似 KIM-LHC 會用到的姿勢特徵。

    參數
    ----
    seq : np.ndarray, shape (T, 33, 3)
        該段的骨架序列。
    fps : float
        該段影片的影格率，用來換算秒數。
    flex_thr_low : float
        判斷「開始算前屈」的低門檻 (deg)，預設 20 度。
    flex_thr_high : float
        高前屈門檻 (deg)，預設 60 度。
    twist_thr : float
        扭轉門檻 (deg)，預設 20 度。

    回傳
    ----
    features : dict
        例如：
        {
          "T": 50,
          "duration_sec": 1.67,
          "twist_p95":  22.3,
          "twist_max":  35.1,
          "twist_ratio_over_20": 0.42,
          "flex_p95":   65.8,
          "flex_max":   78.2,
          "flex_ratio_20_60": 0.30,
          "flex_ratio_over_60": 0.12,
        }
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    duration_sec = T / float(fps) if fps > 0 else 0.0

    twist_deg, flex_deg = compute_twist_flex(seq)

    # 安全處理 NaN / inf
    twist_deg = np.nan_to_num(twist_deg, nan=0.0, posinf=0.0, neginf=0.0)
    flex_deg = np.nan_to_num(flex_deg, nan=0.0, posinf=0.0, neginf=0.0)

    if T == 0:
        # 空段保險 return
        return {
            "T": 0,
            "duration_sec": 0.0,
            "twist_p95": 0.0,
            "twist_max": 0.0,
            "twist_ratio_over_20": 0.0,
            "flex_p95": 0.0,
            "flex_max": 0.0,
            "flex_ratio_20_60": 0.0,
            "flex_ratio_over_60": 0.0,
        }

    # ---- 基本統計 ----
    twist_p95 = float(np.percentile(twist_deg, 95))
    twist_max = float(twist_deg.max())
    flex_p95 = float(np.percentile(flex_deg, 95))
    flex_max = float(flex_deg.max())

    # ---- 暴露比例（占整段時間的比例）----
    # 扭轉 > twist_thr 的比例
    twist_ratio_over_thr = float((twist_deg > twist_thr).sum()) / T

    # 前屈 20~60 之間的比例、>60 的比例
    flex_mask_20_60 = (flex_deg >= flex_thr_low) & (flex_deg < flex_thr_high)
    flex_mask_over_60 = flex_deg >= flex_thr_high

    flex_ratio_20_60 = float(flex_mask_20_60.sum()) / T
    flex_ratio_over_60 = float(flex_mask_over_60.sum()) / T

    features = {
        "T": T,
        "duration_sec": duration_sec,
        "twist_p95": twist_p95,
        "twist_max": twist_max,
        "twist_ratio_over_20": twist_ratio_over_thr,
        "flex_p95": flex_p95,
        "flex_max": flex_max,
        "flex_ratio_20_60": flex_ratio_20_60,
        "flex_ratio_over_60": flex_ratio_over_60,
    }

    return features
