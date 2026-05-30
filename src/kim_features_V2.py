# src/kim_features.py
import numpy as np

# Mediapipe Pose 33 點索引
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

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

def compute_hand_arm_features(seq, eps=1e-6):
    """
    給一段骨架 seq: (T, 33, 3)
    回傳:
      hand_dist_norm : (T,) 手到軀幹中心的距離 / 軀幹長度
      arm_mid_mask   : (T,) bool, 手在「手肘與肩膀高度之間」
      arm_high_mask  : (T,) bool, 手高於肩膀
    """
    seq = np.asarray(seq, dtype=np.float32)
    pts = seq[:, :, :2]
    T = pts.shape[0]

    hand_dist_norm = np.zeros(T, dtype=np.float32)
    arm_mid_mask = np.zeros(T, dtype=bool)
    arm_high_mask = np.zeros(T, dtype=bool)

    for t in range(T):
        f = pts[t]

        ls, rs = f[LEFT_SHOULDER], f[RIGHT_SHOULDER]
        lh, rh = f[LEFT_HIP], f[RIGHT_HIP]
        le, re = f[LEFT_ELBOW], f[RIGHT_ELBOW]
        lw, rw = f[LEFT_WRIST], f[RIGHT_WRIST]

        shoulder_c = (ls + rs) / 2.0
        hip_c = (lh + rh) / 2.0
        torso_len = np.linalg.norm(shoulder_c - hip_c)
        if torso_len < eps:
            torso_len = eps

        torso_center = (shoulder_c + hip_c) / 2.0

        d_l = np.linalg.norm(lw - torso_center) / torso_len
        d_r = np.linalg.norm(rw - torso_center) / torso_len
        hand_dist_norm[t] = max(d_l, d_r)  # 取較嚴重那側

        # y 向下為正
        y_sh = shoulder_c[1]
        y_el = min(le[1], re[1])        # 較高的那個手肘
        y_hand = min(lw[1], rw[1])      # 較高的那隻手

        # 手高於肩膀
        if y_hand < y_sh:
            arm_high_mask[t] = True

        # 手在「手肘與肩膀高度之間」
        # 用「介於 y_sh 跟 y_el 之間」來近似，不管誰比較高
        y_min = min(y_sh, y_el)
        y_max = max(y_sh, y_el)
        if y_min <= y_hand <= y_max:
            arm_mid_mask[t] = True

    return hand_dist_norm, arm_mid_mask, arm_high_mask

def compute_kim_features(seq, fps=30.0,
                         flex_thr_low=20.0, flex_thr_high=60.0,
                         twist_thr=20.0):
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    duration_sec = T / float(fps) if fps > 0 else 0.0

    twist_deg, flex_deg = compute_twist_flex(seq)
    twist_deg = np.nan_to_num(twist_deg, nan=0.0, posinf=0.0, neginf=0.0)
    flex_deg = np.nan_to_num(flex_deg, nan=0.0, posinf=0.0, neginf=0.0)

    hand_dist_norm, arm_mid_mask, arm_high_mask = compute_hand_arm_features(seq)

    if T == 0:
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
        "hand_dist_p95": 0.0,
        "hand_ratio_far": 0.0,
        "arm_mid_ratio": 0.0,
        "arm_high_ratio": 0.0,
        "extra_score": 0.0,
    }  # 你原來的空值處理

    # --- 原本的扭轉/前屈統計 ---
    twist_p95 = float(np.percentile(twist_deg, 95))
    twist_max = float(twist_deg.max())
    twist_ratio_over_thr = float((twist_deg > twist_thr).sum()) / T

    flex_p95 = float(np.percentile(flex_deg, 95))
    flex_max = float(flex_deg.max())
    flex_mask_20_60 = (flex_deg >= flex_thr_low) & (flex_deg < flex_thr_high)
    flex_mask_over_60 = flex_deg >= flex_thr_high
    flex_ratio_20_60 = float(flex_mask_20_60.sum()) / T
    flex_ratio_over_60 = float(flex_mask_over_60.sum()) / T

    # --- 新增：手距離 & 手高度統計 ---
    hand_dist_p95 = float(np.percentile(hand_dist_norm, 95))
    hand_ratio_far = float((hand_dist_norm > 0.4).sum()) / T  # 0.4 可再調
    arm_mid_ratio = float(arm_mid_mask.sum()) / T
    arm_high_ratio = float(arm_high_mask.sum()) / T

    # --- 依照比例換成額外加分 ---
    extra_score = 0.0

    # 1) 軀幹扭轉/側傾 +1 / +3
    if twist_ratio_over_thr >= 0.3:
        extra_score += 3.0
    elif twist_ratio_over_thr >= 0.05:
        extra_score += 1.0

    # 2) 負重重心或手遠離身體 +1 / +3
    if hand_ratio_far >= 0.3:
        extra_score += 3.0
    elif hand_ratio_far >= 0.05:
        extra_score += 1.0

    # 3) 手在手肘與肩膀之間 +0.5 / +1
    if arm_mid_ratio >= 0.3:
        extra_score += 1.0
    elif arm_mid_ratio >= 0.05:
        extra_score += 0.5

    # 4) 手高過肩膀 +1 / +2
    if arm_high_ratio >= 0.3:
        extra_score += 2.0
    elif arm_high_ratio >= 0.05:
        extra_score += 1.0

    extra_score = min(extra_score, 6.0)

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
        "hand_dist_p95": hand_dist_p95,
        "hand_ratio_far": hand_ratio_far,
        "arm_mid_ratio": arm_mid_ratio,
        "arm_high_ratio": arm_high_ratio,
        "extra_score": extra_score,
    }

    return features
