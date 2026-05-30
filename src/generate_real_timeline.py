import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NTU/stgcn')))
from net.st_gcn import Model as STGCN

sys.path.append(os.path.dirname(__file__))
from models_transformer import SpatialTemporalTransformer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def normalize_skeleton(seq, eps=1e-6):
    """
    Normalizes skeleton frames based on shoulder/hip center and scale.
    seq: (T, 33, 3) with (x, y, visibility)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, K, C = seq.shape
    coords = seq.copy()

    for t in range(T):
        frame = coords[t]
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

        ls = frame[LEFT_SHOULDER, :2]
        rs = frame[RIGHT_SHOULDER, :2]
        lh = frame[LEFT_HIP, :2]
        rh = frame[RIGHT_HIP, :2]

        center = (ls + rs + lh + rh) / 4.0
        frame[:, :2] -= center

        shoulder_w = np.linalg.norm(ls - rs)
        hip_w = np.linalg.norm(lh - rh)
        scale = shoulder_w + hip_w
        if scale < eps:
            scale = 1.0
        frame[:, :2] /= scale
        coords[t] = frame
    return coords

def resample_seq(seq, out_len):
    """
    Resamples a sequence to a fixed length out_len.
    """
    T = seq.shape[0]
    if T == out_len:
        return seq
    if T > out_len:
        idxs = np.linspace(0, T - 1, num=out_len, dtype=int)
        return seq[idxs]
    else:
        pad_len = out_len - T
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        return np.concatenate([seq, pad], axis=0)

def main():
    video_path = "C:/Users/r13941031/Desktop/cam1_live_20260409_170422.mp4"
    skel_cache_path = "data/cam1_skeletons.npy"
    sample_stride = 2
    
    # 1. Extract or load skeletons
    if os.path.exists(skel_cache_path):
        print(f"Loading skeletons from cache: {skel_cache_path}")
        skeletons = np.load(skel_cache_path)
    else:
        print("Extracting skeletons using MediaPipe Pose...")
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return
            
        frames = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with pose:
            for _ in tqdm(range(total_frames), desc="Running MediaPipe Pose"):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_stride != 0:
                    frame_idx += 1
                    continue
                    
                # Mirror flip just like in live_cam_onnx_sqlite.py
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)
                
                if result.pose_landmarks is None:
                    kp = np.zeros((33, 3), dtype=np.float32)
                else:
                    kp = []
                    for lm in result.pose_landmarks.landmark:
                        kp.append([lm.x, lm.y, getattr(lm, "visibility", 0.0)])
                    kp = np.array(kp, dtype=np.float32)
                frames.append(kp)
                frame_idx += 1
        cap.release()
        skeletons = np.stack(frames, axis=0) # (T_skel, 33, 3)
        os.makedirs("data", exist_ok=True)
        np.save(skel_cache_path, skeletons)
        print(f"Saved skeletons to cache: {skel_cache_path}, shape={skeletons.shape}")

    # Skeletons shape is (T, 33, 3). Let's pad/clean NaNs
    skeletons = np.nan_to_num(skeletons, nan=0.0, posinf=0.0, neginf=0.0)
    T = skeletons.shape[0]
    
    # Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    
    # 2. Load Robust Transformer
    transformer_model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8, 
        num_spatial_layers=3, num_temporal_layers=4, num_classes=num_classes, dropout=0.0
    )
    transformer_ckpt = torch.load("checkpoints/robust_transformer_model/lift_class_id/best.pth", map_location=device)
    if "state_dict" in transformer_ckpt:
        transformer_model.load_state_dict(transformer_ckpt["state_dict"])
    else:
        transformer_model.load_state_dict(transformer_ckpt)
    transformer_model.to(device).eval()
    
    # 3. Load ST-GCN
    stgcn_model = STGCN(
        in_channels=3, num_class=num_classes, 
        graph_args={"layout": "coco", "strategy": "spatial"}, edge_importance_weighting=True
    )
    stgcn_ckpt = torch.load("checkpoints/stgcn_v2/lift_class_id/best.pth", map_location=device)
    if "state_dict" in stgcn_ckpt:
        stgcn_model.load_state_dict(stgcn_ckpt["state_dict"])
    else:
        stgcn_model.load_state_dict(stgcn_ckpt)
    stgcn_model.to(device).eval()
    
    # Joint mapping definitions
    CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # 13 joints
    coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # 17 joints
    
    # Run frame-by-frame predictions
    transformer_preds = []
    stgcn_preds = []
    
    # Window settings: length 64 frames (with stride=2, it represents 128 video frames)
    # Or window length 32 resampled to 64? Let's use 32 frames (64 video frames) to make it more responsive.
    window_size = 32
    
    print("Running inference over skeletons...")
    with torch.no_grad():
        for t in range(T):
            # Take window ending at t
            start_idx = max(0, t - window_size + 1)
            win_skel = skeletons[start_idx:t+1]
            # Normalization
            win_skel_norm = normalize_skeleton(win_skel)
            # Resample to 64
            win_skel_rs = resample_seq(win_skel_norm, 64)
            
            # --- Spatio-Temporal Transformer Inference ---
            trans_in = win_skel_rs[:, CORE_JOINTS, :] # (64, 13, 3)
            trans_tensor = torch.from_numpy(trans_in).unsqueeze(0).float().to(device) # (1, 64, 13, 3)
            logits_trans = transformer_model(trans_tensor)
            pred_trans = torch.argmax(logits_trans, dim=-1).item()
            transformer_preds.append(pred_trans)
            
            # --- ST-GCN Inference ---
            stgcn_in = win_skel_rs[:, coco_mapping, :] # (64, 17, 3)
            stgcn_tensor = torch.from_numpy(stgcn_in).float().to(device) # (64, 17, 3)
            # Reshape (64, 17, 3) -> (3, 64, 17) -> (1, 3, 64, 17, 1)
            stgcn_tensor = stgcn_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
            logits_stgcn = stgcn_model(stgcn_tensor)
            pred_stgcn = torch.argmax(logits_stgcn, dim=-1).item()
            stgcn_preds.append(pred_stgcn)
            
    print(f"Inference complete! Length={len(transformer_preds)}")
    
    # Let's count predictions to see what was predicted
    trans_counts = np.bincount(transformer_preds)
    stgcn_counts = np.bincount(stgcn_preds)
    print("Transformer class counts:", {i: c for i, c in enumerate(trans_counts)})
    print("STGCN class counts:", {i: c for i, c in enumerate(stgcn_counts)})
    
    # Save the raw predictions
    np.savez("data/real_timeline_predictions.npz", 
             transformer=np.array(transformer_preds), 
             stgcn=np.array(stgcn_preds))
    print("Real timeline predictions saved to data/real_timeline_predictions.npz")

if __name__ == "__main__":
    main()
