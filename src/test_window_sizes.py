import numpy as np
import torch
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NTU/stgcn')))
from net.st_gcn import Model as STGCN
sys.path.append(os.path.dirname(__file__))
from models_transformer import SpatialTemporalTransformer
from generate_real_timeline import normalize_skeleton, resample_seq

def test_size(window_size):
    skeletons = np.load("data/cam1_skeletons.npy")
    skeletons = np.nan_to_num(skeletons, nan=0.0, posinf=0.0, neginf=0.0)
    T = skeletons.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    
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
    
    CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    preds = []
    with torch.no_grad():
        for t in range(T):
            start_idx = max(0, t - window_size + 1)
            win_skel = skeletons[start_idx:t+1]
            win_skel_norm = normalize_skeleton(win_skel)
            win_skel_rs = resample_seq(win_skel_norm, 64)
            trans_in = win_skel_rs[:, CORE_JOINTS, :]
            trans_tensor = torch.from_numpy(trans_in).unsqueeze(0).float().to(device)
            logits = transformer_model(trans_tensor)
            pred = torch.argmax(logits, dim=-1).item()
            preds.append(pred)
            
    # Print contiguous segments
    print(f"\n--- Window Size {window_size} ---")
    current_class = -1
    start = 0
    for idx, cls in enumerate(preds):
        if cls != current_class:
            if current_class != -1:
                print(f"Class {current_class}: {start/15.0:.2f}s to {(idx-1)/15.0:.2f}s")
            current_class = cls
            start = idx
    print(f"Class {current_class}: {start/15.0:.2f}s to {(len(preds)-1)/15.0:.2f}s")

def main():
    for w in [16, 24, 32, 48, 64]:
        test_size(w)

if __name__ == "__main__":
    main()
