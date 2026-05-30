import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NTU/stgcn')))
from net.st_gcn import Model as STGCN

sys.path.append(os.path.dirname(__file__))
from models_transformer import SpatialTemporalTransformer
from generate_real_timeline import normalize_skeleton, resample_seq

def main():
    skeletons = np.load("data/cam1_skeletons.npy")
    skeletons = np.nan_to_num(skeletons, nan=0.0, posinf=0.0, neginf=0.0)
    T = skeletons.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    window_size = 16
    
    # Load Robust Transformer
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
    
    # Load ST-GCN
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
    
    CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # 13 joints
    coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # 17 joints
    
    transformer_preds = []
    stgcn_preds = []
    
    with torch.no_grad():
        for t in range(T):
            start_idx = max(0, t - window_size + 1)
            win_skel = skeletons[start_idx:t+1]
            win_skel_norm = normalize_skeleton(win_skel)
            win_skel_rs = resample_seq(win_skel_norm, 64)
            
            # Robust Transformer
            trans_in = win_skel_rs[:, CORE_JOINTS, :]
            trans_tensor = torch.from_numpy(trans_in).unsqueeze(0).float().to(device)
            logits_trans = transformer_model(trans_tensor)
            pred_trans = torch.argmax(logits_trans, dim=-1).item()
            transformer_preds.append(pred_trans)
            
            # ST-GCN
            stgcn_in = win_skel_rs[:, coco_mapping, :]
            stgcn_tensor = torch.from_numpy(stgcn_in).float().to(device)
            stgcn_tensor = stgcn_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
            logits_stgcn = stgcn_model(stgcn_tensor)
            pred_stgcn = torch.argmax(logits_stgcn, dim=-1).item()
            stgcn_preds.append(pred_stgcn)
            
    print("Transformer segments:")
    current_class = -1
    start = 0
    for idx, cls in enumerate(transformer_preds):
        if cls != current_class:
            if current_class != -1:
                print(f"Class {current_class}: {start/15.0:.2f}s to {(idx-1)/15.0:.2f}s")
            current_class = cls
            start = idx
    print(f"Class {current_class}: {start/15.0:.2f}s to {(len(transformer_preds)-1)/15.0:.2f}s")
    
    print("\nST-GCN segments:")
    current_class = -1
    start = 0
    for idx, cls in enumerate(stgcn_preds):
        if cls != current_class:
            if current_class != -1:
                print(f"Class {current_class}: {start/15.0:.2f}s to {(idx-1)/15.0:.2f}s")
            current_class = cls
            start = idx
    print(f"Class {current_class}: {start/15.0:.2f}s to {(len(stgcn_preds)-1)/15.0:.2f}s")
    
    np.savez("data/real_timeline_predictions_w16.npz", 
             transformer=np.array(transformer_preds), 
             stgcn=np.array(stgcn_preds))
    print("Saved Window Size 16 predictions to data/real_timeline_predictions_w16.npz")

if __name__ == "__main__":
    main()
