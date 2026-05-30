import os
import sys
import torch

# Add NTU/stgcn path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NTU/stgcn')))
from net.st_gcn import Model as STGCN

sys.path.append(os.path.dirname(__file__))
from models_transformer import SpatialTemporalTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    
    # 1. Load Robust Transformer
    try:
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
        print("Success: Robust Transformer model loaded successfully!")
    except Exception as e:
        print("Error: Failed to load Robust Transformer model:", e)
        
    # 2. Load ST-GCN
    try:
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
        print("Success: ST-GCN model loaded successfully!")
    except Exception as e:
        print("Error: Failed to load ST-GCN model:", e)

if __name__ == "__main__":
    main()
