import os
import sys
import torch
import numpy as np
import warnings
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from models_transformer import SpatialTemporalTransformer
from models_skeleton import GRUClassifier
from train_phase_class4_advanced import build_phase_table, split_by_video, EpisodePhaseDataset
# from train_stgcn import custom_collate_fn

sys.path.append(os.path.join(os.path.dirname(__file__), '../NTU/stgcn'))
from net.st_gcn import Model as STGCN

warnings.filterwarnings('ignore')

def get_features(model_type, untrained=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    
    if model_type == "transformer":
        model = SpatialTemporalTransformer(num_joints=13, in_channels=3, d_model=256, nhead=8, num_spatial_layers=3, num_temporal_layers=4, num_classes=num_classes, dropout=0.0)
        ckpt_path = "../checkpoints/transformer_v1/lift_class_id/best.pth"
    elif model_type == "robust_transformer":
        model = SpatialTemporalTransformer(num_joints=13, in_channels=3, d_model=256, nhead=8, num_spatial_layers=3, num_temporal_layers=4, num_classes=num_classes, dropout=0.0)
        ckpt_path = "../checkpoints/robust_transformer_model/lift_class_id/best.pth"
    elif model_type == "stgcn":
        model = STGCN(in_channels=3, num_class=num_classes, graph_args={"layout": "coco", "strategy": "spatial"}, edge_importance_weighting=True)
        ckpt_path = "../checkpoints/stgcn_v2/lift_class_id/best.pth"
    else:
        model = GRUClassifier(num_joints=33, in_channels=6, num_classes=num_classes, dropout=0.0)
        ckpt_path = "../checkpoints/phase_newway_class4_gru/lift_class_id/best.pth"
        
    if not untrained:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
    model.to(device)
    model.eval()

    features_list = []
    def hook(module, input):
        if model_type == "stgcn":
            feat = input[0] # (N, 256, T, V)
            feat = feat.mean(dim=(2, 3)) # (N, 256)
            features_list.append(feat.detach().cpu().numpy())
        else:
            features_list.append(input[0].detach().cpu().numpy())
            
    if model_type == "stgcn":
        handle = model.fcn.register_forward_pre_hook(hook)
    else:
        handle = model.fc[0].register_forward_pre_hook(hook)

    # ?å¶è¼¸åºï¼é¿?ç«?¢å¤ª??    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    episodes_csv = "../analysis/episodes_from_json_all_single.csv"
    phase_df = build_phase_table(episodes_csv, "lift", "class_id", sample_stride=2)
    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=42)
    
    val_ds = EpisodePhaseDataset(val_df, skeleton_dir="../data_original/npy", out_len=64, normalize=True)
    
    if model_type == "stgcn":
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    else:
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    sys.stdout = old_stdout # ?¢å¾©è¼¸åº

    labels_list = []
    with torch.no_grad():
        for batch_data in val_loader:
            if model_type == "stgcn":
                x, y = batch_data
            else:
                x, y = batch_data
                
            x = x.to(device)
            if model_type == "gru":
                velocity = torch.zeros_like(x)
                velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
                x = torch.cat([x, velocity], dim=-1)
            elif model_type == "transformer" or model_type == "robust_transformer":
                CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x = x[:, :, CORE_JOINTS, :] 
            elif model_type == "stgcn":
                # Map 33 joints to 17 COCO joints
                coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x = x[:, :, coco_mapping, :]
                if len(x.shape) == 4:
                    if x.size(-1) == 3: # (N, T, V, C) -> (N, C, T, V)
                        x = x.permute(0, 3, 1, 2)
                    x = x.unsqueeze(-1) # (N, C, T, V, M)
                
            _ = model(x)
            labels_list.append(y.cpu().numpy())

    handle.remove()
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

def main():
    print("Computing quantitative clustering metrics (Silhouette Score)...")
    
    feat_un, labels = get_features("transformer", untrained=True)
    score_un = silhouette_score(feat_un, labels)
    print(f"1. Untrained Model Silhouette Score: {score_un:.4f}")
    
    feat_gru, _ = get_features("gru", untrained=False)
    score_gru = silhouette_score(feat_gru, labels)
    print(f"2. GRU Model Silhouette Score: {score_gru:.4f}")
    
    feat_stgcn, _ = get_features("stgcn", untrained=False)
    score_stgcn = silhouette_score(feat_stgcn, labels)
    print(f"3. ST-GCN Model Silhouette Score: {score_stgcn:.4f}")
    
    feat_tr, _ = get_features("transformer", untrained=False)
    score_tr = silhouette_score(feat_tr, labels)
    print(f"4. Transformer Model (v1) Silhouette Score: {score_tr:.4f}")
    
    feat_robust, _ = get_features("robust_transformer", untrained=False)
    score_robust = silhouette_score(feat_robust, labels)
    print(f"5. Robust Transformer Model Silhouette Score: {score_robust:.4f}")

if __name__ == "__main__":
    main()
