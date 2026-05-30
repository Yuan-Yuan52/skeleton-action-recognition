import os
import sys
import torch
import numpy as np
import warnings
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Add NTU/stgcn path
sys.path.append(os.path.abspath(os.path.join(src_dir, '../NTU/stgcn')))
from net.st_gcn import Model as STGCN
from models_skeleton import GRUClassifier
from models_transformer import SpatialTemporalTransformer
from train_transformer import build_phase_table, split_by_video, normalize_skeleton, resample_seq

warnings.filterwarnings('ignore')

EPISODES_CSV   = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR   = "data_original/npy"
TARGET_COL     = "class_id"
NUM_CLASSES    = 5
SEED           = 42

class LocalDataset(Dataset):
    def __init__(self, df, skeleton_dir, out_len=64, normalize=True):
        self.df = df.reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.out_len = out_len
        self.normalize = normalize
        self.labels = self.df["label"].astype(int).tolist()
        self._skeleton_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_skeleton(self, video_id):
        if video_id in self._skeleton_cache:
            return self._skeleton_cache[video_id]
        npy_path = os.path.join(self.skeleton_dir, f"{video_id}.npy")
        seq = np.load(npy_path).astype(np.float32)  # (T, 33, 3)
        self._skeleton_cache[video_id] = seq
        return seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        s = int(row["start_frame_skel"])
        e = int(row["end_frame_skel"])
        label = int(row["label"])

        full_seq = self._load_skeleton(video_id)
        T = full_seq.shape[0]
        s = max(0, min(s, T - 1))
        e = max(0, min(e, T - 1))
        if e < s:
            e = s

        seg = full_seq[s:e + 1]
        if self.normalize:
            seg = normalize_skeleton(seg)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        seg = resample_seq(seg, self.out_len)

        x = torch.from_numpy(seg).float()  # (T, 33, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def get_features(model_type, untrained=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define splits
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    train_df, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)
    
    if model_type == "untrained_transformer":
        # Untrained baseline (S=2, T=2, noNorm, matching Ours configuration)
        model = SpatialTemporalTransformer(num_joints=13, in_channels=3, d_model=256, nhead=8, num_spatial_layers=2, num_temporal_layers=2, num_classes=NUM_CLASSES, dropout=0.0)
        ckpt_path = None
        normalize_data = False
    elif model_type == "transformer_ours":
        # Ours Best Model (S=2, T=2, No Norm, Flip)
        model = SpatialTemporalTransformer(num_joints=13, in_channels=3, d_model=256, nhead=8, num_spatial_layers=2, num_temporal_layers=2, num_classes=NUM_CLASSES, dropout=0.0)
        ckpt_path = "checkpoints/best_S2T2_noNorm_flip/best.pth"
        normalize_data = False
    elif model_type == "stgcn":
        model = STGCN(in_channels=3, num_class=NUM_CLASSES, graph_args={"layout": "coco", "strategy": "spatial"}, edge_importance_weighting=True)
        ckpt_path = "checkpoints/stgcn_v2/lift_class_id/best.pth"
        normalize_data = True
    elif model_type == "gru":
        model = GRUClassifier(num_joints=33, in_channels=6, num_classes=NUM_CLASSES, dropout=0.0)
        ckpt_path = "checkpoints/phase_newway_class4_gru/lift_class_id/best.pth"
        normalize_data = True
    elif model_type == "gru13":
        model = GRUClassifier(num_joints=13, in_channels=6, num_classes=NUM_CLASSES, dropout=0.0)
        ckpt_path = "checkpoints/gru_13joints/best.pth"
        normalize_data = True
        
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        
    val_ds = LocalDataset(val_df, skeleton_dir=SKELETON_DIR, out_len=64, normalize=normalize_data)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    labels_list = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            if model_type == "gru" or model_type == "gru13":
                if model_type == "gru13":
                    CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                    x_slice = x[:, :, CORE_JOINTS, :]
                else:
                    x_slice = x
                velocity = torch.zeros_like(x_slice)
                velocity[:, 1:, :, :] = x_slice[:, 1:, :, :] - x_slice[:, :-1, :, :]
                x_in = torch.cat([x_slice, velocity], dim=-1)
            elif model_type == "transformer_ours" or model_type == "untrained_transformer":
                CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_in = x[:, :, CORE_JOINTS, :]
            elif model_type == "stgcn":
                coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_in = x[:, :, coco_mapping, :]
                x_in = x_in.permute(0, 3, 1, 2).unsqueeze(-1)
                
            _ = model(x_in)
            labels_list.append(y.cpu().numpy())

    handle.remove()
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

def main():
    print("Evaluating clustering quality (Silhouette Score) on new 5-class validation set...\n")
    
    # 1. Untrained
    feat, labels = get_features("untrained_transformer")
    sc_un = silhouette_score(feat, labels)
    print(f"1. Untrained (S=2, T=2) Silhouette Score : {sc_un:.4f}")
    
    # 2. GRU (33J)
    feat, labels = get_features("gru")
    sc_gru = silhouette_score(feat, labels)
    print(f"2. GRU (33J) Baseline Silhouette Score   : {sc_gru:.4f}")
    
    # 2b. GRU (13J)
    feat, labels = get_features("gru13")
    sc_gru13 = silhouette_score(feat, labels)
    print(f"2b. GRU (13J) Baseline Silhouette Score  : {sc_gru13:.4f}")
    
    # 3. ST-GCN
    feat, labels = get_features("stgcn")
    sc_stgcn = silhouette_score(feat, labels)
    print(f"3. ST-GCN Baseline Silhouette Score      : {sc_stgcn:.4f}")
    
    # 4. Ours Best Transformer
    feat, labels = get_features("transformer_ours")
    sc_ours = silhouette_score(feat, labels)
    print(f"4. Ours Best Transformer Silhouette Score : {sc_ours:.4f}")

if __name__ == "__main__":
    main()
