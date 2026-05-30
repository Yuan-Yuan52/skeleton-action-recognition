import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

sys.path.append(os.path.abspath(os.path.join(src_dir, '../NTU/stgcn')))
from net.st_gcn import Model as STGCN
from models_skeleton import GRUClassifier
from train_transformer import build_phase_table, split_by_video
from train_comparison_models import EpisodePhaseDataset, evaluate_model

EPISODES_CSV   = "analysis/episodes_from_json_all_zero.csv"
SKELETON_DIR   = "data_original/npy"
TARGET_COL     = "class_id"
NUM_CLASSES    = 5
SEED           = 42

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_df = build_phase_table(EPISODES_CSV, "lift", TARGET_COL, sample_stride=2)
    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=SEED)
    val_ds = EpisodePhaseDataset(val_df, skeleton_dir=SKELETON_DIR, out_len=64, normalize=True, use_aug=False, mode="val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # GRU
    gru_model = GRUClassifier(num_joints=33, in_channels=6, num_classes=5).to(device)
    gru_ckpt = torch.load("checkpoints/phase_newway_class4_gru/lift_class_id/best.pth", map_location=device, weights_only=False)
    gru_model.load_state_dict(gru_ckpt["state_dict"])
    gru_acc, gru_cm, gru_preds, gru_labels = evaluate_model(gru_model, val_loader, device, "gru")

    # ST-GCN
    stgcn_model = STGCN(in_channels=3, num_class=5, graph_args={"layout": "coco", "strategy": "spatial"}, edge_importance_weighting=True).to(device)
    stgcn_ckpt = torch.load("checkpoints/stgcn_v2/lift_class_id/best.pth", map_location=device, weights_only=False)
    stgcn_model.load_state_dict(stgcn_ckpt["state_dict"])
    stgcn_acc, stgcn_cm, stgcn_preds, stgcn_labels = evaluate_model(stgcn_model, val_loader, device, "stgcn")

    print("GRU Confusion Matrix:")
    print(gru_cm)
    print("\nST-GCN Confusion Matrix:")
    print(stgcn_cm)
    
    # Check overlap of misclassifications
    gru_wrong = np.where(gru_preds != gru_labels)[0]
    stgcn_wrong = np.where(stgcn_preds != stgcn_labels)[0]
    
    common_wrong = np.intersect1d(gru_wrong, stgcn_wrong)
    print(f"\nGRU wrong indices ({len(gru_wrong)}): {gru_wrong}")
    print(f"ST-GCN wrong indices ({len(stgcn_wrong)}): {stgcn_wrong}")
    print(f"Common wrong indices ({len(common_wrong)}): {common_wrong}")

if __name__ == "__main__":
    main()
