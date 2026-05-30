import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

# ç¢ºä???src ?®é?ä¸å·è¡?import sys
sys.path.append('.')

from models_transformer import SpatialTemporalTransformer
from train_transformer import build_phase_table, split_by_video, EpisodePhaseDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '../checkpoints/robust_transformer_model/lift_class_id/best.pth'
    csv_path = '../analysis/episodes_from_json_all_single.csv'
    
    # å»ºç??é?è­é??¸å??è???    phase_df = build_phase_table(csv_path, 'lift', 'class_id', sample_stride=2)
    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=42)
    
    val_ds = EpisodePhaseDataset(
        val_df, skeleton_dir='../data_original/npy', out_len=64, normalize=True, mode='val'
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # è¼å¥ Robust æ¨¡å?
    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=3, num_temporal_layers=4, num_classes=5, dropout=0.3
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    val_df['predicted_label'] = all_preds
    
    # ç¯©é¸?ºé?æ¸¬é¯èª¤ç?å½±ç?
    misclassified = val_df[val_df['label'] != val_df['predicted_label']]
    
    print("\n" + "="*50)
    print("Find Misclassified Videos:")
    print("="*50)
    
    if len(misclassified) == 0:
        print("100% correct!")
    else:
        for idx, row in misclassified.iterrows():
            vid = row['video_id']
            t_label = row['label']
            p_label = row['predicted_label']
            print(f"Video ID: {vid:<15} | True Label: Class {t_label}  -->  Pred Label: Class {p_label}")
            
if __name__ == '__main__':
    main()
