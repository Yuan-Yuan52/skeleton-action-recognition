import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from models_skeleton import GRUClassifier
from train_phase_class4_advanced import build_phase_table, split_by_video, EpisodePhaseDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. æºå?æ¨¡å?ä¸¦è???GRU æ¬é?
    num_classes = 5
    model = GRUClassifier(
        num_joints=33, 
        in_channels=6, # GRU ä½¿ç¨ Velocity
        num_classes=num_classes,
        dropout=0.0
    )
    
    ckpt_path = "../checkpoints/phase_newway_class4_gru/lift_class_id/best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: ?¾ä???GRU æ¨¡å?æ¬é? {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    # 2. è¨­ç½® Forward Hook ?æ??¹å¾µ
    features_list = []
    def hook(module, input):
        features_list.append(input[0].detach().cpu().numpy())
    
    handle = model.fc[0].register_forward_pre_hook(hook)

    # 3. æºå?é©è??è???    # ==== ä¿®æ­£ï¼ä½¿?¨å???class_id (0~4) ?æ­£ç¢?CSV ====
    episodes_csv = "../analysis/episodes_from_json_all_single.csv"
    skeleton_dir = "../data_original/npy"
    phase = "lift"
    target = "class_id" 
    
    phase_df = build_phase_table(episodes_csv, phase, target, sample_stride=2)

    _, val_df = split_by_video(phase_df, val_ratio=0.2, seed=42)
    
    val_ds = EpisodePhaseDataset(
        val_df,
        skeleton_dir=skeleton_dir,
        out_len=64,
        normalize=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # 4. ?½å??¹å¾µ
    print("?å??½å? GRU ?¹å¾µ...")
    labels_list = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            
            # ==== ? ä??åº¦?¹å¾µ (Velocity) ====
            velocity = torch.zeros_like(x)
            velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
            x_combined = torch.cat([x, velocity], dim=-1)

            _ = model(x_combined)
            labels_list.append(y.cpu().numpy())

    handle.remove()

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # 5. ?·è? t-SNE ?ç¶­
    print("æ­?¨?·è? t-SNE ?ç¶­...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_features)-1), random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    # 6. è¦è¦º?ä¸¦?²å?
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=all_labels,
        palette=sns.color_palette("Set1", n_colors=len(np.unique(all_labels))),
        legend="full",
        alpha=0.7
    )
    
    plt.title(f"t-SNE Visualization (GRU Model on 5 Classes)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Action Class")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = "../tsne_visualization_gru.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GRU t-SNE ?è¡¨å·²å²å­è³: {output_path}")

if __name__ == "__main__":
    main()
