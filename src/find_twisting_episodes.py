import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from kim_features_V2 import compute_kim_features

def main():
    train_csv = "data/split/train.csv"
    val_csv = "data/split/val.csv"
    skeleton_dir = "data/skeleton_segments_npy"
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print("CSV files not found!")
        return

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_all = pd.concat([df_train, df_val])
    
    # Group by prefix (first two characters, e.g. "01_")
    df_all['prefix'] = df_all['video_id'].apply(lambda x: str(x)[:3] if '_' in str(x) else str(x))
    prefixes = df_all['prefix'].unique()
    
    print(f"Total unique prefixes: {len(prefixes)}")
    print("Scanning prefixes for twisting...")
    
    twisting_candidates = []
    
    for prefix in sorted(prefixes):
        if not prefix.endswith('_'):
            continue
        # Get all segments for this prefix
        df_src = df_all[df_all['prefix'] == prefix].sort_values('video_id')
        
        # Load all skeletons for these segments and calculate twist
        for _, row in df_src.iterrows():
            vid = row['video_id']
            npy_path = os.path.join(skeleton_dir, f"{vid}.npy")
            if os.path.exists(npy_path):
                seq = np.load(npy_path)
                seq = np.nan_to_num(seq, nan=0.0).astype('float32')
                if len(seq) > 0:
                    features = compute_kim_features(seq, fps=30.0)
                    if features['twist_max'] > 20.0:
                        twisting_candidates.append({
                            'video_id': vid,
                            'label': row['label'],
                            'twist_max': features['twist_max'],
                            'twist_p95': features['twist_p95'],
                            'twist_ratio_over_20': features['twist_ratio_over_20']
                        })
                        
    df_candidates = pd.DataFrame(twisting_candidates)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(by='twist_max', ascending=False)
        print("\nTop 15 Segments with Twisting (> 20 degrees):")
        print(df_candidates.head(15).to_string(index=False))
    else:
        print("No segments found with twist_max > 20!")

if __name__ == "__main__":
    main()
