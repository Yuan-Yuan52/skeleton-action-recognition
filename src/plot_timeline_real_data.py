import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    # Load real predictions (New system & STGCN baseline on MediaPipe)
    data_path = "data/real_timeline_predictions_w16.npz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    data = np.load(data_path)
    transformer_preds = data['transformer']
    stgcn_preds = data['stgcn']
    
    # Load YOLO+STGCN predictions (Old NTU system, at 30 FPS)
    yolo_path = "data/yolo_stgcn_predictions.npy"
    if os.path.exists(yolo_path):
        yolo_preds = np.load(yolo_path)
    else:
        yolo_preds = np.zeros(1003, dtype=int)
    
    T = len(transformer_preds)
    fps = 15.0 # Since sample_stride=2 on 30fps video
    total_seconds = T / fps
    
    class_colors = {
        0: '#dfe6e9', # Light gray for Non
        1: '#ffeaa7', # Soft yellow for Desktop Lift (Cls 1)
        2: '#fab1a0', # Soft orange for Chair Putdown (Cls 2)
        3: '#74b9ff', # Soft blue for Bent-Knee Lift (Cls 3)
        4: '#ff7675'  # Soft red for Straight-Knee Lift (Cls 4)
    }
    
    class_names = {
        0: 'Non (Rest/Move)',
        1: 'Desktop Lift (Cls 1)',
        2: 'Chair Putdown (Cls 2)',
        3: 'Bent-Knee Lift (Cls 3 / YOLO Lift)',
        4: 'Straight-Knee Lift (Cls 4)'
    }
    
    # Construct Ground Truth based on the user's manual video frame annotations (30 FPS)
    gt = np.zeros(T, dtype=int)
    
    def set_gt(cls, start_frame_30fps, end_frame_30fps):
        start_sec = start_frame_30fps / 30.0
        end_sec = end_frame_30fps / 30.0
        start_idx = int(start_sec * fps)
        end_idx = int(end_sec * fps)
        gt[start_idx:end_idx+1] = cls

    # User's manual observation (original 30fps video frames)
    set_gt(3, 86, 131)
    set_gt(2, 181, 225)
    set_gt(2, 241, 278)
    set_gt(1, 335, 368)
    set_gt(1, 388, 408)
    set_gt(4, 451, 496)
    
    # Plotting setup
    fig, ax = plt.subplots(figsize=(14.5, 7.5), dpi=300)
    
    y_labels = ['Ground Truth (Manual)', 'New: Robust Transformer', 'Baseline 1: MediaPipe+STGCN', 'Baseline 2: YOLO+STGCN (Old NTU)']
    y_positions = [3, 2, 1, 0]
    
    # Helper to plot colored timeline bars
    def plot_bar(y_pos, labels_seq, curr_fps):
        for i in range(len(labels_seq)):
            color = class_colors[labels_seq[i]]
            ax.broken_barh([(i/curr_fps, 1/curr_fps)], (y_pos - 0.25, 0.5), facecolors=color)
            
    plot_bar(3, gt, fps)
    plot_bar(2, transformer_preds, fps)
    plot_bar(1, stgcn_preds, fps)
    plot_bar(0, yolo_preds, 30.0) # YOLO preds are at 30 FPS
    
    # Style axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11, labelpad=10)
    
    # We can crop the x-axis to 20 seconds, since nothing happens after 20 seconds
    ax.set_xlim(0, 20.0)
    ax.set_ylim(-0.6, 3.6)
    
    # Grid and spines
    ax.set_xticks(np.arange(0, 20.1, 1.0))
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add title with details
    plt.title('Action Detection Timeline Comparison (Live Inference vs. Manual Annotation)\n[Old NTU System vs. New Proposed System]', 
              fontsize=14, fontweight='bold', pad=15)
    
    # Create legend patches
    legend_patches = [
        mpatches.Patch(color=class_colors[c], label=class_names[c]) for c in sorted(class_colors.keys())
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.12), loc='upper center', 
               ncol=5, frameon=True, fontsize=10, facecolor='#f8f9fa', edgecolor='#dfe6e9')
    
    plt.tight_layout()
    
    # Save plot to Desktop and Project dir
    output_path = "action_timeline_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    
    desktop_path = "C:/Users/r13941031/Desktop/action_timeline_comparison.png"
    plt.savefig(desktop_path, bbox_inches='tight')
    
    print(f"Successfully generated timeline plot including YOLO+STGCN baseline!")

if __name__ == "__main__":
    main()
