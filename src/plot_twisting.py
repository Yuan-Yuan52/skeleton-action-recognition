import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from kim_features_V2 import compute_twist_flex, compute_kim_features

def main():
    video_id = "01_seg024_act1"
    npy_path = f"data/skeleton_segments_npy/{video_id}.npy"
    
    if not os.path.exists(npy_path):
        print(f"File {npy_path} not found!")
        return
        
    # Load skeleton
    seq = np.load(npy_path)
    seq = np.nan_to_num(seq, nan=0.0).astype('float32')
    T = len(seq)
    fps = 30.0
    time_axis = np.arange(T) / fps
    
    # Compute twist and flex
    twist_deg, flex_deg = compute_twist_flex(seq)
    features = compute_kim_features(seq, fps=fps)
    
    # Create plot
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.plot(time_axis, twist_deg, color='#e056fd', linewidth=2.5, label='Twisting Angle (Shoulder vs Hip line)')
    
    # Threshold line
    plt.axhline(y=20.0, color='#ff7675', linestyle='--', linewidth=1.5, label='KIM-LHC Twisting Threshold (20°)')
    
    # Highlight high risk area
    plt.fill_between(time_axis, twist_deg, 20.0, where=(twist_deg > 20.0), 
                     color='#ff7675', alpha=0.3, interpolate=True, label='High Twisting Exposure (>20°)')
    
    # Styling
    plt.title(f'Trunk Twisting Angle Analysis - {video_id}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.ylabel('Twisting Angle (degrees)', fontsize=11)
    plt.xlim(0, time_axis[-1])
    plt.ylim(0, 95)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Add metrics text box
    info_text = (
        f"Max Twist Angle: {features['twist_max']:.1f}°\n"
        f"95th Percentile: {features['twist_p95']:.1f}°\n"
        f"Twist Ratio (>20°): {features['twist_ratio_over_20']*100:.1f}%\n"
        f"KIM-LHC Extra Score: +3.0 Points (High Twist Risk)"
    )
    props = dict(boxstyle='round', facecolor='#dfe6e9', alpha=0.8, edgecolor='#b2bec3')
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    # Save image
    output_path = "twisting_validation_01_seg024.png"
    plt.savefig(output_path)
    print(f"Plot saved successfully to {output_path}!")

if __name__ == "__main__":
    main()
