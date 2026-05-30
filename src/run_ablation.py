import os
import subprocess
import time
import torch
import pandas as pd
from models_transformer import SpatialTemporalTransformer

def measure_model_metrics(spatial, temporal):
    model = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8,
        num_spatial_layers=spatial, num_temporal_layers=temporal,
        num_classes=5
    )
    # count params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # measure latency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 64, 13, 3).to(device)
    # warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    
    # ms per iteration
    latency_ms = ((time.time() - start_time) / iters) * 1000
    return params, latency_ms

def main():
    combinations = [
        (1, 1),
        (1, 2),
        (2, 1)
    ]
    
    results = []
    
    for s, t in combinations:
        print(f"\n{'='*50}")
        print(f"Running Ablation: Spatial={s}, Temporal={t}")
        print(f"{'='*50}")
        
        params, latency = measure_model_metrics(s, t)
        print(f"Model Params: {params/1e6:.2f} M")
        print(f"Inference Latency: {latency:.2f} ms")
        
        ckpt_dir = f"checkpoints/ablation_s{s}_t{t}"
        cmd = [
            "python", "src/train_transformer.py",
            "--num_spatial_layers", str(s),
            "--num_temporal_layers", str(t),
            "--epochs", "40", # 這裡可以根據需要調整 epoch 數量
            "--ckpt_dir", ckpt_dir,
            "--wandb_name", f"ablation_s{s}_t{t}"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        best_acc = 0.0
        for line in process.stdout:
            print(line, end='')
            # parse the line like: Epoch 40/40 Loss=0.1234 ValAcc=0.9500 BestAcc=0.9600
            if "BestAcc=" in line:
                try:
                    best_acc_str = line.split("BestAcc=")[1].strip()
                    # It might have classification reports after, so we split by space if any
                    best_acc_str = best_acc_str.split()[0]
                    best_acc = float(best_acc_str)
                except Exception as e:
                    pass
        
        process.wait()
        
        results.append({
            "Spatial Layers": s,
            "Temporal Layers": t,
            "Parameters (M)": f"{params/1e6:.2f}",
            "Latency (ms)": f"{latency:.2f}",
            "Best Val Acc (%)": f"{best_acc*100:.2f}"
        })
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv("ablation_results_extra.csv", index=False)
        print("\nIntermediate Results saved to ablation_results_extra.csv:")
        print(df.to_string(index=False))

    print("\n[DONE] All combinations finished!")
    print("Final results saved in ablation_results_extra.csv")

if __name__ == "__main__":
    main()
