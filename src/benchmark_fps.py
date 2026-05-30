import os
import sys
import time
import torch
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths for STGCN
sys.path.append(os.path.join(os.path.dirname(__file__), '../NTU/stgcn'))

from models_transformer import SpatialTemporalTransformer
from models_skeleton import GRUClassifier
from net.st_gcn import Model as STGCN

def benchmark_model(model, input_shape, device, model_name="Model", iterations=1000, warmup=100):
    model.eval()
    # Dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    fps = 1000.0 / avg_time_ms
    
    print(f"[{model_name}]")
    print(f"  Input Shape : {input_shape}")
    print(f"  Avg Latency : {avg_time_ms:.2f} ms")
    print(f"  Inference FPS : {fps:.2f} sequences/sec")
    print()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}\n")
    
    # 1. Robust Transformer (Input: N, T, V, C)
    # The Transformer takes shape (N, T=64, V=13, C=3)
    transformer = SpatialTemporalTransformer(
        num_joints=13, in_channels=3, d_model=256, nhead=8, 
        num_spatial_layers=3, num_temporal_layers=4, num_classes=5, dropout=0.0
    ).to(device)
    benchmark_model(transformer, (1, 64, 13, 3), device, model_name="Robust Transformer (T=64, V=13) - 本研究")
    
    # Non-lightweighted Transformer (V=33)
    transformer_heavy = SpatialTemporalTransformer(
        num_joints=33, in_channels=3, d_model=256, nhead=8, 
        num_spatial_layers=3, num_temporal_layers=4, num_classes=5, dropout=0.0
    ).to(device)
    benchmark_model(transformer_heavy, (1, 64, 33, 3), device, model_name="Heavy Transformer (T=64, V=33) - 未輕量化前")
    
    # 2. ST-GCN (Input: N, C, T, V, M)
    # The STGCN used for 17 joints takes shape (N, C=3, T=16 or 64?, V=17, M=1)
    # Let's benchmark it with T=16 (as used in Su's 8 FPS system) and T=64 (to be fair)
    stgcn_coco = STGCN(
        in_channels=3, num_class=3, 
        graph_args={"layout": "coco", "strategy": "spatial"}, 
        edge_importance_weighting=True
    ).to(device)
    
    # Su 2025 setup: T=16
    benchmark_model(stgcn_coco, (1, 3, 16, 17, 1), device, model_name="ST-GCN Baseline (T=16, V=17)")
    # Fair comparison: T=64
    benchmark_model(stgcn_coco, (1, 3, 64, 17, 1), device, model_name="ST-GCN Extended (T=64, V=17)")

    # 3. GRU (Input: N, T, V, C)
    # GRU takes shape (N, T=64, V=33, C=6) (since we added velocity)
    gru = GRUClassifier(
        num_joints=33, in_channels=6, num_classes=5, dropout=0.0
    ).to(device)
    benchmark_model(gru, (1, 64, 33, 6), device, model_name="RNN-GRU (T=64, V=33)")

if __name__ == "__main__":
    main()
