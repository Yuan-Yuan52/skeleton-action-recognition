import os
import subprocess
import pandas as pd

def main():
    # We fix the architecture to our best performing one: 2+2
    s = 2
    t = 2
    
    configurations = [
        {"name": "Ours (Norm + Aug)", "flags": []},
        {"name": "No Norm", "flags": ["--disable_norm"]},
        {"name": "No Aug", "flags": ["--disable_aug"]},
        {"name": "No Norm & No Aug", "flags": ["--disable_norm", "--disable_aug"]}
    ]
    
    results = []
    
    for config in configurations:
        name = config["name"]
        flags = config["flags"]
        
        print(f"\n{'='*50}")
        print(f"Running Methods Ablation: {name}")
        print(f"{'='*50}")
        
        # Format a safe string for folder names
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("&", "and")
        ckpt_dir = f"checkpoints/ablation_methods_{safe_name}"
        
        cmd = [
            "python", "src/train_transformer.py",
            "--num_spatial_layers", str(s),
            "--num_temporal_layers", str(t),
            "--epochs", "40",
            "--ckpt_dir", ckpt_dir,
            "--wandb_name", f"ablation_methods_{safe_name}"
        ] + flags
        
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        best_acc = 0.0
        for line in process.stdout:
            print(line, end='')
            if "BestAcc=" in line:
                try:
                    best_acc_str = line.split("BestAcc=")[1].strip()
                    best_acc_str = best_acc_str.split()[0]
                    best_acc = float(best_acc_str)
                except Exception as e:
                    pass
        
        process.wait()
        
        results.append({
            "Method": name,
            "Center-Scale Norm": "Yes" if "--disable_norm" not in flags else "No",
            "Data Augmentation": "Yes" if "--disable_aug" not in flags else "No",
            "Best Val Acc (%)": f"{best_acc*100:.2f}"
        })
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv("ablation_methods_results.csv", index=False)
        print("\nIntermediate Results saved to ablation_methods_results.csv:")
        print(df.to_string(index=False))

    print("\n[DONE] All combinations finished!")
    print("Final results saved in ablation_methods_results.csv")

if __name__ == "__main__":
    main()
