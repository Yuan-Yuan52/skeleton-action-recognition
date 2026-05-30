import torch
import os
import sys

# 將 src 納入環境路徑，確保可以 import models_transformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_transformer import SpatialTemporalTransformer

def export_model_to_onnx(pth_path, onnx_path, num_joints=13, num_classes=5):
    print(f"[*] 正在載入 PyTorch 模型: {pth_path}")
    
    # 1. 初始化模型架構 (根據你的輸入維度 13 個關節)
    # 注意：如果你的 num_classes 或其他參數跟預設不同，請在這裡修改！
    model = SpatialTemporalTransformer(num_joints=num_joints, num_classes=num_classes)
    
    # 2. 載入訓練好的權重
    device = torch.device('cpu')
    checkpoint = torch.load(pth_path, map_location=device)
    try:
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("[+] 載入成功！")
    except Exception as e:
        print(f"[-] 載入失敗: {e}")
        return

    # 3. 切換到推論模式 (非常重要，會關閉 Dropout 等機制)
    model.eval()

    # 4. 建立一個 Dummy Input (模擬推論時的假資料)
    # 形狀為 (Batch=1, Frames=64, Joints=13, Channels=3)
    dummy_input = torch.randn(1, 64, num_joints, 3)

    print(f"[*] 準備輸出 ONNX 格式到: {onnx_path}")
    
    # 5. 執行導出
    torch.onnx.export(
        model,                      # 要轉換的模型
        dummy_input,                # 模擬的輸入資料
        onnx_path,                  # 輸出的檔案路徑
        export_params=True,         # 儲存訓練好的權重
        opset_version=11,           # ONNX opset 版本 (11 通常最穩定)
        do_constant_folding=True,   # 執行常數摺疊優化
        input_names=['input'],      # 給輸入 Tensor 取個名字
        output_names=['output'],    # 給輸出 Tensor 取個名字
        # ★ 關鍵設定：設定動態維度 (Dynamic Axes)
        # 這樣導出的模型，Batch 就不會鎖死在 1，以後丟 32 也可以跑
        dynamic_axes={
            'input': {0: 'batch_size'},    
            'output': {0: 'batch_size'}
        }
    )
    print(f"[+] ONNX 模型轉換完成！檔案大小約: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # ★ 請確認這裡的權重檔路徑是你最準的那一個
    BEST_MODEL_PATH = r"D:\r13941031\model\video_action_project\checkpoints\robust_transformer_model\lift_class_id\best.pth"
    OUTPUT_ONNX_PATH = r"D:\r13941031\model\video_action_project\models\robust_transformer_best.onnx"
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)
    
    export_model_to_onnx(BEST_MODEL_PATH, OUTPUT_ONNX_PATH, num_joints=13, num_classes=5)
