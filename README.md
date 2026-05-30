# 即時下背傷病風險監測系統

**國立臺灣大學 碩士論文 — 2026 年 6 月**

基於 3D 骨架與時空 Transformer 的非接觸式即時監測系統，偵測人工物料搬運（MMH）作業中的下背傷病風險。系統透過 RTSP 串流接入現有工業攝影機，符合中華民國職業安全衛生法及 KIM-LHC 人因評估標準。

---

## 核心成果

| 指標 | 數值 |
|:-----|:----:|
| 5 分類驗證準確率 | **97.67%** |
| 特徵空間輪廓係數（Silhouette Score）| **0.7339** |
| 背景動作（Class 0）特異度 | **100%** |
| 端到端處理幀率（純 CPU，無 GPU）| **85.3 FPS** |
| 模型參數量 / 大小 | **4.08M / 15.57 MB** |
| 高斯噪聲抗擾（σ=0.1）| **98.06%** |
| 2 關節遮擋抗擾 | **96.12%** |

---

## 系統架構

```
RTSP 攝影機串流
      │
      ▼
MediaPipe Pose（5.98 ms/幀）
   33 關節 → 13 核心關節
      │
      ▼
時空雙通道 Transformer（5.71 ms/次）
   S=2 空間層 + T=2 時間層
   關閉 CS-Normalization ← 保留絕對高度特徵
      │
      ├─► 5 類搬運動作辨識
      ├─► 軀幹扭轉角判定（肩髖投影夾角，閾值 20°）
      └─► 雙層狀態機 + EMA 機率平滑
              │
              ▼
        SQLite 事件記錄 + 即時 GUI 警示
```

**5 分類動作（對應 KIM-LHC 評估）：**

| 類別 | 動作名稱 | 風險等級 |
|:---:|:--------|:------:|
| 0 | 背景／非搬運動作 | — |
| 1 | 桌面高度蹲舉 | 低 |
| 2 | 髖鉸鏈搬運 | 中 |
| 3 | 屈膝非對稱搬運 | 中 |
| 4 | **直膝彎腰搬運** | **高 ⚠️** |

---

## 分類性能

**最佳模型混淆矩陣（S=2, T=2 ｜ 關閉正規化 ｜ Fair-Flip ｜ 驗證準確率 97.67%）：**

![混淆矩陣](thesis_materials/confusion_matrix_best_model_zero.png)

- Class 0（背景）：**99% Recall**，誤報率為零
- Class 1（蹲舉）：**100% Recall**
- Class 3 vs Class 4（安全 vs 危險）：正確區分率極高

---

## 特徵空間品質（t-SNE 視覺化）

未訓練模型（隨機權重）vs 訓練完成的時空 Transformer：

![t-SNE 對比](thesis_materials/tsne_comparison_zero.png)

輪廓係數從 **-0.0057 → 0.7339**，5 類動作特徵群呈現清晰邊界與高度凝聚性。

---

## 核心貢獻：關閉 CS-Normalization 策略

傳統骨架動作辨識普遍使用 Center-Scale Normalization（CS-Norm）消除相機視角差異。本研究發現，CS-Norm 會**抹除絕對高度特徵**——而這正是區分背景日常動作（Class 0）與搬運動作的關鍵物理資訊。

**方法消融實驗：**

![方法消融](thesis_materials/confusion_matrix_method_ablation_zero.png)

| 方法 | 驗證準確率 | 輪廓係數 | Class 0 Recall |
|:-----|:--------:|:-------:|:--------------:|
| 完整（CS-Norm + 增強）| 96.51% | — | 0.98 |
| 無增強 | 96.51% | — | 0.98 |
| **關閉 Norm + 增強（本研究）** | **97.67%** | **0.7339** | **1.00** |

補償方案：在線隨機旋轉 + 離線水平翻轉（Fair Flip），訓練集從 1,124 擴增至 2,506 筆。

---

## 空間自注意力可解釋性分析

模型在**無任何解剖學監督**的情況下，自主學習到具有生物力學意義的關節互動模式：

![注意力差異熱圖](thesis_materials/attention_class3_vs_class4_diff.png)

- **Class 4（直膝彎腰，最高風險）** → 高 **膝關節 ↔ 踝關節** 注意力 — 精確對應 KIM-LHC「直膝起吊」最大 L5/S1 剪力危害指標
- **Class 3（屈膝安全搬運）** → 高 **肩膀 ← 鼻子** 注意力 — 捕捉非對稱搬運的軀幹旋轉特徵

**各類別前五名最強關節注意力對：**

![Top 注意力對](thesis_materials/attention_top_pairs_per_class.png)

---

## 系統魯棒性分析

模型在各種真實場景干擾下仍維持高準確率：

![魯棒性分析](thesis_materials/robustness_comparison.png)

| 干擾條件 | 本研究 | GRU-33J | ST-GCN |
|:--------|:------:|:-------:|:------:|
| 無干擾 | **97.67%** | 96.51% | 96.51% |
| 高斯噪聲 σ=0.1 | **98.06%** | 94.19% | 96.51% |
| 隨機遮擋 2 關節 | **96.12%** | 65.50% | 71.32% |
| 隨機遮擋 5 關節 | **75.97%** | 65.50% | 45.74% |

自注意力機制的全局加權特性能補償缺失關節，而 ST-GCN 的固定圖拓撲導致雪崩式準確率下降。

---

## 解剖注意力偏置（AAB）消融實驗

本研究嘗試以骨架圖距離初始化空間注意力偏置（AAB）。結果顯示加入解剖先驗反而**降低性能**。

![AAB 消融學習曲線](thesis_materials/aab_ablation_curve.png)

| 設定 | 驗證準確率 | 輪廓係數 | Class 0 Recall |
|:-----|:--------:|:-------:|:--------------:|
| + AAB（解剖偏置）| 96.51% | 0.5813 | 0.98 |
| **無 AAB（本研究）** | **97.67%** | **0.7339** | **1.00** |

解剖偏置干擾了 No-Norm 策略所保留的絕對位置特徵，驗證了無約束自注意力優於先驗結構約束設計。

---

## 即時動作時序偵測

結合雙層狀態機與 EMA 平滑，系統能正確切分連續影片中的所有搬運事件：

![動作時序](thesis_materials/action_timeline_comparison.png)

與人工標記基準比較，**6 段搬運事件全部正確偵測，無任何幀間閃爍**。

---

## 軀幹扭轉角度即時判定

KIM-LHC 將軀幹旋轉 ≥20° 列為高權重風險因子。系統利用肩寬向量與髖寬向量的水平投影夾角即時計算扭轉角：

![軀幹扭轉](thesis_materials/twisting_validation_01_seg024.png)

---

## 模型比較

| 模型 | 輸入關節數 | 維度 | 驗證準確率 | 輪廓係數 | 推論延遲 |
|:-----|:--------:|:----:|:---------:|:-------:|:-------:|
| GRU Baseline（33J）| 33 | 6D | 96.51% | 0.4219 | 8.14ms |
| GRU Baseline（13J）| 13 | 3D | 96.12% | 0.4838 | 8.22ms |
| ST-GCN Baseline（17J）| 17 | 3D | 96.51% | 0.4634 | 6.79ms |
| **ST-Transformer（本研究）** | **13** | **3D** | **97.67%** | **0.7339** | **5.71ms** |

---

## 程式碼結構

```
src/
├── models_transformer.py      # 時空 Transformer（S=2, T=2, d_model=256）
├── models_transformer_aab.py  # 解剖注意力偏置消融版本
├── train_transformer.py       # EpisodePhaseDataset + 訓練工具
├── run_best_configs.py        # 最佳模型訓練（No-Norm + Fair-Flip）
├── train_aab.py               # AAB 消融實驗訓練腳本
├── visualize_attention.py     # 空間注意力權重提取與視覺化
├── extract_skeleton.py        # MediaPipe 骨架擷取 → NPY
├── evaluate_robustness.py     # 高斯噪聲 + 遮擋魯棒性測試
├── kim_scoring.py             # KIM-LHC 風險評分計算
├── benchmark_fps.py           # 端到端 FPS 基準測試
└── utils.py                   # 訓練工具函式

thesis_materials/
├── deployment/
│   ├── live_cam_onnx_sqlite.py  # 部署端：ONNX 推論 + SQLite 事件記錄
│   └── export_to_onnx.py        # PyTorch → ONNX 匯出
└── *.png                        # 所有實驗結果圖表

analysis/
└── *.csv                        # 消融實驗與評估數值結果
```

---

## 快速開始

```bash
pip install torch mediapipe opencv-python numpy pandas scikit-learn matplotlib seaborn onnxruntime

# 即時推論（RTSP 串流、網路攝影機或影片檔）
python thesis_materials/deployment/live_cam_onnx_sqlite.py \
    --source 0 \
    --onnx_path path/to/model.onnx

# 視覺化空間自注意力權重
python src/visualize_attention.py

# 執行 AAB 消融實驗（建議使用 GPU）
python src/train_aab.py

# 魯棒性評估（噪聲 + 遮擋）
python src/evaluate_robustness.py
```

---

## 技術棧

`PyTorch 2.x` · `MediaPipe` · `ONNX Runtime` · `OpenCV` · `SQLite` · `NumPy` · `scikit-learn` · `seaborn`

---

## 相關專案

[**camera-isp-pipeline**](https://github.com/Yuan-Yuan52/camera-isp-pipeline) — 從零實作的相機 ISP 管線（Demosaicing、White Balance、CLAHE 等），用於研究影像前處理對工業現場骨架偵測品質的影響。

---

## 作者

**顏慶源（Ching-Yuan Yen）**  
國立臺灣大學 光電工程學研究所  
指導教授：林晃巖博士
