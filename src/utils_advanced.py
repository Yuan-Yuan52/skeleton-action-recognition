#訓練用模型的training/evaluation/util函式
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, ckpt_dir, filename="last.pth"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, best_path)

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, cm, report

def evaluate_advanced(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    # 我們在這裡也可以順便評估 Validation Loss 如果有需要，但為了配合原版回傳值，我們先計算就好
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # ==== 新增：計算 Velocity ====
            velocity = torch.zeros_like(x)
            velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
            x_combined = torch.cat([x, velocity], dim=-1)

            logits = model(x_combined)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report = classification_report(all_labels, all_preds, digits=4)
    val_loss = total_loss / max(1, len(dataloader.dataset))
    return val_loss, acc, cm, report

def make_weighted_sampler(labels):
    import torch
    from torch.utils.data import WeightedRandomSampler
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler
