import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# Local Modules
from dataset import FaceDataset, collate_fn
#from model import FaceDetector
from model_resnet import FaceDetector
from loss import YOLOLoss

# --- CONFIGURATION ---
EPOCHS = 200
PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADVANCED_MODE = True

# Ensure results directories exist
if not os.path.exists("results_resnet"):
    os.makedirs("results_resnet")

def compute_batch_metrics(outputs, targets, iou_thresh=0.5, conf_thresh=0.5):
    """Calculates TP, FP, FN for metrics."""
    tp, fp, fn = 0, 0, 0
    
    for pred in outputs:
        batch, _, h, w = pred.shape
        pred = pred.permute(0, 2, 3, 1)
        conf = torch.sigmoid(pred[..., 0])
        
        target_mask = torch.zeros((batch, h, w), device=pred.device)
        
        for t in targets:
            b_id = int(t[0])
            if b_id >= batch: continue
            gx = int(t[2] * w)
            gy = int(t[3] * h)
            if 0 <= gx < w and 0 <= gy < h:
                target_mask[b_id, gy, gx] = 1

        matches = (conf > conf_thresh) & (target_mask == 1)
        tp += matches.sum().item()
        
        false_alarms = (conf > conf_thresh) & (target_mask == 0)
        fp += false_alarms.sum().item()
        
        misses = (conf < conf_thresh) & (target_mask == 1)
        fn += misses.sum().item()
        
    return tp, fp, fn

def train():
    print(f"Training Started... Device: {DEVICE}")
    print(f"Target: {EPOCHS} Epochs | Patience: {PATIENCE}")
    
    # 1. Datasets
    print("Loading datasets...")
    # Update paths if necessary
    train_ds = FaceDataset(r"dataset/images/train", r"dataset/labels/train")
    val_ds = FaceDataset(r"dataset/images/val", r"dataset/labels/val")
    
    print(f"Samples -> Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # 2. Model & Optimizer
    model = FaceDetector(advanced=ADVANCED_MODE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = YOLOLoss()
    
    history = {"train_loss": [], "val_loss": [], "f1": []}
    
    # Early Stopping Variables
    best_f1 = 0.0
    patience_counter = 0
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, targets in loop:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_tp, val_fp, val_fn = 0, 0, 0
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                tp, fp, fn = compute_batch_metrics(outputs, targets)
                val_tp += tp; val_fp += fp; val_fn += fn
        
        avg_val_loss = val_loss / len(val_loader)
        
        epsilon = 1e-7
        precision = val_tp / (val_tp + val_fp + epsilon)
        recall = val_tp / (val_tp + val_fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # --- LOGGING ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["f1"].append(f1_score)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f" -> Loss: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F1: {f1_score:.4f} | LR: {current_lr:.6f}")
        
        # --- EARLY STOPPING ---
        if f1_score > best_f1:
            best_f1 = f1_score
            patience_counter = 0 
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  *** New Record! Model Saved (F1: {best_f1:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  ... No Improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("\nEARLY STOPPING TRIGGERED!")
                break

    print("Training Completed.")

if __name__ == "__main__":
    train()