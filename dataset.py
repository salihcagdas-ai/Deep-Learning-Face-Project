import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FaceDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # List valid image files
        try:
            self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        except FileNotFoundError:
            print(f"ERROR: Directory not found -> {img_dir}")
            self.img_files = []
            
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Determine label filename (image.jpg -> image.txt)
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 1. Read Image
        img = cv2.imread(img_path)
        if img is None:
            # Return empty tensor to avoid crash
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros((0, 5))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 3. Normalize (0-1) and Transpose (HWC -> CHW)
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) 
        
        # 4. Read Labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Format: class_id center_x center_y width height
                    data = line.strip().split()
                    if len(data) >= 5:
                        coords = [float(x) for x in data[1:]]
                        # Valid coordinates check
                        if all(0 <= c <= 1 for c in coords):
                            boxes.append(coords)
        
        # Create target tensor
        target = torch.zeros((len(boxes), 5))
        if len(boxes) > 0:
            boxes = np.array(boxes)
            target[:, 1:] = torch.from_numpy(boxes) # [class, x, y, w, h]
            target[:, 0] = 0 # Class ID 0 for Face

        return img_tensor, target

def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes per image.
    """
    images, targets = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Merge targets with batch index
    new_targets = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            # Format: [batch_index, class, x, y, w, h]
            batch_idx = torch.full((t.shape[0], 1), i)
            new_targets.append(torch.cat((batch_idx, t), 1))
    
    if len(new_targets) > 0:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.zeros((0, 6))
        
    return images, targets

if __name__ == "__main__":
    # Test Block
    print("Checking dataset paths...")
    # Update these paths if testing locally
    TRAIN_IMG_DIR = r"dataset/images/train"
    TRAIN_LABEL_DIR = r"dataset/labels/train"
    
    if os.path.exists(TRAIN_IMG_DIR):
        ds = FaceDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
        print(f"Dataset initialized. Found {len(ds)} images.")
    else:
        print("Dataset path not found.")