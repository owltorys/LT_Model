import argparse
import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
from PIL import Image

class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = ['Hand', 'Tool', 'Block', 'Safe_Operation']
        self.classes = self.label_cols 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image: {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        labels = row[self.label_cols].values.astype(np.float32)
        target = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, target

def get_args():
    parser = argparse.ArgumentParser(description="Test Teacher Model (X2.0)")
    parser.add_argument('--model-path', type=str, default="model/ntd/model_best_ntd1_1.pth", help='Path to model weights')
    parser.add_argument('--data-root', type=str, default="DowDwen_set_resized", help='Dataset root directory')
    parser.add_argument('--test-csv', type=str, default="test.csv", help='CSV filename inside data-root')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader')
    # parse_known_args_確保不干擾其他 import
    args, _ = parser.parse_known_args()
    return args

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")
        
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    csv_full_path = os.path.join(args.data_root, args.test_csv)
    test_dataset = MultiLabelDataset(csv_full_path, args.data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
    model = models.shufflenet_v2_x2_0(weights=None)
    num_classes = 4 
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    print(f"🔄 Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
        
    print("🚀 Starting Inference...")
    with torch.inference_mode():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
                
            preds = (outputs > 0).float()
                
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
        
    exact_match_acc = np.mean(np.all(all_preds == all_targets, axis=1)) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
        
    print("   ----------------------------")
    print(f"   Exact Match Acc (全對率): {exact_match_acc:.2f}%")
    print(f"   Macro F1-Score: {f1_macro:.4f}")
    print("   ----------------------------")
    print("   Recall per Class:")
    for i, name in enumerate(test_dataset.classes):
        print(f"     - {name:<15}: {recall_per_class[i]*100:.2f}%")
    print("   ----------------------------")

if __name__ == '__main__':
    main()
