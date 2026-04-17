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
# [修改] 引入 v2 用於現代化增強
from torchvision.transforms import v2 
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
from PIL import Image
# [新增] 引入 EMA 工具
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ===========================================================
# 參數設定
# ===========================================================
parser = argparse.ArgumentParser(description='ShuffleNetV2 Multi-Label Training (Mixup + EMA)')
parser.add_argument('--width-mult', default=1.0, type=float, help='model width multiplier')
parser.add_argument('--epochs', default=60, type=int, help='epochs') # 維持 60 或更高
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') 
parser.add_argument('--save-dir', default='checkpoints', help='save directory')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--num-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('--data-root', required=True, help='folder containing images (e.g. ./data)')
parser.add_argument('--train-csv', default='train.csv', help='train.csv')
parser.add_argument('--val-csv', default='val.csv', help='val.csv')
parser.add_argument('--seed', default=24, type=int, help='random seed')
# [新增] EMA 衰減率參數
parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
args = parser.parse_args()

# ===========================================================
# 自定義多標籤 Dataset
# ===========================================================
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

# ===========================================================
# 工具函式
# ===========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Random Seed fixed to: {seed}")

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'best_model.pth'))
        print(f"💾 Saved new best model to: {os.path.join(save_dir, 'best_model.pth')}")

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_pos_weights(dataset):
    df = dataset.df
    labels = df[dataset.label_cols].values
    pos_counts = np.sum(labels, axis=0)
    total_counts = len(df)
    neg_counts = total_counts - pos_counts
    pos_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"⚖️ Calculated Pos Weights: {pos_weights}")
    return pos_weights

def set_trainable_layers(model, unfreeze_target):
    for param in model.parameters(): param.requires_grad = False
    print(f"🔓 Setting trainable layers for target: {unfreeze_target}")
    
    if unfreeze_target == 'classifier':
        if hasattr(model, 'fc'):
            for p in model.fc.parameters(): p.requires_grad = True
    elif unfreeze_target == 'last_stage':
        if hasattr(model, 'fc'):
            for p in model.fc.parameters(): p.requires_grad = True
        if hasattr(model, 'conv5'):
            for p in model.conv5.parameters(): p.requires_grad = True
        if hasattr(model, 'stage4'):
            for p in model.stage4.parameters(): p.requires_grad = True
    elif unfreeze_target == 'all':
        for param in model.parameters(): param.requires_grad = True
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> Total Trainable Params: {trainable:,}")

# ===========================================================
# 定義 Mixup/CutMix 轉換器
# ===========================================================
def get_mixup_cutmix_transforms(num_classes):
    mixup_cutmix = v2.RandomChoice([
        v2.RandomApply([v2.MixUp(num_classes=num_classes, alpha=0.4)], p=1.0),
        v2.RandomApply([v2.CutMix(num_classes=num_classes, alpha=1.0)], p=1.0)
    ])
    return mixup_cutmix

# ===========================================================
# [修改] 訓練迴圈 (加入 EMA update)
# ===========================================================
def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, scheduler, mixup_fn=None, ema_model=None):
    model.train()
    losses = AverageMeter()
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # [新增] 更新 EMA 模型參數
        if ema_model is not None:
            ema_model.update_parameters(model)
        
        if scheduler is not None:
            scheduler.step()
        
        losses.update(loss.item(), images.size(0))
        
        if i % (len(train_loader) // 3) == 0 and i > 0:
            print(f"   Step [{i}/{len(train_loader)}] Loss: {losses.val:.4f} LR: {optimizer.param_groups[0]['lr']:.2e}")

# ===========================================================
# 驗證迴圈 (使用 EMA 模型進行評估)
# ===========================================================
def validate(val_loader, model, criterion, device, class_names, epoch=None, title="Model"):
    # 如果傳入的是 EMA model，它包了一層，需要取 .module 或是直接用
    # AveragedModel 的 forward 預設就是呼叫內部的 module
    model.eval()
    losses = AverageMeter()
    
    all_preds = []
    all_targets = []
    
    with torch.inference_mode():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))
            
            # 驗證時保持 0.5 門檻
            preds = (outputs > 0).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    exact_match_acc = np.mean(np.all(all_preds == all_targets, axis=1)) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    
    print(f"\n🧩 {title} Val Summary (Epoch {epoch}):")
    print(f"   Loss: {losses.avg:.4f}")
    print(f"   Exact Match Acc: {exact_match_acc:.2f}%")
    print(f"   Macro F1-Score: {f1_macro:.4f}")
    print("   ----------------------------")
    print("   Recall per Class:")
    for i, name in enumerate(class_names):
        print(f"     - {name:<15}: {recall_per_class[i]*100:.2f}%")
    print("   ----------------------------")
    
    return f1_macro

# ===========================================================
# 主程式
# ===========================================================
def main():
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    # [保留] td4.2 的 scale 設定 (0.8, 1.0)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    print("🎨 Initializing Mixup/CutMix Augmentation...")
    mixup_fn = get_mixup_cutmix_transforms(num_classes=4)

    print(f"📂 Loading data from CSV...")
    train_dataset = MultiLabelDataset(f"{args.data_root}/{args.train_csv}", args.data_root, transform=train_transform)
    val_dataset = MultiLabelDataset(f"{args.data_root}/{args.val_csv}", args.data_root, transform=val_transform)
    
    print(f"📚 Total Images: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    print(f"🔄 Loading ShuffleNet V2 (width={args.width_mult})...")
    
    if args.width_mult == 0.5:
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        model = models.shufflenet_v2_x0_5(weights=weights)
    elif args.width_mult == 1.0:
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        model = models.shufflenet_v2_x1_0(weights=weights)
    elif args.width_mult == 2.0: 
        weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        model = models.shufflenet_v2_x2_0(weights=weights)
    else:
        raise ValueError("Unsupported width_mult")

    num_classes = 4 
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # [新增] EMA 模型初始化
    print(f"🧠 Initializing EMA Model (decay={args.ema_decay})...")
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
    ema_model = ema_model.to(device)

    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        # 如果有存 EMA 狀態也應該載入，這裡簡化處理，假設從頭訓練

    pos_weights = get_pos_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    scaler = GradScaler()
    os.makedirs(args.save_dir, exist_ok=True)

    stages = [
        {'name': 'Stage 1_Classifier', 'lr_factor': 1.0, 'target': 'classifier', 'epochs': 5},
        {'name': 'Stage 2_Full',       'lr_factor': 0.1, 'target': 'all',        'epochs': args.epochs},
    ]
    
    global_best_f1 = 0.0

    for stage in stages:
        print(f"\n🔔 Starting {stage['name']}...")
        set_trainable_layers(model, stage['target'])

        current_lr = args.lr * stage['lr_factor']
        stage_epochs = stage['epochs']
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=current_lr, weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=current_lr, 
                                                  epochs=stage_epochs, steps_per_epoch=len(train_loader))
        
        # [策略] Stage 1 通常不需要 EMA (因為變化太快)，Stage 2 再開啟
        use_ema = (stage['target'] == 'all')
        
        for epoch in range(stage_epochs):
            print(f"\nEpoch {epoch+1}/{stage_epochs}")
            
            current_mixup = mixup_fn if stage['target'] == 'all' else None
            current_ema = ema_model if use_ema else None
            
            train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, scheduler, 
                            mixup_fn=current_mixup, ema_model=current_ema)
            
            # [修改] 驗證邏輯：如果是在 Stage 2，我們優先看 EMA 的表現
            if use_ema:
                # 為了確保 EMA 內的 BatchNorm 統計數據正確，通常在驗證前要 update_bn
                # 但 AveragedModel 在 train loop 中如果有 forward 就會自動更新 (PyTorch 1.6+)
                # 這裡直接驗證 EMA 模型
                val_f1 = validate(val_loader, ema_model, criterion, device, train_dataset.classes, epoch=epoch+1, title="EMA")
                
                # 順便看一下原始模型表現 (可選)
                # validate(val_loader, model, criterion, device, train_dataset.classes, epoch=epoch+1, title="Raw")
            else:
                val_f1 = validate(val_loader, model, criterion, device, train_dataset.classes, epoch=epoch+1, title="Raw")

            is_best = val_f1 > global_best_f1
            if is_best:
                global_best_f1 = val_f1
                print(f"⭐ New Best F1: {global_best_f1:.4f}")

            # 存檔時，如果是 Stage 2，記得存 EMA 的權重
            state_dict_to_save = ema_model.module.state_dict() if use_ema else model.state_dict()
            
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': state_dict_to_save, # 存的是 EMA 權重
                'best_f1': global_best_f1,
            }, is_best, args.save_dir, filename=f"{stage['name']}_last.pth")

    print(f"\n✅ Training Completed. Best F1: {global_best_f1:.4f}")

if __name__ == "__main__":
    main()
