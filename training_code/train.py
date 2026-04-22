import argparse
import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2 
import torchvision.models as models 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

def get_args():
    parser = argparse.ArgumentParser(description='ShuffleNetV2 Teacher Training (EMA + GroupKFold + AutoML)')
    parser.add_argument('--width-mult', default=2.0, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float) 
    parser.add_argument('--save-dir', default='models/checkpoints/teacher_model')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--full-csv', default='DowDwen_set_resized.csv')
    parser.add_argument('--k-folds', default=5, type=int)
    parser.add_argument('--fold', default=0, type=int, help='Which fold to run if not auto-pipeline')
    parser.add_argument('--auto-pipeline', action='store_true', help='Run K-folds auto and 100% retrain.')
    parser.add_argument('--seed', default=24, type=int)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    args, _ = parser.parse_known_args()
    return args

class MultiLabelDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = ['Hand', 'Tool', 'Block', 'Safe_Operation']
        self.classes = self.label_cols 

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        labels = row[self.label_cols].values.astype(np.float32)
        target = torch.tensor(labels)
        if self.transform: image = self.transform(image)
        return image, target

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_pos_weights(df):
    label_cols = ['Hand', 'Tool', 'Block', 'Safe_Operation']
    labels = df[label_cols].values
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(df) - pos_counts
    pos_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)
    return torch.tensor(pos_weights, dtype=torch.float32)

def set_trainable_layers(model, unfreeze_target):
    for param in model.parameters(): param.requires_grad = False
    if unfreeze_target == 'classifier':
        if hasattr(model, 'fc'):
            for p in model.fc.parameters(): p.requires_grad = True
    elif unfreeze_target == 'all':
        for param in model.parameters(): param.requires_grad = True

def get_mixup_cutmix_transforms(num_classes):
    return v2.RandomChoice([
        v2.RandomApply([v2.MixUp(num_classes=num_classes, alpha=0.4)], p=1.0),
        v2.RandomApply([v2.CutMix(num_classes=num_classes, alpha=1.0)], p=1.0)
    ])

def smooth_labels(targets, smoothing):
    return targets * (1.0 - smoothing) + 0.5 * smoothing

def prob_to_logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))

def find_best_thresholds(all_targets, all_probs):
    best_th = np.full(4, 0.5)
    for c in range(4):
        best_f1 = 0
        best_t = 0.5
        for th in np.arange(0.1, 0.95, 0.05):
            preds = (all_probs[:, c] > th).astype(int)
            f1 = f1_score(all_targets[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = th
        best_th[c] = best_t
    return best_th

def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, scheduler, mixup_fn, ema_model, smoothing):
    model.train()
    losses = AverageMeter()
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            targets_smoothed = smooth_labels(targets, smoothing)
            loss = criterion(outputs, targets_smoothed)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ema_model is not None:
            ema_model.update_parameters(model)
        if scheduler is not None:
            scheduler.step()
        
        losses.update(loss.item(), images.size(0))
        if i % (len(train_loader) // 3) == 0 and i > 0:
            print(f"   Step [{i}/{len(train_loader)}] Loss: {losses.val:.4f} LR: {optimizer.param_groups[0]['lr']:.2e}")

def validate(val_loader, model, criterion, device, class_names):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_targets = []
    
    with torch.inference_mode():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # 動態閾值搜尋
    best_th = find_best_thresholds(all_targets, all_probs)
    
    # 套用最佳閾值
    preds = np.zeros_like(all_probs)
    for c in range(4):
        preds[:, c] = (all_probs[:, c] > best_th[c]).astype(int)

    f1_macro = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    print(f"   Loss: {losses.avg:.4f} | Dynamic Thresh F1: {f1_macro:.4f}")
    print(f"   Best Thresholds (Prob): Hand={best_th[0]:.2f}, Tool={best_th[1]:.2f}, Block={best_th[2]:.2f}, Safe={best_th[3]:.2f}")
    
    # 計算轉化為佈署使用的 Logit 閾值
    best_th_logit = [prob_to_logit(th) for th in best_th]
    return f1_macro, best_th, best_th_logit

def run_training_session(args, train_df, val_df, session_name="Fold", is_final=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}\n🚀 Starting Session: {session_name}\n{'='*50}")
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

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

    mixup_fn = get_mixup_cutmix_transforms(num_classes=4)

    train_dataset = MultiLabelDataset(train_df, args.data_root, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
                              
    if not is_final:
        val_dataset = MultiLabelDataset(val_df, args.data_root, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True)

    print(f"🔄 Initializing Teacher Model (X{args.width_mult})...")
    if args.width_mult == 0.5:
        model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    elif args.width_mult == 1.0:
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    elif args.width_mult == 1.5:
        model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.DEFAULT)
    elif args.width_mult == 2.0: 
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
    else:
        raise ValueError("Unsupported width_mult")

    num_classes = 4 
    in_features = model.fc.in_features
    # [修改] 加入 Dropout 防過擬合
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)).to(device)
    pos_weights = get_pos_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    scaler = GradScaler()

    stages = [
        {'name': 'Stage 1_Classifier', 'lr_factor': 1.0, 'target': 'classifier', 'epochs': 5},
        {'name': 'Stage 2_Full',       'lr_factor': 0.1, 'target': 'all',        'epochs': args.epochs},
    ]

    best_f1 = 0.0
    best_epoch_global = 0
    best_th_logit = None
    
    os.makedirs(args.save_dir, exist_ok=True)

    for stage in stages:
        print(f"\n🔔 Stage: {stage['name']}...")
        set_trainable_layers(model, stage['target'])
        current_lr = args.lr * stage['lr_factor']
        stage_epochs = stage['epochs']
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=current_lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=current_lr, 
                                                  epochs=stage_epochs, steps_per_epoch=len(train_loader))
        
        use_ema = (stage['target'] == 'all')
        
        for epoch in range(stage_epochs):
            print(f"\nEpoch {epoch+1}/{stage_epochs}")
            current_mixup = mixup_fn if use_ema else None
            
            train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, scheduler, 
                            current_mixup, ema_model if use_ema else None, args.label_smoothing)
            
            if not is_final and use_ema:
                val_f1, _, th_logit = validate(val_loader, ema_model, criterion, device, train_dataset.classes)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch_global = epoch + 1
                    best_th_logit = th_logit
                    torch.save({'state_dict': ema_model.module.state_dict()}, os.path.join(args.save_dir, f"{session_name}_best.pth"))
                    print(f"⭐ New Best F1: {best_f1:.4f}")

    if is_final:
        # 最終訓練直接儲存最後的 EMA 權重
        torch.save({'state_dict': ema_model.module.state_dict()}, os.path.join(args.save_dir, "FINAL_TEACHER.pth"))
        print(f"\n✅ 100% Final Training Completed. Model saved to FINAL_TEACHER.pth")
        return None, None, None

    print(f"\n✅ Session {session_name} Completed. Best F1: {best_f1:.4f} at Epoch: {best_epoch_global}")
    return best_f1, best_epoch_global, best_th_logit

def main():
    args = get_args()
    set_seed(args.seed)
    
    csv_path = os.path.join(args.data_root, args.full_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")
        
    df_all = pd.read_csv(csv_path)
    # 取出子資料夾名稱作為 Group (例如: Safe_Operation/o11 -> o11)
    groups = df_all['filename'].apply(lambda x: x.split('/')[1] if '/' in x else 'unknown').values

    if args.auto_pipeline:
        print("🤖 [Auto-Pipeline] Starting GroupKFold Search & Retrain...")
        gkf = GroupKFold(n_splits=args.k_folds)
        
        fold_best_epochs = []
        fold_best_th_logits = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_all, groups=groups)):
            train_df = df_all.iloc[train_idx]
            val_df = df_all.iloc[val_idx]
            _, best_ep, best_th_l = run_training_session(args, train_df, val_df, session_name=f"Fold_{fold}")
            
            fold_best_epochs.append(best_ep)
            fold_best_th_logits.append(best_th_l)
            
        avg_epoch = int(np.mean(fold_best_epochs))
        avg_th_logits = np.mean(fold_best_th_logits, axis=0)
        
        print(f"\n{'='*50}\n🎯 AutoML Search Completed!\n{'='*50}")
        print(f"Averaged Optimal Stage-2 Epoch: {avg_epoch}")
        print(f"Averaged Logit Thresholds: {avg_th_logits}")
        
        # 覆寫為最佳 Epoch
        args.epochs = avg_epoch
        run_training_session(args, df_all, None, session_name="FINAL_100_PERCENT", is_final=True)
        
        # 儲存最佳參數給 DK.py 使用
        logs = pd.DataFrame({'Class': ['Hand', 'Tool', 'Block', 'Safe_Operation'], 'Logit_Threshold': avg_th_logits})
        logs.to_csv(os.path.join(args.save_dir, "optimal_thresholds.csv"), index=False)
        print("💾 Saved optimal thresholds to optimal_thresholds.csv")

    else:
        print(f"單獨執行 GroupKFold 的第 {args.fold} 折...")
        gkf = GroupKFold(n_splits=args.k_folds)
        splits = list(gkf.split(df_all, groups=groups))
        if args.fold >= len(splits):
            raise ValueError(f"Fold {args.fold} is out of bounds for k_folds={args.k_folds}")
            
        train_idx, val_idx = splits[args.fold]
        train_df = df_all.iloc[train_idx]
        val_df = df_all.iloc[val_idx]
        run_training_session(args, train_df, val_df, session_name=f"Fold_{args.fold}")

if __name__ == "__main__":
    main()
