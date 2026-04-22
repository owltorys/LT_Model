import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2 
import torchvision.models as models 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
from PIL import Image
from torch.ao.quantization import get_default_qat_qconfig_mapping, prepare_qat
from torch.cuda.amp import autocast, GradScaler

def get_args():
    parser = argparse.ArgumentParser(description='ShuffleNetV2 Knowledge Distillation (QAT + GroupKFold + AutoML)')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--temperature', type=float, default=2.0, help='KD Temperature T')
    parser.add_argument('--teacher-width-mult', type=float, default=2.0)
    parser.add_argument('--teacher-model-path', type=str, default='models/checkpoints/teacher_model/FINAL_TEACHER.pth')
    parser.add_argument('--target-conv5-channels', type=int, default=960)
    parser.add_argument('--save-dir', type=str, default='models/checkpoints/student_model')
    parser.add_argument('--q-frac-weight', type=int, default=8)
    parser.add_argument('--q-frac-act', type=int, default=8)
    parser.add_argument('--seed', default=24, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--full-csv', default='DowDwen_set_resized.csv')
    parser.add_argument('--k-folds', default=5, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--auto-pipeline', action='store_true')
    parser.add_argument('--qat-start-epoch', type=int, default=40, help='Epoch to enable Fake Quantize')
    
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
    return torch.tensor((neg_counts + 1e-5) / (pos_counts + 1e-5), dtype=torch.float32)

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

class MultiLabelDistillLoss(nn.Module):
    def __init__(self, pos_weights, alpha=2.0, temperature=2.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.alpha = alpha
        self.T = temperature

    def forward(self, s_logits, t_logits, targets):
        hard_loss = self.bce_loss(s_logits, targets)
        
        t_pred = torch.sigmoid(t_logits / self.T)
        soft_loss = self.bce_loss(s_logits / self.T, t_pred) * (self.T ** 2)
        
        return hard_loss + self.alpha * soft_loss, hard_loss, soft_loss

def create_deploy_model(student_model):
    from copy import deepcopy
    deploy_model = deepcopy(student_model)
    deploy_model.cpu()
    deploy_model.eval()
    
    for name, module in deploy_model.named_modules():
        if type(module) == torch.ao.quantization.FakeQuantize:
            module.disable_fake_quant()
            module.disable_observer()
    
    return deploy_model

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

def train_one_epoch(train_loader, teacher, student, criterion, optimizer, device, scaler, scheduler, mixup_fn, qat_active):
    student.train()
    if qat_active:
        student.apply(torch.ao.quantization.enable_fake_quant)
        student.apply(torch.ao.quantization.enable_observer)
    else:
        student.apply(torch.ao.quantization.disable_fake_quant)
        student.apply(torch.ao.quantization.disable_observer)
        
    losses = AverageMeter()
    h_losses = AverageMeter()
    s_losses = AverageMeter()
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
            
        with torch.no_grad():
            t_logits = teacher(images)
            
        optimizer.zero_grad()
        with autocast():
            s_logits = student(images)
            loss, h_loss, s_loss = criterion(s_logits, t_logits, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        losses.update(loss.item(), images.size(0))
        h_losses.update(h_loss.item(), images.size(0))
        s_losses.update(s_loss.item(), images.size(0))
        
        if i % (len(train_loader) // 3) == 0 and i > 0:
            print(f"   Step [{i}/{len(train_loader)}] Loss: {losses.val:.4f} (H:{h_losses.val:.4f} S:{s_losses.val:.4f})")

def validate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_targets = []
    
    with torch.inference_mode():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            # Validation 時使用不帶溫度的基礎效能
            outputs = model(images)
            loss = criterion.bce_loss(outputs, targets)
            losses.update(loss.item(), images.size(0))
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    best_th = find_best_thresholds(all_targets, all_probs)
    
    preds = np.zeros_like(all_probs)
    for c in range(4):
        preds[:, c] = (all_probs[:, c] > best_th[c]).astype(int)

    f1_macro = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    print(f"   Loss: {losses.avg:.4f} | Dynamic Thresh F1: {f1_macro:.4f}")
    best_th_logit = [prob_to_logit(th) for th in best_th]
    return f1_macro, best_th, best_th_logit

def run_distillation_session(args, train_df, val_df, session_name="Fold", is_final=False):
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

    # --- Load Teacher ---
    print(f"📖 Loading Teacher (X{args.teacher_width_mult})...")
    if args.teacher_width_mult == 1.0:
        teacher = models.shufflenet_v2_x1_0(weights=None)
    elif args.teacher_width_mult == 1.5:
        teacher = models.shufflenet_v2_x1_5(weights=None)
    elif args.teacher_width_mult == 2.0:
        teacher = models.shufflenet_v2_x2_0(weights=None)
    else:
        raise ValueError("Unsupported teacher_width_mult")
    
    teacher.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(teacher.fc.in_features, 4))
    try:
        t_checkpoint = torch.load(args.teacher_model_path, map_location=device)
        teacher.load_state_dict(t_checkpoint.get('state_dict', t_checkpoint))
        print("✅ Teacher weights loaded.")
    except Exception as e:
        print(f"⚠️ Could not load Teacher weights: {e}")
    teacher = teacher.to(device).eval()

    # --- Load Student ---
    print(f"🧒 Initializing Student (X0.5 QAT)...")
    student = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    student.fc = nn.Linear(student.fc.in_features, 4)
    # Target Conv5 pruning logic could be added here if needed, keeping simple.
    
    qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
    student.train()
    student = prepare_qat(student, qconfig_mapping)
    student = student.to(device)

    pos_weights = get_pos_weights(train_df).to(device)
    criterion = MultiLabelDistillLoss(pos_weights, alpha=args.alpha, temperature=args.temperature)
    scaler = GradScaler()
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    stages = [
        {'name': 'QAT_Warmup', 'lr_factor': 1.0, 'target': 'classifier', 'epochs': 5},
        {'name': 'Distill_QAT', 'lr_factor': 0.1, 'target': 'all', 'epochs': args.epochs},
    ]

    best_f1 = 0.0
    best_epoch_global = 0
    best_th_logit = None
    
    os.makedirs(args.save_dir, exist_ok=True)

    global_epoch = 0
    for stage in stages:
        print(f"\n🔔 Stage: {stage['name']}...")
        set_trainable_layers(student, stage['target'])
        current_lr = args.learning_rate * stage['lr_factor']
        stage_epochs = stage['epochs']
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=current_lr, 
                                                  epochs=stage_epochs, steps_per_epoch=len(train_loader))
        
        use_mixup = (stage['target'] == 'all')
        
        for epoch in range(stage_epochs):
            global_epoch += 1
            print(f"\nEpoch {epoch+1}/{stage_epochs} (Global {global_epoch})")
            
            # 延遲量化機制 (QAT Start Epoch 控制)
            qat_active = (global_epoch >= args.qat_start_epoch)
            if global_epoch == args.qat_start_epoch:
                print("⚡ [Fake Quantize ENABLED] Switching to exact QAT emulation!")
            
            train_one_epoch(train_loader, teacher, student, criterion, optimizer, device, scaler, scheduler, 
                            mixup_fn if use_mixup else None, qat_active=qat_active)
            
            if not is_final and use_mixup:
                deploy_model = create_deploy_model(student)
                val_f1, _, th_logit = validate(val_loader, deploy_model, criterion, device)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch_global = epoch + 1
                    best_th_logit = th_logit
                    torch.save(student.state_dict(), os.path.join(args.save_dir, f"{session_name}_student_best.pth"))
                    print(f"⭐ New Best Student F1: {best_f1:.4f}")

    if is_final:
        torch.save(student.state_dict(), os.path.join(args.save_dir, "FINAL_STUDENT_QAT.pth"))
        print(f"\n✅ 100% Final QAT Distillation Completed. Model saved to FINAL_STUDENT_QAT.pth")
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
    groups = df_all['filename'].apply(lambda x: x.split('/')[1] if '/' in x else 'unknown').values

    if args.auto_pipeline:
        print("🤖 [Auto-Pipeline] Starting DK GroupKFold Search & Retrain...")
        gkf = GroupKFold(n_splits=args.k_folds)
        
        fold_best_epochs = []
        fold_best_th_logits = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_all, groups=groups)):
            train_df = df_all.iloc[train_idx]
            val_df = df_all.iloc[val_idx]
            _, best_ep, best_th_l = run_distillation_session(args, train_df, val_df, session_name=f"DK_Fold_{fold}")
            
            fold_best_epochs.append(best_ep)
            fold_best_th_logits.append(best_th_l)
            
        avg_epoch = int(np.mean(fold_best_epochs))
        avg_th_logits = np.mean(fold_best_th_logits, axis=0)
        
        print(f"\n{'='*50}\n🎯 AutoML KD Search Completed!\n{'='*50}")
        print(f"Averaged Optimal Stage-2 Epoch: {avg_epoch}")
        print(f"Averaged Logit Thresholds: {avg_th_logits}")
        
        args.epochs = avg_epoch
        run_distillation_session(args, df_all, None, session_name="FINAL_DK_100_PERCENT", is_final=True)
        
        logs = pd.DataFrame({'Class': ['Hand', 'Tool', 'Block', 'Safe_Operation'], 'Logit_Threshold': avg_th_logits})
        logs.to_csv(os.path.join(args.save_dir, "student_optimal_thresholds.csv"), index=False)
        print("💾 Saved Student optimal thresholds to student_optimal_thresholds.csv")

    else:
        print(f"單獨執行 DK GroupKFold 的第 {args.fold} 折...")
        gkf = GroupKFold(n_splits=args.k_folds)
        splits = list(gkf.split(df_all, groups=groups))
        if args.fold >= len(splits):
            raise ValueError(f"Fold {args.fold} is out of bounds for k_folds={args.k_folds}")
            
        train_idx, val_idx = splits[args.fold]
        train_df = df_all.iloc[train_idx]
        val_df = df_all.iloc[val_idx]
        run_distillation_session(args, train_df, val_df, session_name=f"DK_Fold_{args.fold}")

if __name__ == "__main__":
    main()
