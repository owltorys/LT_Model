import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, f1_score
import numpy as np
import pandas as pd
from PIL import Image

# 引入 QAT 必要工具
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, QConfig, FakeQuantize, ObserverBase, fuse_modules
from torch.nn.utils.fusion import fuse_conv_bn_eval

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Test Student Model (X0.5 QAT)")
    parser.add_argument('--model-path', type=str, default="model/ntd/model_best_ntd1_1_1.pth", help='Path to best_student_qat.pth')
    parser.add_argument('--data-root', type=str, default="./DowDwen_set_resized", help='Dataset root directory')
    parser.add_argument('--test-csv', type=str, default="test.csv", help='CSV filename')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--target-conv5-channels', type=int, default=960, help='Pruned conv5 channels')
    parser.add_argument('--q-frac-weight', type=int, default=8, help='QAT weights fraction')
    parser.add_argument('--q-frac-act', type=int, default=8, help='QAT acts fraction')
    args, _ = parser.parse_known_args()
    
    return {
        'model_path': args.model_path,
        'data_root': args.data_root,
        'test_csv': args.test_csv,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_classes': 4,
        'class_names': ['Hand', 'Tool', 'Block', 'Safe_Operation'],
        'target_conv5_channels': args.target_conv5_channels,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std':  [0.229, 0.224, 0.225],
        'q_frac_weight': args.q_frac_weight,
        'q_frac_act': args.q_frac_act
    }

CONFIG = get_args()

# ==========================================
# 1. Dataset & Transform (模擬硬體)
# ==========================================
class HardwareSimulateTransform:
    """
    模擬 ZCU104 硬體輸入：
    1. 讀取圖片 RGB (0-255)
    2. 右移 8 bits (除以 256.0)
    3. 不做標準 Normalize (因為已經融合進模型權重)
    """
    def __call__(self, pic):
        img_tensor = transforms.functional.pil_to_tensor(pic).float()
        img_tensor = img_tensor / 256.0
        return img_tensor

class MultiLabelDataset(Dataset):
    def __init__(self, csv_filename, root_dir, transform=None):
        # 組合完整路徑
        csv_path = os.path.join(root_dir, csv_filename)
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = CONFIG['class_names']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image: {img_path}")
            image = Image.new('RGB', (128, 128), (0, 0, 0))

        labels = row[self.label_cols].values.astype(np.float32)
        target = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, target

# ==========================================
# 2. QAT 相關工具 (必須與訓練代碼一致)
# ==========================================
class StaticFixedPointObserver(ObserverBase):
    def __init__(self, frac_bits, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, **kwargs):
        super().__init__(dtype=dtype)
        self.frac_bits = frac_bits
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        scale_val = 1.0 / (2 ** frac_bits)
        self.register_buffer('fixed_scale', torch.tensor([scale_val]))
        self.register_buffer('fixed_zp', torch.tensor([0], dtype=torch.int32))

    def forward(self, x): return x
    def calculate_qparams(self): return self.fixed_scale, self.fixed_zp

def get_fixed_point_qconfig(frac_weight, frac_act):
    weight_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, frac_bits=frac_weight)
    act_fq = FakeQuantize.with_args(observer=StaticFixedPointObserver, quant_min=-32768, quant_max=32767, dtype=torch.qint32, qscheme=torch.per_tensor_symmetric, frac_bits=frac_act)
    return QConfig(activation=act_fq, weight=weight_fq)

class DistillQATModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = original_model

    def forward(self, x):
        x = self.quant(x)
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        x = x.mean([2, 3])
        x = self.model.fc(x)
        x = self.dequant(x)
        return x

# ==========================================
# 3. 模型結構重建工具 (與 DK.py 一致)
# ==========================================
def prune_shufflenet_conv5(model, target_channels=960):
    conv5_block = model.conv5
    conv = conv5_block[0]
    bn = conv5_block[1]
    # 注意：測試時我們只重建結構，權重會從 pth 載入
    new_conv = nn.Conv2d(conv.in_channels, target_channels, kernel_size=1, stride=1, padding=0, bias=False)
    new_bn = nn.BatchNorm2d(target_channels)
    model.conv5[0] = new_conv
    model.conv5[1] = new_bn
    model.fc = nn.Linear(target_channels, CONFIG['num_classes'])
    return model

# [新增] 必須與訓練時一樣，先把 Normalization 融合進 Conv1
def fuse_normalization_to_conv1(model, mean, std):
    print("🔨 Fusing Normalization (Mean/Std) into Conv1...")
    conv = model.conv1[0]
    # 這裡我們只做運算，不需回傳 model，因為是 inplace 修改
    mean_t = torch.tensor(mean).view(3, 1, 1).to(conv.weight.device)
    std_t = torch.tensor(std).view(3, 1, 1).to(conv.weight.device)
    with torch.no_grad():
        conv.weight.data.div_(std_t)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
        weight_sum = conv.weight.data.sum(dim=(2, 3))
        bias_adjustment = (weight_sum * mean_t.squeeze()).sum(dim=1)
        conv.bias.data.sub_(bias_adjustment)
    return model

def create_deploy_model(model):
    print("🔨 Converting to deploy mode (FusedConv + ReLU)...")
    deploy_model = copy.deepcopy(model)
    deploy_model.eval()
    try:
        import torch.ao.nn.intrinsic.qat as nniqat
    except ImportError:
        nniqat = None
        
    def _recursive_fuse(module):
        for name, child in module.named_children():
            try:
                if nniqat is not None and isinstance(child, nniqat.ConvBnReLU2d):
                    if hasattr(child, 'bn'):
                        fused_conv = fuse_conv_bn_eval(child, child.bn)
                    else:
                        fused_conv = child
                    if hasattr(child, 'weight_fake_quant'): fused_conv.weight_fake_quant = child.weight_fake_quant
                    replaced_module = nn.Sequential(fused_conv, nn.ReLU(inplace=True))
                    setattr(module, name, replaced_module)
                elif nniqat is not None and isinstance(child, nniqat.ConvBn2d):
                    if hasattr(child, 'bn'):
                        fused_conv = fuse_conv_bn_eval(child, child.bn)
                    else:
                        fused_conv = child
                    if hasattr(child, 'weight_fake_quant'): fused_conv.weight_fake_quant = child.weight_fake_quant
                    setattr(module, name, fused_conv)
                else:
                    _recursive_fuse(child)
            except Exception as e:
                pass
    _recursive_fuse(deploy_model)
    return deploy_model

# ==========================================
# 主程式
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")
    
    # 1. 準備測試資料
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        HardwareSimulateTransform()
    ])
    
    test_dataset = MultiLabelDataset(CONFIG['test_csv'], CONFIG['data_root'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    print(f"📚 Test Images: {len(test_dataset)}")

    # 2. 重建模型 (步驟必須與訓練完全一致！)
    print("🏗️ Reconstructing Student Model (X0.5)...")
    base_model = models.shufflenet_v2_x0_5(weights=None) # 不需預訓練權重，因為會覆蓋
    
    # 2.1 剪枝 Conv5
    base_model = prune_shufflenet_conv5(base_model, target_channels=CONFIG['target_conv5_channels'])
    
    # 2.2 [關鍵] 融合 Normalization 到 Conv1
    base_model = fuse_normalization_to_conv1(base_model, CONFIG['norm_mean'], CONFIG['norm_std'])
    
    model = DistillQATModel(base_model)
    
    # 2.3 設定 QConfig
    model.qconfig = get_fixed_point_qconfig(CONFIG['q_frac_weight'], CONFIG['q_frac_act'])
    
    # 2.4 執行融合 (重現 QAT 結構)
    model.eval()
    fuse_modules(model.model, [['conv1.0', 'conv1.1', 'conv1.2']], inplace=True)
    # [關鍵] 加入 Conv5 的 Fusion (與訓練時一致)
    fuse_modules(model.model, [['conv5.0', 'conv5.1']], inplace=True)
    
    for name, module in model.model.named_modules():
        if isinstance(module, models.shufflenetv2.InvertedResidual):
            for i in range(len(module.branch1)):
                if isinstance(module.branch1[i], nn.Conv2d):
                    fuse_modules(module.branch1, [str(i), str(i+1)], inplace=True) 
            for i in range(len(module.branch2)):
                if isinstance(module.branch2[i], nn.Conv2d):
                    if i+1 < len(module.branch2) and isinstance(module.branch2[i+1], nn.BatchNorm2d):
                        if i+2 < len(module.branch2) and isinstance(module.branch2[i+2], nn.ReLU):
                            fuse_modules(module.branch2, [str(i), str(i+1), str(i+2)], inplace=True)
                        else:
                            fuse_modules(module.branch2, [str(i), str(i+1)], inplace=True)
    
    model.train()
    prepare_qat(model, inplace=True)
    
    # 3. 載入權重
    print(f"🔄 Loading weights from {CONFIG['model_path']}...")
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ Error: Model file not found at {CONFIG['model_path']}")
        return

    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    try:
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except RuntimeError as e:
        print(f"⚠️ Error loading state_dict: {e}")
        print("提示：請檢查 best_student_qat.pth 是否包含 Conv5 的融合結構。")
        return
    
    # 4. 轉換為部署模式 (模擬硬體推論)
    model = create_deploy_model(model)
    model = model.to(device)
    model.eval()

    # 5. 執行推論
    all_preds = []
    all_targets = []
    
    print("🚀 Starting Inference...")
    with torch.inference_mode():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # 多標籤判斷：Logits > 0 (Sigmoid > 0.5)
            preds = (outputs > 0).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 6. 計算指標
    exact_match_acc = np.mean(np.all(all_preds == all_targets, axis=1)) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    
    print("\n📊 Test Results (Student X0.5 QAT - Deploy Mode):")
    print("   ----------------------------")
    print(f"   Exact Match Acc (全對率): {exact_match_acc:.2f}%")
    print(f"   Macro F1-Score: {f1_macro:.4f}")
    print("   ----------------------------")
    print("   Recall per Class:")
    for i, name in enumerate(CONFIG['class_names']):
        print(f"     - {name:<15}: {recall_per_class[i]*100:.2f}%")
    print("   ----------------------------")

if __name__ == "__main__":
    main()
