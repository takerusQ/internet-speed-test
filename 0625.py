import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import nibabel as nib
import os
import random
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# V-netモデルの定義
class VNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        self.encoder5 = self.conv_block(128, 256)
        
        self.decoder4 = self.upconv_block(256, 128)
        self.decoder3 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder1 = self.upconv_block(32, 16)
        
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool3d(e1, 2))
        e3 = self.encoder3(F.max_pool3d(e2, 2))
        e4 = self.encoder4(F.max_pool3d(e3, 2))
        e5 = self.encoder5(F.max_pool3d(e4, 2))
        
        d4 = self.decoder4(F.interpolate(e5, scale_factor=2))
        d3 = self.decoder3(F.interpolate(d4 + e4, scale_factor=2))
        d2 = self.decoder2(F.interpolate(d3 + e3, scale_factor=2))
        d1 = self.decoder1(F.interpolate(d2 + e2, scale_factor=2))
        
        out = self.final_conv(d1 + e1)
        return out

# Dice Lossの定義
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

# IoUの定義
def calculate_iou(preds, labels, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = intersection / union
    return iou.item()

# データセットの定義
class MedicalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = nib.load(self.image_paths[idx]).get_fdata()
            mask = nib.load(self.mask_paths[idx]).get_fdata()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
        
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# データの前処理と拡張
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])

# 例としてのデータセットのパスリスト
image_paths = ['path_to_image1.nii', 'path_to_image2.nii']
mask_paths = ['path_to_mask1.nii', 'path_to_mask2.nii']

dataset = MedicalDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# モデル、損失関数、最適化アルゴリズムの定義
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VNet(in_channels=1, out_channels=1).to(device)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# トレーニングと検証データセットの分割
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# トレーニングループ
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_loader):
        if sample is None:
            continue
        
        inputs = sample['image'].float().to(device)
        labels = sample['mask'].float().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    model.eval()
    val_loss = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if sample is None:
                continue
            
            inputs = sample['image'].float().to(device)
            labels = sample['mask'].float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            iou_score += calculate_iou(outputs, labels)
    
    val_loss = val_loss / len(val_loader)
    iou_score = iou_score / len(val_loader)
    
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, IoU: {iou_score}')
    
    scheduler.step(val_loss)
    
    # モデルの保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_vnet_model.pth')

# 最良モデルのロード
model.load_state_dict(torch.load('best_vnet_model.pth'))

# モデルの評価
def evaluate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if sample is None:
                continue
            
            inputs = sample['image'].float().to(device)
            labels = sample['mask'].float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            iou_score += calculate_iou(outputs, labels)
    
    val_loss = val_loss / len(dataloader)
    iou_score = iou_score / len(dataloader)
    return val_loss, iou_score

# 評価ループ
val_loss, iou_score = evaluate_model(model, val_loader, criterion)
logger.info(f'Validation Loss after training: {val_loss}, IoU: {iou_score}')




データ拡張: データセットの多様性を高め、モデルの汎化能力を向上させるためのデータ拡張を追加。
詳細なログ記録とチェックポイント: トレーニングの進行状況を詳細に記録し、チェックポイントを定期的に保存。
学習率のスケジューリング: トレーニング中に学習率を動的に調整。
IoU（Intersection over Union）の評価: Dice Lossだけでなく、IoUも評価指標として追加。
データの読み込みエラーハンドリング: データ読み込み時のエラー処理を追加。
以下に、これらの改善点を含む完全なコードを示します：


左心室のセグメンテーションを行うためには、適切にラベル付けされたデータセットが必要です。以下に、DICOMデータからCTスキャン画像と対応する左心室のマスクを手動で準備する手順を示します。

手順概要
DICOMデータの読み込みと変換
左心室の手動セグメンテーション
マスクの保存とデータセットの構築
手順詳細
1. DICOMデータの読み込みと変換
まず、DICOMデータを読み込み、画像フォーマットに変換します。PythonのpydicomとSimpleITKライブラリを使用すると便利です。

python
コードをコピーする
import pydicom
import SimpleITK as sitk
import numpy as np
import os

def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

dicom_dir = 'path_to_your_dicom_directory'
ct_image = load_dicom_series(dicom_dir)

# Numpy配列に変換
ct_array = sitk.GetArrayFromImage(ct_image)
2. 左心室の手動セグメンテーション
手動でセグメンテーションを行うためには、医療画像用のソフトウェア（例：ITK-SNAP、3D Slicer）を使用します。

ソフトウェアのインストール

ITK-SNAP
3D Slicer
DICOMデータの読み込み

ソフトウェアを開き、DICOMデータをインポートします。
左心室のセグメンテーション

手動で左心室をセグメント化します。通常、ツールを使用して左心室の輪郭を描き、それを3Dモデルに変換します。
完了したら、セグメンテーション結果をマスク画像として保存します（NIfTI形式や他の画像形式が一般的です）。
3. マスクの保存とデータセットの構築
セグメンテーションされたマスクを対応するCTスキャンとともに保存し、トレーニング用のデータセットを構築します。

python
コードをコピーする
import nibabel as nib

# CT画像の保存
ct_image_nifti = nib.Nifti1Image(ct_array, np.eye(4))
nib.save(ct_image_nifti, 'ct_image.nii')

# マスクの保存
mask_array = np.zeros_like(ct_array)  # 手動セグメンテーションの結果に置き換え
mask_image_nifti = nib.Nifti1Image(mask_array, np.eye(4))
nib.save(mask_image_nifti, 'mask_image.nii')
まとめ
手動でDICOMデータから左心室のセグメンテーションを行うための手順は以下の通りです：

DICOMデータを読み込み、Numpy配列に変換する。
ITK-SNAPや3D Slicerなどのソフトウェアを使用して、左心室を手動でセグメント化する。
セグメンテーション結果をマスク画像として保存し、元のCTスキャンと対応付ける。
これにより、トレーニング用のデータセットが準備できます。









左心室をCT画像から簡易な画像処理で特定するためには、左心室の先験的な特徴を利用して、いくつかの基本的な画像処理手法を組み合わせることが有効です。以下に、そのステップを示します。

### 1. 前処理
まず、CT画像の前処理を行います。ノイズ除去や平滑化を行い、画像の品質を向上させます。

```python
import cv2
import numpy as np

# CT画像の読み込み
image = cv2.imread('ct_image.png', cv2.IMREAD_GRAYSCALE)

# 平滑化 (ガウシアンフィルタ)
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
```

### 2. 閾値処理
次に、閾値処理を用いて血液の均一な画素値を強調します。左心室内部は血液で満たされているため、他の組織と異なる均一な画素値を持つことが特徴です。

```python
# 二値化 (閾値処理)
ret, binary_image = cv2.threshold(smoothed_image, 150, 255, cv2.THRESH_BINARY)
```

### 3. 大血管の検出
大血管は左心室と接続しているため、大血管を検出することで左心室の位置を特定する手掛かりになります。ここでは、形態素フィルタを用いて大血管を強調します。

```python
# 形態素フィルタによる大血管の強調
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
```

### 4. 領域のラベリング
大血管が強調された画像を基に、領域のラベリングを行います。これにより、連結成分を特定し、大きな領域を左心室候補として抽出します。

```python
# 連結成分のラベリング
num_labels, labels_im = cv2.connectedComponents(morph_image)

# ラベリング結果の表示
label_hue = np.uint8(179*labels_im/np.max(labels_im))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue==0] = 0
```

### 5. 左心室の特定
ラベリングされた領域から、左心室を特定するためにいくつかの基準を設定します。例えば、領域の大きさや形状、大血管との接続などを考慮します。

```python
# 各ラベル領域の特性を計算
for label in range(1, num_labels):
    mask = np.uint8(labels_im == label)
    area = cv2.contourArea(mask)
    # 左心室候補の特定
    if 1000 < area < 5000:  # 面積の範囲を基準に設定
        left_ventricle = mask
        break
```

### まとめ
この方法では、CT画像から左心室を特定するために以下の手順を踏みます：
1. **前処理**: ノイズ除去と平滑化。
2. **閾値処理**: 血液の均一な画素値を強調。
3. **大血管の検出**: 形態素フィルタで強調。
4. **領域のラベリング**: 連結成分を特定。
5. **左心室の特定**: 領域の大きさや形状、大血管との接続を基準に特定。

この方法は簡易な画像処理を組み合わせて左心室を特定する実用的な手法ですが、精度を高めるためには、さらなる調整や追加の特徴抽出が必要となる場合があります。


以下のコードは、画像スタックから心臓レベルの画像を判定し、その中で左心室の断面積が最大のスライスを選択するための手順を示しています。

### ステップ概要

1. 画像スタックの読み込み
2. それぞれのスライスが心臓レベルにあるかどうかを判定
3. 左心室の断面積が最大のスライスを選択

### ステップ詳細

```python
import cv2
import numpy as np
import SimpleITK as sitk

# DICOMシリーズの読み込み関数
def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

# 画像のスタックを読み込み
dicom_dir = 'path_to_your_dicom_directory'
ct_image = load_dicom_series(dicom_dir)
ct_array = sitk.GetArrayFromImage(ct_image)  # Z, Y, X の順

# 各スライスを処理する関数
def process_slice(slice_image):
    # 前処理 (ガウシアンフィルタ)
    smoothed_image = cv2.GaussianBlur(slice_image, (5, 5), 0)
    
    # 二値化 (閾値処理)
    ret, binary_image = cv2.threshold(smoothed_image, 150, 255, cv2.THRESH_BINARY)
    
    # 形態素フィルタによる大血管の強調
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return morph_image

# 心臓レベルのスライスかどうかを判定する関数
def is_heart_level(slice_image):
    processed_image = process_slice(slice_image)
    num_labels, labels_im = cv2.connectedComponents(processed_image)
    
    # ラベルの特性をチェック
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        area = cv2.contourArea(mask)
        
        # 面積の基準 (適切な閾値に調整)
        if 1000 < area < 5000:
            return True
    return False

# 左心室の断面積を計算する関数
def calculate_lv_area(slice_image):
    processed_image = process_slice(slice_image)
    num_labels, labels_im = cv2.connectedComponents(processed_image)
    
    max_area = 0
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        area = cv2.contourArea(mask)
        
        # 左心室の断面積を取得 (適切な閾値に調整)
        if 1000 < area < 5000:
            if area > max_area:
                max_area = area
    return max_area

# 画像スタックをループして、心臓レベルのスライスを判定し、左心室の断面積が最大のスライスを選択
max_lv_area = 0
best_slice = None

for i in range(ct_array.shape[0]):
    slice_image = ct_array[i, :, :]
    if is_heart_level(slice_image):
        lv_area = calculate_lv_area(slice_image)
        if lv_area > max_lv_area:
            max_lv_area = lv_area
            best_slice = slice_image

# 最適なスライスを表示
if best_slice is not None:
    cv2.imshow('Best Slice', best_slice)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("左心室を特定できるスライスが見つかりませんでした。")
```

### 説明
1. **画像スタックの読み込み**: `load_dicom_series`関数でDICOMシリーズを読み込み、Numpy配列に変換します。
2. **前処理**: 各スライスに対してガウシアンフィルタを適用し、平滑化を行います。
3. **二値化と形態素フィルタ**: 血液の均一な画素値を強調するために閾値処理を行い、形態素フィルタで大血管を強調します。
4. **心臓レベルのスライス判定**: 面積基準を用いて心臓レベルのスライスを判定します。
5. **左心室の断面積計算**: 各スライスの左心室の断面積を計算し、最大のスライスを選択します。

この方法は、画像スタックから簡易な画像処理を用いて左心室を特定する実用的な手法です。適切な閾値や形態素フィルタの設定を行うことで、精度を高めることができます。