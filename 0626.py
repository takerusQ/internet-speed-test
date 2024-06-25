import os
import numpy as np
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# データパスの設定
example_person_dicom_dir_path = r"C:\Users\kota\Desktop\suyama\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"

# DICOMファイルの読み込みとソート
dicomfilepaths = sorted(os.listdir(example_person_dicom_dir_path))

# 正規化関数
def normalize_ct_image(image, window_width, window_level):
    min_value = window_level - (window_width / 2)
    max_value = window_level + (window_width / 2)
    normalized_image = (image - min_value) / (max_value - min_value)
    normalized_image[normalized_image < 0] = 0
    normalized_image[normalized_image > 1] = 1
    return normalized_image

# データの準備
def load_and_preprocess_dicom_files(dicom_dir, dicomfilepaths):
    CTfiles = []
    normalizedCTfiles = []
    for filepath in dicomfilepaths:
        full_path = os.path.join(dicom_dir, filepath)
        d = pydicom.dcmread(full_path)
        hu = d.pixel_array
        nhu = normalize_ct_image(hu, window_width=d.WindowWidth, window_level=d.WindowCenter)
        CTfiles.append(hu)
        normalizedCTfiles.append(nhu)
    
    mizumashi_of_CTfiles = np.array(CTfiles).reshape(-1, 512, 512, 1)
    mizumashi_of_normalizedCTfiles = np.array(normalizedCTfiles).reshape(-1, 512, 512, 1)
    return mizumashi_of_CTfiles, mizumashi_of_normalizedCTfiles

# 正規化したデータを取得
_, mizumashi_of_normalizedCTfiles = load_and_preprocess_dicom_files(example_person_dicom_dir_path, dicomfilepaths)

# 学習データの準備
train_images = torch.from_numpy(mizumashi_of_normalizedCTfiles[20:40]).permute(0, 3, 1, 2).float()

# ハイパーパラメータの設定
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 100
EPOCHS = 20

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 128 * 128, Z_DIM)  # 修正: 64チャネル、128x128ピクセル

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# デコーダの定義
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(Z_DIM, 64 * 128 * 128)  # 修正: 64チャネル、128x128ピクセル
        self.conv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 128, 128)  # Reshape
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # 最終出力は0~1にするためsigmoidを使用
        return x

# データローダの作成
dataset = TensorDataset(train_images)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# モデルの定義
encoder = Encoder()
decoder = Decoder()

# 最適化関数
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# 損失関数
criterion = nn.MSELoss()

# 学習ループ
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        z = encoder(x)
        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}')

# 画像の復元
z_points = encoder(train_images)
reconst_images = decoder(z_points)

# 画像のプロット関数
def plot_images(original, decoded, n=5):
    plt.figure(figsize=(15, 10))
    for i in range(n):
        # オリジナル画像
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # 再構築された画像
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded[i].permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # 差分画像
        difference = decoded[i] - original[i]
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(difference.permute(1, 2, 0).squeeze(), cmap="bwr", vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis("off")
    plt.show()

# トレーニングデータから最初の5つの画像を表示
plot_images(train_images, reconst_images.detach(), n=5)