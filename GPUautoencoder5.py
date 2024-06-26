import os
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# OpenMPエラー回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# 再構築されたデータを元のCT値に戻す関数
def denormalize_ct_image(normalized_image, window_width, window_level):
    min_value = window_level - (window_width / 2)
    max_value = window_level + (window_width / 2)
    denormalized_image = normalized_image * (max_value - min_value) + min_value
    return denormalized_image

# データの準備
def load_and_preprocess_dicom_files(dicom_dir, dicomfilepaths):
    CTfiles = []
    normalizedCTfiles = []
    window_width = None
    window_level = None
    for filepath in dicomfilepaths:
        full_path = os.path.join(dicom_dir, filepath)
        d = pydicom.dcmread(full_path)
        hu = d.pixel_array
        window_width = d.WindowWidth
        window_level = d.WindowCenter
        nhu = normalize_ct_image(hu, window_width=window_width, window_level=window_level)
        CTfiles.append(hu)
        normalizedCTfiles.append(nhu)
    
    mizumashi_of_CTfiles = np.array(CTfiles).reshape(-1, 512, 512, 1)
    mizumashi_of_normalizedCTfiles = np.array(normalizedCTfiles).reshape(-1, 512, 512, 1)
    return mizumashi_of_CTfiles, mizumashi_of_normalizedCTfiles, window_width, window_level

# 正規化したデータを取得
_, mizumashi_of_normalizedCTfiles, window_width, window_level = load_and_preprocess_dicom_files(example_person_dicom_dir_path, dicomfilepaths)

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

# 損失を記録するリスト
train_losses = []
val_losses = []

# 学習ループ
for epoch in range(EPOCHS):
    total_train_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        z = encoder(x)
        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(dataloader)
    train_losses.append(avg_train_loss)
    
    # Validation lossの計算
    if (epoch + 1) % 5 == 0:
        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                z = encoder(x)
                x_hat = decoder(z)
                val_loss = criterion(x_hat, x)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(dataloader)
        val_losses.append(avg_val_loss)
        print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        encoder.train()
        decoder.train()

# 画像の復元
z_points = encoder(train_images)
reconst_images = decoder(z_points)

# 損失のプロット関数
def plot_loss_curve(train_losses, val_losses, save_path="results"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(save_path, r"C:\Users\kota\Desktop\suyama\loss_curve.png"))
    plt.show()

# 画像のプロット関数
def plot_images(original, decoded, window_width, window_level, n=5, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(15, 10))
    for i in range(n):
        # オリジナル画像
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # 再構築された画像
        denormalized_image = denormalize_ct_image(decoded[i].permute(1, 2, 0).squeeze().numpy(), window_width, window_level)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(denormalized_image, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # 差分画像
        difference = denormalized_image - original[i].permute(1, 2, 0).squeeze().numpy()
        ax = plt.subplot(3, n, i + 1 + 2 * n)        
        plt.imshow(difference, cmap="bwr", vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis("off")

    # 画像を保存
    plt.savefig(os.path.join(save_path, r"C:\Users\kota\Desktop\suyama\reconstructed_images.png"))
    plt.show()

# 損失曲線をプロットおよび保存
plot_loss_curve(train_losses, val_losses)

# トレーニングデータから最初の5つの画像を表示および保存
plot_images(train_images, reconst_images.detach(), window_width, window_level, n=5)