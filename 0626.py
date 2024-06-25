import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pydicom
import glob
import cv2
import os
import time

from pydicom.uid import UID
from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes, center_of_mass
from skimage.morphology import disk

#import plotly.express as px
#from ipywidgets import interact

import gc  # ガベージコレクションのインポート
import matplotlib.pyplot as plt  # グラフ表示用


import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



example_person_dicom_dir_path =r"C:\Users\kota\Desktop\suyama\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"


aaa = os.listdir(example_person_dicom_dir_path)
aaa = sorted(aaa)
dicomfilepaths = aaa
# 1人の1枚のdicom画像で確認していく
CTfiles = []
normalizedCTfiles = []
#CTfiles_resized = []


# ウィンドウ幅とウィンドウレベルを用いて正規化する関数
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

## 再構築された画像を元のCT値に戻す
#denormalized_ct_image = denormalize_ct_image(decoded_image, window_width= d.WindowCenter, window_level= d.WindowWidth)


#学習データの正規化
for k in range(len(dicomfilepaths)):
    example_slice_of_example_person_dicom_dir_path = \
    example_person_dicom_dir_path + "/" + dicomfilepaths[k]
    
    d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
    hu = d.pixel_array
    nhu = normalize_ct_image(hu, window_width= d.WindowWidth, window_level= d.WindowCenter)
    #hu_resized = cv2.resize(hu,dsize=None,fx=(d.ReconstructionDiameter / 512),fy=(d.ReconstructionDiameter / 512),interpolation=cv2.INTER_CUBIC)
    CTfiles.append(hu)
    normalizedCTfiles.append(nhu)
    #CTfiles_resized.append(hu_resized)

mizumashi_of_CTfiles = np.array(CTfiles)
mizumashi_of_normalizedCTfiles = np.array(normalizedCTfiles)
mizumashi_of_CTfiles=mizumashi_of_CTfiles.reshape(
    mizumashi_of_CTfiles.shape[0],
    mizumashi_of_CTfiles.shape[1],
    mizumashi_of_CTfiles.shape[2],1)
mizumashi_of_normalizedCTfiles=mizumashi_of_normalizedCTfiles.reshape(
    mizumashi_of_normalizedCTfiles.shape[0],
    mizumashi_of_normalizedCTfiles.shape[1],
    mizumashi_of_normalizedCTfiles.shape[2],1)



# 学習データの読み込み＆前処理
train_images = mizumashi_of_normalizedCTfiles[20:40,:,:,:]
train = []
for i in train_images:
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    train.append(image)

train = np.array(train)
train = train.astype('float32') / 255
train = torch.from_numpy(train).permute(0, 3, 1, 2)  # PyTorchは(バッチサイズ, チャネル, 高さ, 幅)の形式

# 学習用ハイパーパラメータ
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 100
EPOCHS = 20

# エンコーダ
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.fc = nn.Linear(64*75*75, Z_DIM)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# デコーダ
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(Z_DIM, 64*75*75)
        self.conv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 75, 75)  # Reshape
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # 最終出力は0~1にするためsigmoidを使用
        return x

# データローダの作成
dataset = TensorDataset(train)
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
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        z = encoder(x)
        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

#以降未完成

# 画像の復元
z_points = encoder(original_images)
reconst_images = decoder(z_points)

# 元画像との差分を計算
diff_images = torch.abs(reconst_images - original_images)

!!!!!!!!!!!!

# 元の形状に戻す

## 再構築された画像を元のCT値に戻す
#denormalized_ct_image = denormalize_ct_image(decoded_image, window_width= d.WindowCenter, window_level= d.WindowWidth)


def plot_images(original, decoded, n=5):
    plt.figure(figsize=(15, 10))
    for i in range(n):
        # オリジナル画像
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(512, 512), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # 再構築された画像
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(512, 512), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # 差分画像
        difference = decoded[i] - original[i]
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(difference.reshape(512, 512), cmap="bwr", vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis("off")
    plt.show()


# トレーニングデータから最初の3つの画像を表示
plot_images(train_data, decoded_images, n=3)
