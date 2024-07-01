説明

	1.	Input CT Images:
	•	最初にCT画像を入力します。
	2.	Normalization:
	•	画像を正規化して、ピクセル値を0〜1の範囲にします。
	3.	Extract 3D Patches:
	•	正規化した画像から3Dパッチを抽出します。
	4.	Convert to Tensor:
	•	抽出した3DパッチをPyTorchテンソルに変換します。
	5.	3D Encoder:
	•	テンソルパッチを3Dエンコーダに入力し、潜在空間に変換します。
	6.	Latent Space:
	•	3Dエンコーダの出力として潜在空間の表現を取得します。
	7.	3D Decoder:
	•	潜在空間から3Dデコーダを通じて再構築された画像を生成します。
	8.	Reconstructed Images:
	•	再構築された画像を取得します。
	9.	Calculate Loss:
	•	入力画像と再構築画像の間の損失を計算します（平均二乗誤差）。
	10.	Training Process:
	•	損失を基にモデルを学習させます。
	11.	Save Models:
	•	学習済みのエンコーダとデコーダを保存します。
	12.	Transfer Learning:
	•	新しいCT画像に対して保存したモデルをロードし、同様の処理を実行して異常を検出します。
    
    
### 既存コードとの違い

1. **データの形式と前処理**：
   - 既存のコードは2DのDICOM画像を読み込み、512x512の正規化されたスライスに変換しています。
   - 新しいコードは3Dのパッチ（32x32x32）を抽出し、正規化とテンソル変換を行っています。

2. **モデルのアーキテクチャ**：
   - 既存のコードは2DのAutoencoderを使用しており、エンコーダとデコーダの各層は2D Convolutional層です。
   - 新しいコードは3D Convolutional Autoencoderを使用しており、各層が3D Convolutional層です。

3. **異常度の計算**：
   - 既存のコードは差分を計算して異常度を定量化するための仕組みを明示的に実装していません。
   - 新しいコードは平均二乗誤差（MSE）を用いて入力と出力の差分を計算し、異常度を定量化しています。

4. **学習の設定**：
   - 既存のコードはハイパーパラメータを手動で設定し、オートエンコーダを学習させています。
   - 新しいコードもハイパーパラメータを設定して学習させますが、データセットやバッチサイズ、学習率などは異なる場合があります。

### 既存コードの優れた点

1. **簡潔で直感的な実装**：
   - 既存のコードはシンプルで、2D画像を対象としているため、実装が比較的容易です。
   - オートエンコーダの基本的な学習と再構成プロセスが明確に示されています。

2. **再現性の高いデータ前処理**：
   - DICOM画像の正規化と再正規化の関数が定義されており、医療画像処理の際に役立つ具体的なステップが含まれています。
   - 特定のウィンドウ幅とウィンドウレベルを使用して画像を正規化する方法が示されています。

3. **損失と再構築画像の視覚化**：
   - 学習の進行状況を追跡するための損失曲線のプロット機能があり、モデルの学習性能を評価するのに役立ちます。
   - オリジナル画像と再構築画像、およびその差分を視覚化する機能があり、異常検知の精度を視覚的に確認することができます。

4. **トレーニングと検証**：
   - トレーニングデータに対して検証ロスを計算し、モデルのオーバーフィッティングを防ぐための手法が実装されています。

以上の点を踏まえ、既存のコードはシンプルでありながら、医療画像処理の基本的なステップとオートエンコーダの学習プロセスを直感的に示しています。一方、新しいコードは3Dのパッチを使用し、より複雑な異常検知を目指していますが、既存のコードの視覚化機能や前処理の一部を取り入れることで、より効果的な実装が可能になります。




import os
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib

# OpenMPエラー回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# データパスの設定
example_person_dicom_dir_path = r"C:\Users\kota\Desktop\suyama\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"

# 正規化関数
def normalize_ct_image(image):
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

# 再構築されたデータを元のCT値に戻す関数
def denormalize_ct_image(image):
    scaler = MinMaxScaler()
    return scaler.inverse_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

# データの準備
def load_and_preprocess_dicom_files(dicom_dir):
    CTfiles = []
    dicomfilepaths = sorted(os.listdir(dicom_dir))
    for filepath in dicomfilepaths:
        full_path = os.path.join(dicom_dir, filepath)
        d = pydicom.dcmread(full_path)
        hu = d.pixel_array
        nhu = normalize_ct_image(hu)
        CTfiles.append(nhu)
    CTfiles = np.array(CTfiles)
    return CTfiles

# 正規化したデータを取得
mizumashi_of_normalizedCTfiles = load_and_preprocess_dicom_files(example_person_dicom_dir_path)

# 学習データの準備（3Dパッチの抽出）
def extract_3d_patches(scan, patch_size, num_patches):
    patches = []
    x_max, y_max, z_max = scan.shape
    for _ in range(num_patches):
        x = np.random.randint(0, x_max - patch_size[0])
        y = np.random.randint(0, y_max - patch_size[1])
        z = np.random.randint(0, z_max - patch_size[2])
        patch = scan[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]
        patches.append(patch)
    return np.array(patches)

# 3Dパッチの抽出
patch_size = (32, 32, 32)
num_patches_per_scan = 200
all_patches = []
for scan in mizumashi_of_normalizedCTfiles:
    patches = extract_3d_patches(scan, patch_size, num_patches_per_scan)
    all_patches.extend(patches)

# 正規化したパッチをテンソルに変換
all_patches = np.array(all_patches)
tensor_patches = torch.tensor(all_patches, dtype=torch.float32).unsqueeze(1)

# ハイパーパラメータの設定
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 100
EPOCHS = 20

# 3Dエンコーダの定義
class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 8 * 8 * 8, Z_DIM)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3Dデコーダの定義
class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        self.fc = nn.Linear(Z_DIM, 128 * 8 * 8 * 8)
        self.conv1 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose3d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

# データローダの作成
dataset = TensorDataset(tensor_patches)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# モデルの定義
encoder = Encoder3D()
decoder = Decoder3D()

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
z_points = encoder(tensor_patches)
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
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.show()

# 画像のプロット関数
def plot_images(original, decoded, n=5, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(15, 10))
    for i in range(n):
        # オリジナル画像
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i, 0, :, :, 16], cmap="gray")  # 中心スライスを表示
        plt.title("Original")
        plt.axis("off")

        # 再構築された画像
        denormalized_image = denormalize_ct_image(decoded[i, 0, :, :, 16].detach().numpy())
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(denormalized_image, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # 差分画像
        difference = denormalized_image - original[i, 0, :, :, 16]
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(difference, cmap="bwr", vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis("off")

    # 画像を保存
    plt.savefig(os.path.join(save_path, "reconstructed_images.png"))
    plt.show()

# 損失曲線をプロットおよび保存
plot_loss_curve(train_losses, val_losses)

# トレーニングデータから最初の5つの画像を表示および保存
plot_images(tensor_patches, reconst_images.detach(), n=5)

print("Training and visualization complete.")



# モデルの保存
model_save_path = "saved_models"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

torch.save(encoder.state_dict(), os.path.join(model_save_path, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(model_save_path, "decoder.pth"))

print("Models saved successfully.")

# 転移学習のためのモデルの読み込み
def load_models(encoder_path, decoder_path):
    encoder = Encoder3D()
    decoder = Decoder3D()
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder

# モデルの読み込み
encoder, decoder = load_models(os.path.join(model_save_path, "encoder.pth"), os.path.join(model_save_path, "decoder.pth"))

print("Models loaded successfully for transfer learning.")







ここまで

# 感度、特異度、AUCの計算と結果表示

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# 異常度を計算する関数
def calculate_anomaly_score(patch, model, criterion):
    with torch.no_grad():
        output = model(patch.unsqueeze(0))
        loss = criterion(output, patch.unsqueeze(0))
    return loss.item()

# 症例ごとの異常度を計算する関数
def get_case_anomaly_score(scan, model, patch_size):
    patches = extract_3d_patches(scan, patch_size, num_patches_per_scan)
    tensor_patches = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    anomaly_scores = [calculate_anomaly_score(patch, model, criterion) for patch in tensor_patches]
    return max(anomaly_scores)

# 異常スキャンと正常スキャンの異常度計算
abnormal_ct_files = ["path_to_abnormal_ct1", "path_to_abnormal_ct2", ...] # 異常スキャンのパスをリストで指定
normal_ct_files = ["path_to_normal_ct1", "path_to_normal_ct2", ...] # 正常スキャンのパスをリストで指定

abnormal_scores = []
normal_scores = []

for ct_file in abnormal_ct_files:
    scan = load_ct_scan(ct_file) # CTスキャンをロードする関数
    score = get_case_anomaly_score(scan, model, patch_size)
    abnormal_scores.append(score)

for ct_file in normal_ct_files:
    scan = load_ct_scan(ct_file) # CTスキャンをロードする関数
    score = get_case_anomaly_score(scan, model, patch_size)
    normal_scores.append(score)

# ラベルの作成（正常: 0, 異常: 1）
labels = [0] * len(normal_scores) + [1] * len(abnormal_scores)
scores = normal_scores + abnormal_scores

# 感度、特異度、AUCの計算
auc = roc_auc_score(labels, scores)
print(f'AUC: {auc:.4f}')

# 閾値を決めて感度と特異度を計算
threshold = 0.5  # 適切な閾値を設定
predictions = [1 if score > threshold else 0 for score in scores]
tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')