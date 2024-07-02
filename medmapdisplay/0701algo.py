graph TD
    A[Input CT Images] --> B[Normalization]
    B --> C[Extract 3D Patches]
    C --> D[Convert to Tensor]
    D --> E[3D Encoder]
    E --> F[Latent Space]
    F --> G[3D Decoder]
    G --> H[Reconstructed Images]
    H --> I[Calculate Loss]
    I --> J[Training Process]

    J --> K[Save Models]

    subgraph Training Loop
        E --> I
        F --> G
        I --> E
    end

    K --> L[New CT Images]
    L --> M[Load Models]
    M --> N[Normalization]
    N --> O[Extract 3D Patches]
    O --> P[Convert to Tensor]
    P --> Q[3D Encoder (Transfer Learning)]
    Q --> R[Latent Space (New Data)]
    R --> S[3D Decoder (Transfer Learning)]
    S --> T[Reconstructed Images (New Data)]
    T --> U[Calculate Anomalies]

    subgraph Transfer Learning
        M --> Q
        Q --> R
        R --> S
        S --> T
    end
    
    
    
    
    

以下に、保存したモデルを使用して別の種類の画像に転移学習するサンプルコードを示します。

### 転移学習のサンプルコード

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# モデルの読み込み
def load_models(encoder_path, decoder_path):
    encoder = Encoder3D()
    decoder = Decoder3D()
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder

# 新しいデータセットの準備
def prepare_new_dataset(new_data_dir, patch_size, num_patches):
    new_ct_files = sorted(os.listdir(new_data_dir))
    new_data = []
    for ct_file in new_ct_files:
        scan = load_ct_scan(os.path.join(new_data_dir, ct_file))
        patches = extract_3d_patches(scan, patch_size, num_patches)
        new_data.extend(patches)
    new_data = np.array(new_data)
    tensor_new_data = torch.tensor(new_data, dtype=torch.float32).unsqueeze(1)
    return tensor_new_data

# パスの設定
model_save_path = "saved_models"
new_data_dir = "path_to_new_data"

# 学習済みモデルの読み込み
encoder, decoder = load_models(os.path.join(model_save_path, "encoder.pth"), os.path.join(model_save_path, "decoder.pth"))

# 新しいデータセットの準備
patch_size = (32, 32, 32)
num_patches_per_scan = 200
tensor_new_data = prepare_new_dataset(new_data_dir, patch_size, num_patches_per_scan)

# 新しいデータセットのデータローダの作成
dataset_new = TensorDataset(tensor_new_data)
dataloader_new = DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=True)

# 転移学習のための再定義
encoder.train()
decoder.train()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# 転移学習のトレーニングループ
for epoch in range(EPOCHS):
    total_train_loss = 0
    for batch in dataloader_new:
        x = batch[0]
        optimizer.zero_grad()
        z = encoder(x)
        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(dataloader_new)
    print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}')

# トレーニング後のモデルの評価
z_points_new = encoder(tensor_new_data)
reconst_images_new = decoder(z_points_new)

# トレーニングデータから最初の5つの画像を表示および保存
plot_images(tensor_new_data, reconst_images_new.detach(), n=5)

print("Transfer learning and visualization complete.")
```

### 説明

1. **モデルの読み込み**:
   - 保存済みのエンコーダとデコーダのパラメータをロードします。

2. **新しいデータセットの準備**:
   - 新しいCT画像データディレクトリから画像を読み込み、3Dパッチを抽出します。
   - 抽出した3Dパッチをテンソルに変換します。

3. **データローダの作成**:
   - 新しいデータセットからデータローダを作成します。

4. **転移学習**:
   - 保存済みのモデルを再トレーニングします。
   - トレーニングデータを用いてモデルのパラメータを更新します。

5. **モデルの評価と可視化**:
   - 新しいデータセットに対するモデルの出力を評価し、元の画像、再構築画像、差分画像を可視化します。

このコードを使用することで、既存のモデルを再利用し、新しい種類のCT画像に適応させることができます。