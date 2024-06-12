
)

# ウィンドウ幅とウィンドウレベルの設定（例として設定）
window_width = 1500
window_level = 400

# ウィンドウ幅とウィンドウレベルを用いて正規化する関数
def normalize_ct_image(image, window_width, window_level):
    min_value = window_level - (window_width / 2)
    max_value = window_level + (window_width / 2)
    normalized_image = (image - min_value) / (max_value - min_value)
    normalized_image[normalized_image < 0] = 0
    normalized_image[normalized_image > 1] = 1
    return normalized_image

# 正規化
normalized_ct_image = normalize_ct_image(ct_image, window_width, window_level)

# ハイパーパラメータ
input_dim = 512 * 512  # 512x512ピクセル
encoding_dim = 64 * 64  # エンコードされた表現の次元数を小さくする（例として64x64）
learning_rate = 0.01
epochs = 100
batch_size = 2

# データの前処理
flat_train_data = normalized_ct_image.reshape(1, input_dim)

# 元の形状に戻す
decoded_image = decoded_data.reshape(512, 512)

# 再構築されたデータを元のCT値に戻す関数
def denormalize_ct_image(normalized_image, window_width, window_level):
    min_value = window_level - (window_width / 2)
    max_value = window_level + (window_width / 2)
    denormalized_image = normalized_image * (max_value - min_value) + min_value
    return denormalized_image

# 再構築された画像を元のCT値に戻す
denormalized_ct_image = denormalize_ct_image(decoded_image, window_width, window_level)

LossとValidation Lossの推移を確認するためのコードの例を示します。


import numpy as np
import psutil  # psutilライブラリのインポート
import gc  # ガベージコレクションのインポート
import matplotlib.pyplot as plt  # グラフ表示用

# 512x512の画像データを生成（ここではダミーデータ）
data = np.random.rand(20, 512, 512, 1)  # データセットを増やす

# データをトレーニングとバリデーションに分割
train_data = data[:15]  # 最初の15個をトレーニングデータ
val_data = data[15:]    # 残り5個をバリデーションデータ

# データの形状確認
if train_data.shape[1:] == (512, 512, 1):
    print("トレーニングデータは (512, 512, 1) の形状を持っています。")
if val_data.shape[1:] == (512, 512, 1):
    print("バリデーションデータは (512, 512, 1) の形状を持っています。")

# ハイパーパラメータ
input_dim = 512 * 512  # 512x512ピクセル
encoding_dim = 64 * 64  # エンコードされた表現の次元数を小さくする（例として64x64）
learning_rate = 0.01
epochs = 100  # エポック数を減らす
batch_size = 2  # バッチサイズを設定

# データの前処理
# 画像データをフラット化
flat_train_data = train_data.reshape((train_data.shape[0], input_dim))
flat_val_data = val_data.reshape((val_data.shape[0], input_dim))

# 重みの初期化
weights_input_to_hidden = np.random.randn(input_dim, encoding_dim) * 0.01
weights_hidden_to_output = np.random.randn(encoding_dim, input_dim) * 0.01

# 活性化関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# RAM使用量のモニタリング関数
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")  # メモリ使用量をMBで表示

# トレーニングとバリデーションの損失を記録するリスト
train_losses = []
val_losses = []

# トレーニングループ
for epoch in range(epochs):
    # トレーニングデータでの学習
    for i in range(0, flat_train_data.shape[0], batch_size):
        batch_data = flat_train_data[i:i + batch_size]

        # フォワードパス
        hidden_layer_activation = np.dot(batch_data, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
        reconstructed_output = sigmoid(output_layer_activation)

        # 損失の計算（平均二乗誤差）
        loss = np.mean((batch_data - reconstructed_output) ** 2)

        # バックプロパゲーション
        error = batch_data - reconstructed_output
        d_output = error * sigmoid_derivative(reconstructed_output)

        error_hidden_layer = d_output.dot(weights_hidden_to_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # 重みの更新
        weights_hidden_to_output += hidden_layer_output.T.dot(d_output) * learning_rate
        weights_input_to_hidden += batch_data.T.dot(d_hidden_layer) * learning_rate

        # メモリのクリアリング
        del batch_data, hidden_layer_activation, hidden_layer_output, output_layer_activation, reconstructed_output, error, d_output, error_hidden_layer, d_hidden_layer
        gc.collect()

    # バリデーションデータでの損失計算
    hidden_layer_activation = np.dot(flat_val_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
    reconstructed_output = sigmoid(output_layer_activation)
    val_loss = np.mean((flat_val_data - reconstructed_output) ** 2)

    train_losses.append(loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}')
        print_memory_usage()

# 学習の損失とバリデーション損失をプロット
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# テストデータをエンコードおよびデコード
encoded_data = sigmoid(np.dot(flat_train_data, weights_input_to_hidden))
decoded_data = sigmoid(np.dot(encoded_data, weights_hidden_to_output))

# 元の形状に戻す
decoded_images = decoded_data.reshape((train_data.shape[0], 512, 512, 1))

print("Original Data Shape:", train_data[0].shape)
print("Reconstructed Data Shape:", decoded_images[0].shape)







	•	損失の記録と表示:
	•	train_lossesとval_lossesリストを用意し、各エポックの損失とバリデーション損失を記録します。
	•	トレーニングループの中で、定期的に損失を表示します。
	•	最後に、損失の推移をグラフで表示して学習の進行状況を視覚化します。

このコードにより、トレーニングとバリデーションの損失を比較しながら、
モデルの学習プロセスをモニタリングできます。
これで、過学習やアンダーフィッティングの兆候を確認することができます。

import numpy as np
import gc  # ガベージコレクションのインポート
import matplotlib.pyplot as plt  # グラフ表示用

# 512x512の画像データを生成（ここではダミーデータ）
data = np.random.rand(20, 512, 512, 1)  # データセットを増やす

# データをトレーニングとバリデーションに分割
train_data = data[:15]  # 最初の15個をトレーニングデータ
val_data = data[15:]    # 残り5個をバリデーションデータ

# データの形状確認
if train_data.shape[1:] == (512, 512, 1):
    print("トレーニングデータは (512, 512, 1) の形状を持っています。")
if val_data.shape[1:] == (512, 512, 1):
    print("バリデーションデータは (512, 512, 1) の形状を持っています。")

# ハイパーパラメータ
input_dim = 512 * 512  # 512x512ピクセル
encoding_dim = 64 * 64  # エンコードされた表現の次元数を小さくする（例として64x64）
learning_rate = 0.01
epochs = 100  # エポック数を減らす
batch_size = 2  # バッチサイズを設定

# データの前処理
# 画像データをフラット化
flat_train_data = train_data.reshape((train_data.shape[0], input_dim))
flat_val_data = val_data.reshape((val_data.shape[0], input_dim))

# 重みの初期化
weights_input_to_hidden = np.random.randn(input_dim, encoding_dim) * 0.01
weights_hidden_to_output = np.random.randn(encoding_dim, input_dim) * 0.01

# 活性化関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# トレーニングとバリデーションの損失を記録するリスト
train_losses = []
val_losses = []

# トレーニングループ
for epoch in range(epochs):
    # トレーニングデータでの学習
    for i in range(0, flat_train_data.shape[0], batch_size):
        batch_data = flat_train_data[i:i + batch_size]

        # フォワードパス
        hidden_layer_activation = np.dot(batch_data, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
        reconstructed_output = sigmoid(output_layer_activation)

        # 損失の計算（平均二乗誤差）
        loss = np.mean((batch_data - reconstructed_output) ** 2)

        # バックプロパゲーション
        error = batch_data - reconstructed_output
        d_output = error * sigmoid_derivative(reconstructed_output)

        error_hidden_layer = d_output.dot(weights_hidden_to_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # 重みの更新
        weights_hidden_to_output += hidden_layer_output.T.dot(d_output) * learning_rate
        weights_input_to_hidden += batch_data.T.dot(d_hidden_layer) * learning_rate

        # メモリのクリアリング
        del batch_data, hidden_layer_activation, hidden_layer_output, output_layer_activation, reconstructed_output, error, d_output, error_hidden_layer, d_hidden_layer
        gc.collect()

    # バリデーションデータでの損失計算
    hidden_layer_activation = np.dot(flat_val_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
    reconstructed_output = sigmoid(output_layer_activation)
    val_loss = np.mean((flat_val_data - reconstructed_output) ** 2)

    train_losses.append(loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}')

# 学習の損失とバリデーション損失をプロット
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# テストデータをエンコードおよびデコード
encoded_data = sigmoid(np.dot(flat_train_data, weights_input_to_hidden))
decoded_data = sigmoid(np.dot(encoded_data, weights_hidden_to_output))

# 元の形状に戻す
decoded_images = decoded_data.reshape((train_data.shape[0], 512, 512, 1))

print("Original Data Shape:", train_data[0].shape)
print("Reconstructed Data Shape:", decoded_images[0].shape)





画像の変化を可視化するために、以下のようなステップを実行できます：

	1.	オリジナル画像の表示: 元の画像を表示する。
	2.	エンコードされた画像の表示: エンコードされた表現（潜在空間の画像）を可視化する。
	3.	再構築された画像の表示: デコードされた再構築画像を表示する。


import numpy as np
import gc  # ガベージコレクションのインポート
import matplotlib.pyplot as plt  # グラフ表示用

# 512x512の画像データを生成（ここではダミーデータ）
data = np.random.rand(20, 512, 512, 1)  # データセットを増やす

# データをトレーニングとバリデーションに分割
train_data = data[:15]  # 最初の15個をトレーニングデータ
val_data = data[15:]    # 残り5個をバリデーションデータ

# データの形状確認
if train_data.shape[1:] == (512, 512, 1):
    print("トレーニングデータは (512, 512, 1) の形状を持っています。")
if val_data.shape[1:] == (512, 512, 1):
    print("バリデーションデータは (512, 512, 1) の形状を持っています。")

# ハイパーパラメータ
input_dim = 512 * 512  # 512x512ピクセル
encoding_dim = 64 * 64  # エンコードされた表現の次元数を小さくする（例として64x64）
learning_rate = 0.01
epochs = 100  # エポック数を減らす
batch_size = 2  # バッチサイズを設定

# データの前処理
# 画像データをフラット化
flat_train_data = train_data.reshape((train_data.shape[0], input_dim))
flat_val_data = val_data.reshape((val_data.shape[0], input_dim))

# 重みの初期化
weights_input_to_hidden = np.random.randn(input_dim, encoding_dim) * 0.01
weights_hidden_to_output = np.random.randn(encoding_dim, input_dim) * 0.01

# 活性化関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# トレーニングとバリデーションの損失を記録するリスト
train_losses = []
val_losses = []

# トレーニングループ
for epoch in range(epochs):
    # トレーニングデータでの学習
    for i in range(0, flat_train_data.shape[0], batch_size):
        batch_data = flat_train_data[i:i + batch_size]

        # フォワードパス
        hidden_layer_activation = np.dot(batch_data, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
        reconstructed_output = sigmoid(output_layer_activation)

        # 損失の計算（平均二乗誤差）
        loss = np.mean((batch_data - reconstructed_output) ** 2)

        # バックプロパゲーション
        error = batch_data - reconstructed_output
        d_output = error * sigmoid_derivative(reconstructed_output)

        error_hidden_layer = d_output.dot(weights_hidden_to_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # 重みの更新
        weights_hidden_to_output += hidden_layer_output.T.dot(d_output) * learning_rate
        weights_input_to_hidden += batch_data.T.dot(d_hidden_layer) * learning_rate

        # メモリのクリアリング
        del batch_data, hidden_layer_activation, hidden_layer_output, output_layer_activation, reconstructed_output, error, d_output, error_hidden_layer, d_hidden_layer
        gc.collect()

    # バリデーションデータでの損失計算
    hidden_layer_activation = np.dot(flat_val_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
    reconstructed_output = sigmoid(output_layer_activation)
    val_loss = np.mean((flat_val_data - reconstructed_output) ** 2)

    train_losses.append(loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}')

# 学習の損失とバリデーション損失をプロット
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# テストデータをエンコードおよびデコード
encoded_data = sigmoid(np.dot(flat_train_data, weights_input_to_hidden))
decoded_data = sigmoid(np.dot(encoded_data, weights_hidden_to_output))

# 元の形状に戻す
decoded_images = decoded_data.reshape((train_data.shape[0], 512, 512, 1))

# 画像の変化を可視化する
def plot_images(original, decoded, n=5):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        # オリジナル画像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(512, 512), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # 再構築された画像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(512, 512), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()

# トレーニングデータから最初の5つの画像を表示
plot_images(train_data, decoded_images, n=5)