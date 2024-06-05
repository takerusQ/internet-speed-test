TensorFlowやKeras、PyTorchなどのディープラーニングフレームワークを使わずにオートエンコーダを実装することは可能ですが、手動でニューラルネットワークを定義し、訓練する必要があります。この方法は複雑で、特に3D画像の処理には多くの手作業が必要です。

以下に、Numpyを使って簡単なオートエンコーダを実装する例を示します。ここでは3次元データではなく、2次元データを使って概念を説明します。

説明

	1.	データ生成: ここでは64次元のデータを1000個生成しています。
	2.	ハイパーパラメータ: 入力次元、エンコーディング次元、学習率、およびエポック数を設定します。
	3.	重みの初期化: 入力層から隠れ層、隠れ層から出力層への重みをランダムに初期化します。
	4.	フォワードパス: データを隠れ層に通し、活性化関数を適用します。隠れ層の出力を出力層に通し、再度活性化関数を適用して再構築された出力を得ます。
	5.	損失の計算: 再構築された出力と元のデータとの間の損失を計算します。
	6.	バックプロパゲーション: 誤差を計算し、重みを更新します。
	7.	トレーニングループ: これを所定のエポック数繰り返します。

この例は非常に基本的なもので、3次元データを扱うためには多くの調整が必要ですが、基本的なコンセプトは同じです。


import numpy as np

# 活性化関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# データの生成（ここでは簡単な2次元データを使用）
data = np.random.rand(1000, 64)  # 1000個の64次元データ

# ハイパーパラメータ
input_dim = 64
encoding_dim = 32
learning_rate = 0.1
epochs = 1000

# 重みの初期化
weights_input_to_hidden = np.random.randn(input_dim, encoding_dim)
weights_hidden_to_output = np.random.randn(encoding_dim, input_dim)

# トレーニングループ
for epoch in range(epochs):
    # フォワードパス
    hidden_layer_activation = np.dot(data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
    reconstructed_output = sigmoid(output_layer_activation)
    
    # 損失の計算（平均二乗誤差）
    loss = np.mean((data - reconstructed_output) ** 2)
    
    # バックプロパゲーション
    error = data - reconstructed_output
    d_output = error * sigmoid_derivative(reconstructed_output)
    
    error_hidden_layer = d_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # 重みの更新
    weights_hidden_to_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_to_hidden += data.T.dot(d_hidden_layer) * learning_rate
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# テストデータをエンコードおよびデコード
encoded_data = sigmoid(np.dot(data, weights_input_to_hidden))
decoded_data = sigmoid(np.dot(encoded_data, weights_hidden_to_output))

print("Original Data:", data[0])
print("Reconstructed Data:", decoded_data[0])









import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(12, 6))

# 各層の位置とラベル
layers = [
    {"pos": (0, 0), "width": 2, "height": 5, "label": "Input\n64 units"},
    {"pos": (2, 0.5), "width": 2, "height": 4, "label": "Hidden Layer\n32 units"},
    {"pos": (4, 0.5), "width": 2, "height": 4, "label": "Output Layer\n64 units"},
]

# 各層を描画
for layer in layers:
    rect = Rectangle(layer["pos"], layer["width"], layer["height"], edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(layer["pos"][0] + layer["width"] / 2, layer["pos"][1] + layer["height"] / 2, layer["label"], 
            horizontalalignment='center', verticalalignment='center', fontsize=10)

# 矢印を描画
for i in range(len(layers) - 1):
    ax.arrow(layers[i]["pos"][0] + layers[i]["width"], layers[i]["pos"][1] + layers[i]["height"] / 2,
             layers[i + 1]["pos"][0] - layers[i]["pos"][0] - layers[i]["width"], 0,
             head_width=0.3, head_length=0.3, fc='k', ec='k')

# 図の調整
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.axis('off')

# 図を表示
plt.show()





import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(12, 6))

# 各層の位置とラベル
layers = [
    {"pos": (0, 0), "width": 2, "height": 5, "label": "Input\n64x64x64"},
    {"pos": (2, 0.5), "width": 2, "height": 4, "label": "Conv3D + MaxPooling\n32 Filters\n32x32x32"},
    {"pos": (4, 1), "width": 2, "height": 3, "label": "Conv3D + MaxPooling\n64 Filters\n16x16x16"},
    {"pos": (6, 1.5), "width": 2, "height": 2, "label": "Conv3D + MaxPooling\n128 Filters\n8x8x8"},
    {"pos": (8, 1.5), "width": 2, "height": 2, "label": "Encoded\n128 Filters\n8x8x8"},
    {"pos": (10, 1.5), "width": 2, "height": 2, "label": "Conv3D + UpSampling\n128 Filters\n8x8x8"},
    {"pos": (12, 1), "width": 2, "height": 3, "label": "Conv3D + UpSampling\n64 Filters\n16x16x16"},
    {"pos": (14, 0.5), "width": 2, "height": 4, "label": "Conv3D + UpSampling\n32 Filters\n32x32x32"},
    {"pos": (16, 0), "width": 2, "height": 5, "label": "Output\n64x64x64"},
]

# 各層を描画
for layer in layers:
    rect = Rectangle(layer["pos"], layer["width"], layer["height"], edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(layer["pos"][0] + layer["width"] / 2, layer["pos"][1] + layer["height"] / 2, layer["label"], 
            horizontalalignment='center', verticalalignment='center', fontsize=10)

# 矢印を描画
for i in range(len(layers) - 1):
    ax.arrow(layers[i]["pos"][0] + layers[i]["width"], layers[i]["pos"][1] + layers[i]["height"] / 2,
             layers[i + 1]["pos"][0] - layers[i]["pos"][0] - layers[i]["width"], 0,
             head_width=0.3, head_length=0.3, fc='k', ec='k')

# 図の調整
ax.set_xlim(-1, 19)
ax.set_ylim(-1, 7)
ax.axis('off')

# 図を表示
plt.show()




ここから本番

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# フローチャートの作成
fig, ax = plt.subplots(figsize=(14, 8))

# 各ブロックの位置とラベル
blocks = [
    {"pos": (0, 5), "width": 3, "height": 1, "label": "Start", "desc": ""},
    {"pos": (0, 4), "width": 3, "height": 1, "label": "Generate 512x512 Image Data", "desc": "512x512のダミー画像データを生成"},
    {"pos": (0, 3), "width": 3, "height": 1, "label": "Check Data Shape", "desc": "データの形状が (512, 512, 1) であるか確認"},
    {"pos": (0, 2), "width": 3, "height": 1, "label": "Flatten Data", "desc": "画像データをフラット化"},
    {"pos": (0, 1), "width": 3, "height": 1, "label": "Initialize Weights", "desc": "重みをランダムに初期化"},
    {"pos": (5, 4), "width": 3, "height": 1, "label": "Forward Pass", "desc": "データを隠れ層に通し、活性化関数を適用"},
    {"pos": (5, 3), "width": 3, "height": 1, "label": "Calculate Loss", "desc": "再構築された出力と元のデータとの損失を計算"},
    {"pos": (5, 2), "width": 3, "height": 1, "label": "Backpropagation", "desc": "誤差を計算し、重みを更新"},
    {"pos": (5, 1), "width": 3, "height": 1, "label": "Repeat for Epochs", "desc": "トレーニングループを所定のエポック数繰り返す"},
    {"pos": (0, 0), "width": 3, "height": 1, "label": "Encode & Decode Data", "desc": "フラット化したデータをエンコードおよびデコード"},
    {"pos": (5, 0), "width": 3, "height": 1, "label": "Reshape to Original", "desc": "デコードされたデータを元の形状に戻す"},
]

# 各ブロックを描画
for block in blocks:
    rect = Rectangle(block["pos"], block["width"], block["height"], edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(block["pos"][0] + block["width"] / 2, block["pos"][1] + block["height"] / 2, block["label"], 
            horizontalalignment='center', verticalalignment='center', fontsize=10)
    ax.text(block["pos"][0] + block["width"] / 2, block["pos"][1] - 0.3, block["desc"], 
            horizontalalignment='center', verticalalignment='center', fontsize=8)

# 矢印を描画
arrows = [
    ((1.5, 5), (1.5, 4)),
    ((1.5, 4), (1.5, 3)),
    ((1.5, 3), (1.5, 2)),
    ((1.5, 2), (1.5, 1)),
    ((1.5, 1), (1.5, 0.5)),
    ((1.5, 0.5), (4, 4.5)),
    ((6.5, 4), (6.5, 3)),
    ((6.5, 3), (6.5, 2)),
    ((6.5, 2), (6.5, 1)),
    ((6.5, 1), (6.5, 0.5)),
    ((4.5, 0.5), (4.5, 0))
]

for start, end in arrows:
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
             head_width=0.2, head_length=0.2, fc='k', ec='k')

# 図の調整
ax.set_xlim(-1, 9)
ax.set_ylim(-1, 6)
ax.axis('off')

# 図を表示
plt.show()



import numpy as np

# 512x512の画像データを生成（ここではダミーデータ）
data = np.random.rand(100, 512, 512, 1)  # 100個の512x512のグレースケール画像

# データの形状確認
if data.shape[1:] == (512, 512, 1):
    print("データは (512, 512, 1) の形状を持っています。")
else:
    print("データは指定された形状を持っていません。")

# 全ての画像の形状確認
all_correct_shape = all(img.shape == (512, 512, 1) for img in data)
if all_correct_shape:
    print("すべての画像が (512, 512, 1) の形状を持っています。")
else:
    print("一部の画像が指定された形状を持っていません。")

# ハイパーパラメータ
input_dim = 512 * 512  # 512x512ピクセル
encoding_dim = 128 * 128  # エンコードされた表現の次元数（例として128x128）
learning_rate = 0.1
epochs = 1000

# データの前処理
# 画像データをフラット化
flat_data = data.reshape((data.shape[0], input_dim))

# 重みの初期化
weights_input_to_hidden = np.random.randn(input_dim, encoding_dim)
weights_hidden_to_output = np.random.randn(encoding_dim, input_dim)

# 活性化関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# トレーニングループ
for epoch in range(epochs):
    # フォワードパス
    hidden_layer_activation = np.dot(flat_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_to_output)
    reconstructed_output = sigmoid(output_layer_activation)
    
    # 損失の計算（平均二乗誤差）
    loss = np.mean((flat_data - reconstructed_output) ** 2)
    
    # バックプロパゲーション
    error = flat_data - reconstructed_output
    d_output = error * sigmoid_derivative(reconstructed_output)
    
    error_hidden_layer = d_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # 重みの更新
    weights_hidden_to_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_to_hidden += flat_data.T.dot(d_hidden_layer) * learning_rate
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# テストデータをエンコードおよびデコード
encoded_data = sigmoid(np.dot(flat_data, weights_input_to_hidden))
decoded_data = sigmoid(np.dot(encoded_data, weights_hidden_to_output))

# 元の形状に戻す
decoded_images = decoded_data.reshape((data.shape[0], 512, 512, 1))

print("Original Data Shape:", data[0].shape)
print("Reconstructed Data Shape:", decoded_images[0].shape)