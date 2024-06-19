
https://qiita.com/shinmura0/items/6ec8b1b745176853c9c1

マハラノビスADは

×ピクセル単位の異常検知はできない
〇その代わり、メモリを圧迫しない。画像の解像度を上げてもメモリの記憶サイズは同じ。
×性能的にはやや劣るとされる
PaDimは

〇ピクセル単位の異常検知ができる
×上記のスライドのように1ピクセルずつ正常空間を作るので、位置ズレに弱い
×さらに、構造上メモリを圧迫する。画像の解像度を上げるとメモリの圧迫が顕著に。
〇性能的には良い

上記2つの手法は、異常度の算出にマハラノビス距離を採用していました。PatchCore[3]ではK近傍法を用いて異常度を算出します。

PatchCoreは

〇ピクセル単位の異常検知ができる
〇1ピクセルずつ正常空間を作るわけではなく、embeddingをごちゃ混ぜにするので、位置ズレに強い
〇性能的にはSOTAに近い
△メモリの圧迫を避けるために、CoreSet sampling（下図）を使い、保持するembeddingを効率的に絞る

結局、最強の手法はどれだ？

一旦、従来手法の話しに戻ります。最近の論文に、PatchCoreなど従来の手法・backboneを数個用意して、精度の変化を調べた論文があります[12]。驚くべきことに、従来手法を使ってもbackboneなどを調整すれば、MVTech ADにおいて完全制覇に近いスコア（AUROCが99.9%）が出ます。

つまり、EfficientADなど最新手法を使わなくても、良いスコアが出るということです。では、最新手法を学ぶ必要がないかというと、そうではないと感じています。

なぜなら、異常検知には最強の手法はなく、データに適した手法を選ぶ必要があると思っているからです。

Kaggleでは最強のbackboneは存在しません。データによってVitが強いときもあれば、CNNが強いときもあります。結局、有力なbackboneを泥臭く探すしかありません。

異常検知でも同じことが言え、文献[12]のように手法やbackboneを泥臭く探索する必要があります。性能的には劣るとされるマハラノビスADも、データセットの相性によっては一番性能が良いときもあります。（ちょっとしたテクニックが必要ですが）

実務では、精度に加え、推論時間の制約やメモリの制限もあります。これらを複合的に考えた上で手法を決める必要があり、その際に、この記事が皆さんの一助になると幸いです。




#DICOMの読み取りに必要なpydicomとPNGとして保存する上で必要なopencv-pythonを用意する
#pip show pydicom

!pip3 install pydicom
!pip3 install opencv-python

try:
    import pydicom
    print("pydicom is installed")
except ImportError:
    print("pydicom is not installed")

#################################################
import numpy as np
import pydicom
import glob
import cv2
import os
import time
import matplotlib.pyplot as plt
%matplotlib inline

from pydicom.uid import UID
from skimage.measure import label, regionprops
from skimage.draw import disk
from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes, center_of_mass
from skimage.morphology import disk
from sklearn.cluster import KMeans
import joblib
from matplotlib.colors import LinearSegmentedColormap


# 1人の1枚のdicom画像で確認していく
example_person_dicom_dir_path = \
    ""

aaa = os.listdir(example_person_dicom_dir_path)
aaa = sorted(aaa)


# 1人の1枚のdicom画像で確認していく
example_slice_of_example_person_dicom_dir_path = \
    example_person_dicom_dir_path + "/" + aaa[20]
    #example_person_dicom_dir_path + "/" + aaa[10]



d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
hu = d.pixel_array

# resize
hu_resized = cv2.resize(hu,dsize=None,fx=(d.ReconstructionDiameter / 512),fy=(d.ReconstructionDiameter / 512),interpolation=cv2.INTER_CUBIC)

# d.ReconstructionDiameter / 512 、が●mm/pixelを表す（この●は人によって異なることに注意）
# fx, fyは上記の逆数が正しいかと思ったが、上記が正しかった。
plt.imshow(hu_resized, cmap=plt.cm.Greys_r)
plt.colorbar()

# ★★★★★★★★円形度を判定する関数★★★★★★★★
def is_circular(segmented_image, threshold=0.75):
    labeled_image = label(segmented_image)
    regions = regionprops(labeled_image)
    
    for region in regions:
        if region.area >= 50:  # ノイズを除外するための面積の閾値（必要に応じて調整）
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            print("circularity:", circularity)
            if circularity >= threshold:
                return True
    return False

#segmented_image に求められるデータタイプは、一般的には 2D バイナリ画像（0 と 1 で構成される）です。
#segmented_image はセグメンテーションされた領域を示すためのバイナリマスクで、0 は背景、1 は対象領域を表します。


########################################################
# テスト用の丸っこい領域を含むバイナリ画像を作成
test_image = np.zeros((100, 100), dtype=np.uint8)
rr, cc = disk((50, 50), 20)
test_image[rr, cc] = 1  # 丸っこい領域

# 円形度を評価
is_round = is_circular(test_image)
print(f"Is the segmented region circular? {is_round}")

# テスト画像を表示
plt.imshow(test_image, cmap='gray')
plt.title(f"Is Circular: {is_round}")
plt.show()
#######################################################################

# 右肺を抽出する関数（コード1に基づく）
def extract_right_lung(image):
    # 大外部分が上半分と下半分に分割される場合とされない場合がありおかしなことになったので、すべての画像が分割されないように上下左右両端1行のみ-2048をいれる
    image[:, 0:5] = -2048
    image[:, -5:] = -2048
    image[0:5, :] = -2048
    image[-5:, :] = -2048

    lung_threshold = -600
    lungs = (image < lung_threshold)#.astype(np.uint16)
    """
    # 小さなノイズを除去（オープニング）
    lungs = binary_opening(lungs, structure=disk(3))

    # 肺結節や空洞を埋める（ホールフィリング）
    lungs = binary_fill_holes(lungs)

    # 再度、小さなノイズを除去（クロージング）
    lungs = binary_closing(lungs, structure=disk(3))
    """

    # 連結成分をラベル付け
    labeled_lungs, num_features = label(lungs)

    # 各連結成分の横方向のサイズを計算
    horizontal_sizes = np.array([np.sum(labeled_lungs == label_num, axis=0) for label_num in range(1, num_features + 1)])

    # サイズが最大の連結成分を選択
    if len(horizontal_sizes) > 2:
        largest_labels = np.argsort(horizontal_sizes.sum(axis=1))[-3:-2]  # 0は背景のため除外
        largest_component = np.isin(labeled_lungs, largest_labels + 1)#.astype(np.uint16)
    else:
        largest_component = np.zeros_like(labeled_lungs)

    return largest_component

# 左肺を抽出する関数（コード2に基づく）
def extract_left_lung(image):
    # 大外部分が上半分と下半分に分割される場合とされない場合がありおかしなことになったので、すべての画像が分割されないように上下左右両端1行のみ-2048をいれる
    image[:, 0:5] = -2048
    image[:, -5:] = -2048
    image[0:5, :] = -2048
    image[-5:, :] = -2048

    lung_threshold = -600
    lungs = (image < lung_threshold)#.astype(np.uint16)
    """
    # 小さなノイズを除去（オープニング）
    lungs = binary_opening(lungs, structure=disk(3))

    # 肺結節や空洞を埋める（ホールフィリング）
    lungs = binary_fill_holes(lungs)

    # 再度、小さなノイズを除去（クロージング）
    lungs = binary_closing(lungs, structure=disk(3))
    """

    # 連結成分をラベル付け
    labeled_lungs, num_features = label(lungs)

    # 各連結成分の横方向のサイズを計算
    horizontal_sizes = np.array([np.sum(labeled_lungs == label_num, axis=0) for label_num in range(1, num_features + 1)])

    # サイズが最大の連結成分を選択
    if len(horizontal_sizes) > 2:
        largest_labels = np.argsort(horizontal_sizes.sum(axis=1))[-2:-1]  # 0は背景のため除外
        largest_component = np.isin(labeled_lungs, largest_labels + 1)#.astype(np.uint16)
    else:
        largest_component = np.zeros_like(labeled_lungs)

    return largest_component

# 縦隔を抽出する関数

def extract_mediastinum(left_lung, right_lung):
    mediastinum = np.zeros_like(left_lung)
    if left_lung.any() and right_lung.any():
      upper_edge = np.min((np.max(np.where(left_lung.any(axis=1))[0]),np.max(np.where(right_lung.any(axis=1))[0])))
      lower_edge = np.max((np.min(np.where(right_lung.any(axis=1))[0]),np.min(np.where(left_lung.any(axis=1))[0])))

      for y in range(lower_edge,upper_edge+1):
        #print(np.nonzero(left_lung[:, y])[0].size , np.nonzero(right_lung[:, y])[0].size)
        if np.nonzero(left_lung[y,:])[0].size * np.nonzero(right_lung[y,:])[0].size >0 :
          edge1 = np.min(np.nonzero(left_lung[y,:]))
          edge2 = np.max(np.nonzero(left_lung[y,:]))
          edge3 = np.min(np.nonzero(right_lung[y,:]))
          edge4 = np.max(np.nonzero(right_lung[y,:]))
          left_edge = np.sort([edge1,edge2,edge3,edge4])[1]
          right_edge = np.sort([edge1,edge2,edge3,edge4])[2]
          mediastinum[y,left_edge:right_edge] = 1
          #print(y,edge1,edge2,edge3,edge4,left_edge,right_edge)#デバグ用

    return mediastinum

# 画像をカラーで表示する関数
def display_segmented_image(image, left_lung, right_lung, mediastinum):
    # グレースケール画像をRGBに変換
    rgb_image = np.stack((image,)*3, axis=-1)

    # 左肺を黄色に、右肺を緑色に、縦隔を赤色にする
    rgb_image[left_lung == 1] = [255, 255, 0]  # Yellow
    rgb_image[right_lung == 1] = [0, 255, 0]  # Green
    rgb_image[mediastinum == 1] = [255, 0, 0]  # Red

    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()


