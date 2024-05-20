Resize to a resolution of 1mm^3 using bicubic interopration
Automatically segment lungs using thresholding and volumes cropped to localise the top of the heart
#DICOMの読み取りに必要なpydicomとPNGとして保存する上で必要なopencv-pythonを用意する
!pip3 install pydicom
!pip3 install opencv-python

import numpy as np
import pydicom
import glob
import cv2
import os
import time
import matplotlib.pyplot as plt
%matplotlib inline

from pydicom.uid import UID

# 1人の1枚のdicom画像で確認していく
example_person_dicom_dir_path = \
    "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0010_20190710_2"

aaa = os.listdir(example_person_dicom_dir_path)
aaa = sorted(aaa)


# 1人の1枚のdicom画像で確認していく
example_slice_of_example_person_dicom_dir_path = \
    example_person_dicom_dir_path + "/" + aaa[10]


d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
hu = d.pixel_array

# resize
hu_resized = cv2.resize(hu,dsize=None,fx=(d.ReconstructionDiameter / 512),fy=(d.ReconstructionDiameter / 512),interpolation=cv2.INTER_CUBIC)

# d.ReconstructionDiameter / 512 、が●mm/pixelを表す（この●は人によって異なることに注意）
# fx, fyは上記の逆数が正しいかと思ったが、上記が正しかった。

#################### 1. リサイズ  #####################
resized_image = hu_resized

################## 2. セグメント化 ####################

lung_threshold = -600
# 脂肪組織と軟部組織の閾値設定
fat_threshold_low = -100
fat_threshold_high = -50
soft_tissue_threshold_low = 30
soft_tissue_threshold_high = 70

# 脂肪組織と軟部組織を分けてセグメント化する
def segment_lungs_and_mediastinum(image):

    lungs = (image < lung_threshold).astype(np.uint8)
    fat = ((image >= fat_threshold_low) & (image <= fat_threshold_high)).astype(np.uint8)
    soft_tissue = ((image >= soft_tissue_threshold_low) & (image <= soft_tissue_threshold_high)).astype(np.uint8)

    return lungs, fat, soft_tissue

# 画像をカラーで表示する（脂肪組織は青、軟部組織は赤）
def display_segmented_image(image, lungs, fat, soft_tissue):
    # グレースケール画像をRGBに変換
    rgb_image = np.stack((image,)*3, axis=-1)

    # 肺野を黄色、脂肪組織を青、軟部組織を赤にする
    rgb_image[lungs == 1] = [255, 255, 0]  # Yellow
    rgb_image[fat == 1] = [0, 0, 255]  # Blue
    rgb_image[soft_tissue == 1] = [255, 0, 0]  # Red

    plt.imshow(rgb_image, cmap='gray')
    plt.axis('off')
    plt.show()

# メイン処理
#file_path = 'path_to_your_dicom_file.dcm'  # DICOMファイルのパスを指定してください
#image = load_ct_image(file_path)
lungs, fat, soft_tissue = segment_lungs_and_mediastinum(image=resized_image)
display_segmented_image(resized_image, lungs, fat, soft_tissue)







# 横方向に基づくサイズ計算
from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes, center_of_mass
from skimage.morphology import disk
def segment_lungs_horizontal(image,lung_threshold):
    lung_threshold = lung_threshold
    lungs = (image < lung_threshold).astype(np.uint8)

    # 小さなノイズを除去（オープニング）
    lungs = binary_opening(lungs, structure=disk(3))

    # 肺結節や空洞を埋める（ホールフィリング）
    lungs = binary_fill_holes(lungs)

    # 再度、小さなノイズを除去（クロージング）
    lungs = binary_closing(lungs, structure=disk(3))

    # 連結成分をラベル付け
    labeled_lungs, num_features = label(lungs)

    # 各連結成分の横方向のサイズを計算
    horizontal_sizes = np.array([np.sum(labeled_lungs == label_num, axis=0) for label_num in range(1, num_features + 1)])

    # サイズが最大の2つの連結成分を選択
    if len(horizontal_sizes) > 2:
        largest_labels = np.argsort(horizontal_sizes.sum(axis=1))[-4:-2]  # 0は背景のため除外
        largest_component = np.isin(labeled_lungs, largest_labels + 1).astype(np.uint8)
    else:
        largest_component = np.zeros_like(labeled_lungs)

    return largest_component

# 画像をカラーで表示する
def display_segmented_image(image, lungs,lung_threshold):
    # グレースケール画像をRGBに変換
    rgb_image = np.stack((image,)*3, axis=-1)

    # 肺野を黄色にする
    rgb_image[lungs == 1] = [255, 255, 0]  # Yellow

    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title('Lung Segmented parameter='+str(lung_threshold))
    plt.show()

# メイン処理
image = hu_resized

for parameter in [-950 +num * 50 for num in range(0, 10)]:
  lung_threshold = parameter
  lungs = segment_lungs_horizontal(image,lung_threshold)
  display_segmented_image(image, lungs,lung_threshold)
#




from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes, center_of_mass
from skimage.morphology import disk

# 右肺を抽出する関数（コード1に基づく）
def extract_right_lung(image):
    lung_threshold = -600
    lungs = (image < lung_threshold)#.astype(np.uint16)

    # 小さなノイズを除去（オープニング）
    lungs = binary_opening(lungs, structure=disk(3))

    # 肺結節や空洞を埋める（ホールフィリング）
    lungs = binary_fill_holes(lungs)

    # 再度、小さなノイズを除去（クロージング）
    lungs = binary_closing(lungs, structure=disk(3))

    # 連結成分をラベル付け
    labeled_lungs, num_features = label(lungs)

    # 各連結成分の横方向のサイズを計算
    horizontal_sizes = np.array([np.sum(labeled_lungs == label_num, axis=0) for label_num in range(1, num_features + 1)])

    # サイズが最大の連結成分を選択
    if len(horizontal_sizes) > 2:
        largest_labels = np.argsort(horizontal_sizes.sum(axis=1))[-4:-3]  # 0は背景のため除外
        largest_component = np.isin(labeled_lungs, largest_labels + 1)#.astype(np.uint16)
    else:
        largest_component = np.zeros_like(labeled_lungs)

    return largest_component

# 左肺を抽出する関数（コード2に基づく）
def extract_left_lung(image):
    lung_threshold = -600
    lungs = (image < lung_threshold).astype(np.uint8)

    # 小さなノイズを除去（オープニング）
    lungs = binary_opening(lungs, structure=disk(3))

    # 肺結節や空洞を埋める（ホールフィリング）
    lungs = binary_fill_holes(lungs)

    # 再度、小さなノイズを除去（クロージング）
    lungs = binary_closing(lungs, structure=disk(3))

    # 連結成分をラベル付け
    labeled_lungs, num_features = label(lungs)

    # 各連結成分の横方向のサイズを計算
    horizontal_sizes = np.array([np.sum(labeled_lungs == label_num, axis=0) for label_num in range(1, num_features + 1)])

    # サイズが最大の連結成分を選択
    if len(horizontal_sizes) > 2:
        largest_labels = np.argsort(horizontal_sizes.sum(axis=1))[-3:-2]  # 0は背景のため除外
        largest_component = np.isin(labeled_lungs, largest_labels + 1).astype(np.uint8)
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
        print(np.nonzero(left_lung[:, y])[0].size , np.nonzero(right_lung[:, y])[0].size)
        if np.nonzero(left_lung[y,:])[0].size * np.nonzero(right_lung[y,:])[0].size >0 :
          edge1 = np.min(np.nonzero(left_lung[y,:]))
          edge2 = np.max(np.nonzero(left_lung[y,:]))
          edge3 = np.min(np.nonzero(right_lung[y,:]))
          edge4 = np.max(np.nonzero(right_lung[y,:]))
          left_edge = np.sort([edge1,edge2,edge3,edge4])[1]
          right_edge = np.sort([edge1,edge2,edge3,edge4])[2]
          mediastinum[y,left_edge:right_edge] = 1
          print(y,edge1,edge2,edge3,edge4,left_edge,right_edge)

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

# メイン処理
image=resized_image
right_lung = extract_right_lung(image)
left_lung = extract_left_lung(image)
mediastinum = extract_mediastinum(left_lung, right_lung)
display_segmented_image(image, left_lung, right_lung, mediastinum)
