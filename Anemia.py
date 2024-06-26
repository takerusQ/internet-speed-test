import numpy as np
import pydicom
import glob
import cv2
import os
import time
import matplotlib.pyplot as plt

from pydicom.uid import UID
from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes, center_of_mass
from skimage.morphology import disk

import plotly.express as px
from ipywidgets import interact
from ipywidgets import interact, IntSlider

def load_dicom_series(dicom_dir=r"C:\Users\root\Desktop\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"):
    
    aaa = os.listdir(dicom_dir)
    dicomfilepaths = sorted(aaa)
    # 1人の1枚のdicom画像で確認していく
    CTfiles = []
    CTfiles_resized = []
    normalizedCTfiles = []
    for k in range(len(dicomfilepaths)):
        example_slice_of_example_person_dicom_dir_path = \
        dicom_dir + "/" + dicomfilepaths[k]
        
        d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
        hu = d.pixel_array
        hu_resized = cv2.resize(hu,dsize=None,fx=(d.ReconstructionDiameter / 512),fy=(d.ReconstructionDiameter / 512),interpolation=cv2.INTER_CUBIC)
        CTfiles.append(hu)
        CTfiles_resized.append(hu_resized)
        nhu = normalize_ct_image(hu, window_width= d.WindowWidth, window_level= d.WindowCenter)
        normalizedCTfiles.append(nhu)
    return CTfiles,CTfiles_resized,normalizedCTfiles

# 画像スタックを読み込み
dicom_dir = r"C:\Users\root\Desktop\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"  # ここを適切なパスに置き換えてください
CTfiles,CTfiles_resized,normalizedCTfiles = load_dicom_series(dicom_dir)
ct_array = np.array(normalizedCTfiles)

ct_array = np.array(CTfiles)  # Z, Y, X の順

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
    processed_image = processed_image.astype(np.uint8)  # 確実に8ビットの単一チャンネルに変換
    
    num_labels, labels_im = cv2.connectedComponents(processed_image)
    
    # ラベルの特性をチェック
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # 面積の基準 (適切な閾値に調整)
            if 1000 < area < 5000:
                return True
    return False

# 左心室の断面積を計算する関数
def calculate_lv_area(slice_image):
    processed_image = process_slice(slice_image)
    processed_image = processed_image.astype(np.uint8)  # 確実に8ビットの単一チャンネルに変換
    
    num_labels, labels_im = cv2.connectedComponents(processed_image)
    
    max_area = 0
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # 左心室の断面積を取得 (適切な閾値に調整)
            if 1000 < area < 5000:
                if area > max_area:
                    max_area = area
    return max_area

# 画像スタックをループして、心臓レベルのスライスを判定し、左心室の断面積が最大のスライスを選択
max_lv_area = 0
best_slice = None
best_slice_slicenum = None

for i in range(ct_array.shape[0]):
#for i in range(50):
    slice_image = ct_array[i, :, :]
    if is_heart_level(slice_image):
        lv_area = calculate_lv_area(slice_image)
        if lv_area > max_lv_area:
            max_lv_area = lv_area
            best_slice = slice_image
            best_slice_slicenum = i

# 最適なスライスを表示
if best_slice is not None:
    cv2.imshow('Best Slice (slicenum='+str(best_slice_slicenum), best_slice)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("左心室を特定できるスライスが見つかりませんでした。")




plt.title("original")
plt.axis('off')
plt.imshow(best_slice, cmap=plt.cm.Greys_r)##for i in range(50):でやると３ととなった。全体でやると１１０になってしまう






12 ROIのCT値測定（手動）
Jupyter Lab上で画像スタックを表示し、興味のあるスライスに任意の大きさの丸いROIを画像上のカーソルクリックによる中心座標の指定と半径の入力で作成し、そのROIのCT値の平均と分散を計算する

説明
DICOMシリーズの読み込み: SimpleITKを使用してDICOMシリーズを読み込みます。
画像スタックの表示: matplotlibを使用して画像スタックの各スライスを表示します。スライダーでスライスを選択できます。
クリックイベントの処理: 画像上の任意の位置をクリックしてROIの中心座標を指定します。
ROIの半径設定: スライダーでROIの半径を設定します。
CT値の計算: 指定されたROI内のCT値の平均と分散を計算して表示します。






# DICOMシリーズの読み込み関数
def load_dicom_series(dicom_dir=r"C:\Users\root\Desktop\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"):
    
    aaa = os.listdir(dicom_dir)
    dicomfilepaths = sorted(aaa)
    # 1人の1枚のdicom画像で確認していく
    CTfiles = []
    CTfiles_resized = []
    normalizedCTfiles = []
    for k in range(len(dicomfilepaths)):
        example_slice_of_example_person_dicom_dir_path = \
        dicom_dir + "/" + dicomfilepaths[k]
        
        d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
        hu = d.pixel_array
        hu_resized = cv2.resize(hu,dsize=None,fx=(d.ReconstructionDiameter / 512),fy=(d.ReconstructionDiameter / 512),interpolation=cv2.INTER_CUBIC)
        CTfiles.append(hu)
        CTfiles_resized.append(hu_resized)
        nhu = normalize_ct_image(hu, window_width= d.WindowWidth, window_level= d.WindowCenter)
        normalizedCTfiles.append(nhu)
    return CTfiles,CTfiles_resized,normalizedCTfiles

# 画像スタックを読み込み
dicom_dir = r"C:\Users\root\Desktop\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"  # ここを適切なパスに置き換えてください
CTfiles,CTfiles_resized,normalizedCTfiles = load_dicom_series(dicom_dir)
ct_array = np.array(normalizedCTfiles)  # Z, Y, X の順

# グローバル変数
global selected_slice, roi_center, roi_radius
selected_slice = None
roi_center = None
roi_radius = None

# 画像スタックの表示
def show_slice(slice_index):
    global selected_slice
    selected_slice = slice_index
    plt.imshow(ct_array[slice_index, :, :], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.show()

# スライス選択のためのスライダー
slice_slider = IntSlider(min=0, max=ct_array.shape[0]-1, step=1, description='Slice:')
interact(show_slice, slice_index=slice_slider)

# 画像上のクリックイベントの処理
def onclick(event):
    global roi_center
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        roi_center = (x, y)
        print(f'ROI center selected at: {roi_center}')
        calculate_roi_stats()

# ROIのCT値の平均と分散を計算
def calculate_roi_stats():
    global roi_center, roi_radius, selected_slice
    if roi_center is None or roi_radius is None or selected_slice is None:
        return

    x, y = roi_center
    radius = roi_radius

    mask = np.zeros_like(ct_array[selected_slice, :, :], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 1, -1)
    
    roi_values = ct_array[selected_slice, mask > 0]
    mean_val = np.mean(roi_values)
    std_val = np.std(roi_values)

    print(f'Mean CT value: {mean_val}')
    print(f'Standard Deviation: {std_val}')

# 半径の入力ウィジェット
radius_slider = IntSlider(min=1, max=100, step=1, description='Radius:')
def set_radius(radius):
    global roi_radius
    roi_radius = radius
    print(f'ROI radius set to: {roi_radius}')
    calculate_roi_stats()
    
interact(set_radius, radius=radius_slider)

# 画像上でのクリックを有効に
fig, ax = plt.subplots()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
show_slice(40)





ROIをマウスで指定して自動計算¶

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



dicom_dir = r"C:\Users\root\Desktop\AorticTrauma(Thoracic)\TAOI001a_ZAe6c2d7cd_999999_1.2.392.200080.100.200.1966092444.127.14.80.150.18.57\4_999999_1.2.392.200080.100.200.2354659766.48.112.212.78.158.225"  # ここを適切なパスに置き換えてください
CTfiles,CTfiles_resized,normalizedCTfiles = load_dicom_series(dicom_dir)
ct_array = np.array(CTfiles)  # Z, Y, X の順


selected_slice=36


vmin, vmax=0,200
plt.imshow(CTfiles[selected_slice], cmap=plt.cm.Greys_r,vmin=vmin, vmax=vmax)##for i in range(50):でやると３ととなった。全体でやると１１０になってしまう
plt.colorbar(ticks=np.linspace(vmin, vmax, 5))
plt.axis('off')





#img_raw=CTfiles[selected_slice]
img_raw= ct_array[selected_slice, :, :]

aaa = os.listdir(dicom_dir)
dicomfilepaths = sorted(aaa)
example_slice_of_example_person_dicom_dir_path = \
        dicom_dir + "/" + dicomfilepaths[selected_slice]
d = pydicom.dcmread(example_slice_of_example_person_dicom_dir_path)
hu = d.pixel_array
nhu = normalize_ct_image(hu, window_width= d.WindowWidth, window_level= d.WindowCenter)


ROI = cv2.selectROI('Select ROIs', nhu, fromCenter = False, showCrosshair = False)


x1 = ROI[0]
y1 = ROI[1]
x2 = ROI[2]
y2 = ROI[3]

print('ROI', ROI)

#Crop Image
#img_crop = img_raw[int(y1):int(y1+y2),int(x1):int(x1+x2)]

#cv2.imshow("crop", img_crop)
if ROI is not None:
    x, y, w, h = ROI
    roi_slice = img_raw[y:y+h, x:x+w]
    
    # CT値の平均と分散を計算
    mean_val = np.mean(roi_slice)
    std_val = np.std(roi_slice)
    
    print(f'Mean CT value: {mean_val}')
    print(f'Standard Deviation: {std_val}')
