import pandas as pd
import numpy as np
from scipy import stats

def calculate_mean_confidence_interval(data, confidence=0.95):
    if data.empty:
        return None, (None, None)

    sample_mean = data.mean()

    # 標本の標準偏差を計算
    sample_std = data.std()

    # 信頼区間を計算
    alpha = 1.0 - confidence
    t_critical = stats.t.ppf(1.0 - alpha / 2.0, len(data) - 1)
    margin_of_error = t_critical * sample_std / np.sqrt(len(data))

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return sample_mean, (lower_bound, upper_bound)

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return iqr

# 例として、以下のようなSeriesを想定します
age_data = dfb["Age"]

mean_age, confidence_interval = calculate_mean_confidence_interval(age_data)
print("平均年齢:", mean_age)
print("年齢の信頼区間:", confidence_interval)

iqr = calculate_iqr(age_data)
print("年齢の四分位範囲 (IQR):", iqr)


data = dfb["CTDIvol"]

# 中央値の計算
median = statistics.median(data)

# 平均値の計算
mean = statistics.mean(data)

# 標準偏差の計算
standard_deviation = statistics.stdev(data)

print("中央値:", median)
print("平均値(mGy/cm):", mean)
print("標準偏差:", standard_deviation)



data = dfb["Age"]

# 中央値の計算
median = statistics.median(data)

# 平均値の計算
mean = statistics.mean(data)

# 標準偏差の計算
standard_deviation = statistics.stdev(data)

print("中央値:", median)
print("平均値:", mean)
print("標準偏差:", standard_deviation)




gender_list = dfb["Gender"]

male_percentage = calculate_male_percentage(gender_list)
print("Mの割合:", male_percentage, "%")






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

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale






#''''''''''''''''''''''
#変更すべきパラメータ
dicom0001 = ""
example0001 = pydicom.dcmread(dicom0001)
# dicom画像を読み込んでHU変換し、ReconstructionDiameter(FOVの直径：円の部分の直径)を取得
# そのデータを利用して●cm/pixelを導出し、MU_waterを/pixelに補正
# ReconstructionDiameterの単位はmmであることに注意
MU_water = 0.195 * (example0001.ReconstructionDiameter/10)/512


i0_id = 1.0 * (10 ** 5)
# I0_ldはIEEE_26543245.pdfのfigure2を参考に、40mAsでの値:1.0*10^5を暫定的に採用
#詳しくはおまけのstep4参照
#'''''''''''''''''''''''
#i0_id = 1.0 * (10 ** 5)
def step1(nparray,mu_air = 0,mu_water = MU_water):
  return nparray * (mu_water - mu_air)/1000 + mu_water

def step2(nparray):
  theta = np.linspace(0., 180., 1080, endpoint=False) 
  return radon(nparray, theta=theta, circle=True)

def step3(nparray_sinogram):
  return np.exp(- nparray_sinogram)

def step4(exp_minus_nparray_sinogram,I0_ld = i0_id):
  # I0_ldはIEEE_26543245.pdfのfigure2を参考に、40mAsでの値:1.0*10^5を暫定的に採用
  return I0_ld * exp_minus_nparray_sinogram

#calibration operationしていないので、ガウシアンノイズのパラメータは暫定的に０、低線量で支配的になるポアソンノイズを最低限使う
#🌟clampminは本来光子１個の計測値なので、clampmin=1.0
def step5_clamp(before_noise,mean = 0.0, sigma = 0.0,randomseed = 0,clamp=True,clampmin=1.0):
  np.random.seed(randomseed)

  if clamp:#clamp=Falseで計算すると、np.min(after_noise2)=0.0なので分母になれない。クランプ処理必要。
    minimum = np.max([clampmin,np.min(before_noise)])#2.2948070980502837e-41
    before_noise = np.clip(before_noise, minimum, None)
  
  G=np.random.normal(loc=mean, scale=sigma,size = before_noise.shape)
  afternoise = np.random.poisson(before_noise) + G 
  #plt.imshow(G,vmax=100,vmin=-100) # ⭕️下の関数Gでstep5_clampも組み込まれてますが、複数画像に関数Gを適用するときに一々plt.imshowされて動作が遅くなるかとおもわれるのでよければコメントアウトお願いします。
  afternoise = np.clip(afternoise,1.0,None)

  return afternoise

def step6(after_noise,I0_ld = i0_id):
  #I0_ldはstep4と連動。fig2の30mA付近から読み取る
  #after_noiseには0が入ってるとエラーするので注意。
  return -np.log(after_noise/I0_ld)

#FBP
def step7(noised_sinogram):
  theta = np.linspace(0., 180., 1080, endpoint=False) 
  return iradon(noised_sinogram, theta=theta, circle=True)# filter="ramp"

def step8(noised_lowCT,mu_air = 0,mu_water = MU_water):#mu_water = 0.195):
  return 1000 * (noised_lowCT - mu_water)/(mu_water - mu_air)








exampled = pydicom.dcmread("/content/drive/Shareddrives/data/anony/0001/2/1.2.392.200036.9125.10247187241139.64958978939.10.0.1")
hu = exampled.pixel_array # 変換式を省略

#画像中心からの距離を利用した補正を行う。imgはnumpy.ndarrayを想定
h, w = hu.shape[:2]
#print(h, w)#512,512
size = max([h, w])  # 幅、高の大きい方を確保
 
# 1.画像の中心からの距離画像作成
x = np.linspace(-w / size, w / size, w) #今回はxもyも(-1から1/512刻み)
y = np.linspace(-h / size, h / size, h)  # h!=wの時は長い方の辺が1になるように正規化
xx, yy = np.meshgrid(x, y)
R_2 = xx ** 2 + yy ** 2

##上記はnewG()においてimgsize_is_unknown = Falseにするなら必須


def correctvalue_outside_circle(img=hu,imgsize_is_unknown = False):

  if not imgsize_is_unknown:
    r_2 = R_2#４行上で定義したR_2を使用し、毎回r_2を作り直すのを避ける

  if imgsize_is_unknown:
    h, w = img.shape[:2]#512,512のはず
    #print(h, w)
    size = max([h, w])  # 幅、高の大きい方を確保
 
    # 1.画像の中心からの距離画像作成
    x = np.linspace(-w / size, w / size, w) #今回はxもyも(-1から1/512刻み)
    y = np.linspace(-h / size, h / size, h)  # h!=wの時は長い方の辺が1になるように正規化
    xx, yy = np.meshgrid(x, y)
    r_2 = xx ** 2 + yy ** 2

  # 2.診断サークルの外だけ0で塗りつぶす。
  img[r_2>1] = 0# ⭕️2021-02-16：R_2をr_2に訂正

  return img

def correctvalue_inside_circle(img=hu,imgsize_is_unknown = False):
    
  if not imgsize_is_unknown:
    r_2 = R_2#４行上で定義したR_2を使用し、毎回r_2を作り直すのを避ける 

  if imgsize_is_unknown:
    h, w = img.shape[:2]#512,512のはず
    #print(h, w)
    size = max([h, w])  # 幅、高の大きい方を確保
 
    # 1.画像の中心からの距離画像作成
    x = np.linspace(-w / size, w / size, w) #今回はxもyも(-1から1/512刻み)
    y = np.linspace(-h / size, h / size, h)  # h!=wの時は長い方の辺が1になるように正規化
    xx, yy = np.meshgrid(x, y)
    r_2 = xx ** 2 + yy ** 2

  # 2.診断サークルの内側だけ0で塗りつぶす。
  img[r_2<=1] = 0 # ⭕️2021-02-16：R_2をr_2に訂正

  return img





fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(2,3,1)
ax1.set_title("noised_lowCT\n_gaussnoise_sd=0")
ax1.imshow(passnewGnoised_lowCT_final_0, cmap=plt.cm.Greys_r)

ax2 = fig.add_subplot(2,3,2)
ax2.set_title("noised_lowCT\n_gaussnoise_sd=100^2")
ax2.imshow(passnewGnoised_lowCT_final_100, cmap=plt.cm.Greys_r)

ax3 = fig.add_subplot(2,3,3)
ax3.set_title("noised_lowCT\n_gaussnoise_sd=10000^2")
ax3.imshow(passnewGnoised_lowCT_final_10000, cmap=plt.cm.Greys_r)
#noiseがデカすぎると、何らかのエラーで４隅が-2000にならず0になっている。。

ax6 = fig.add_subplot(2,3,6)
ax6.set_title("Original")
ax6.imshow(original2, cmap=plt.cm.Greys_r)


fig.tight_layout()
plt.show()




このファイルの目的は、『aortic archのatlasを作成する』こと。 この段階では自動化する必要はない。 処理としては以下がある。

Resize to a resolution of 1mm^3 using bicubic interopration
Automatically segment lungs using thresholding and volumes cropped to localise the top of the heart
Rescale contrasts
Images were smoothed of high-frequency artifacts using a median filter
Raycasting was used to extract the heart cavity between the lungs
The heart muscle was automatically segmented using kmeans (k=4)
Cleaned of peripheral blood vessels
このうちstep 6以降は3D処理とする。

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

# resizeされた画像の表示 # しっかりと512*512よりも小さくなったことがわかる
plt.imshow(hu_resized, cmap=plt.cm.Greys_r)
plt.colorbar()
