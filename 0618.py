import numpy as np
from skimage.measure import label, regionprops
from skimage.draw import disk
import matplotlib.pyplot as plt
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
