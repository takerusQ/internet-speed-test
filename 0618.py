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
