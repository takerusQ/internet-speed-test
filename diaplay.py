# 
#orgdir = "//"
dstdir = 

path_upper = dstdir+"/.png"
path_middle = dstdir+"/.png"
path_lower = dstdir+"/.png"

img_upper = cv2.imread(path_upper, cv2.IMREAD_ANYDEPTH)[75:390, 47:467]
img_middle = cv2.imread(path_middle, cv2.IMREAD_ANYDEPTH)[75:390, 47:467]
img_lower = cv2.imread(path_lower, cv2.IMREAD_ANYDEPTH)[75:390, 47:467]
# 基本設定
plt.rcParams['figure.dpi'] = 300
fig = plt.figure(figsize=(8, 4.5))

### original ###
ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(img_upper, cmap="gray")


# 目盛りラベルの削除（参照：https://qiita.com/tsukada_cs/items/8d31a25cd7c860690270）
ax1.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)

# 軸目盛りの削除
ax1.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
# 右・上の枠線の削除
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('.pdf')
plt.show()
