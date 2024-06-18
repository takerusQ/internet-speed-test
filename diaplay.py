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


Image 1 (Labeled CT scan):

	1.	緊急開頭術が必要の有無
	2.	大動脈の弓部から峡部で大動脈損傷の有無、縦隔血腫の有無
	3.	肺底部で広範な肺挫傷、血気胸、心のう血腫の有無を確認
	4.	横隔膜から骨盤腔まで一気にみてモリソン窩、脾腎境界、膀胱直腸窩に腹腔内出血の有無を確認
	5.	骨盤骨折や後腹膜出血の有無を確認しながら頭側へ移動
	6.	実質臓器（肝/脾/膵/腎）損傷、腸間膜内出血の有無を確認しながら尾側へ移動

Image 2 (CT findings table):

	•	頭部: 緊急減圧開頭術の必要性
	•	大動脈（肺動脈レベル）: 大動脈損傷、縦隔血腫
	•	肺底部: 広範な肺挫傷、血気胸、心嚢血腫
	•	横隔膜から骨盤腔まで: 腹腔内出血
	•	骨盤骨折・後腹膜出血: 骨盤骨折、後腹膜出血
	•	実質臓器: （肝/脾/膵/腎）損傷、腸間膜内出血

