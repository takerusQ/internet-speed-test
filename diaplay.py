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

import pandas as pd

# Define the conditions and diseases
conditions = ["緊急減圧開頭術の必要性", "大動脈損傷", "縦隔血腫", "広範な肺挫傷", "血気胸", "心嚢血腫", "腹腔内出血", "骨盤骨折", "後腹膜出血", "（肝/脾/膵/腎）損傷", "腸間膜内出血"]

diseases = [
    {"name": "頭部損傷", "urgency": 3, "commonality": 2},
    {"name": "大動脈損傷", "urgency": 3, "commonality": 1},
    {"name": "肺損傷", "urgency": 2, "commonality": 3},
    {"name": "腹腔内出血", "urgency": 3, "commonality": 2},
    {"name": "骨盤骨折", "urgency": 2, "commonality": 3},
    {"name": "臓器損傷", "urgency": 2, "commonality": 3},
]

# Create an empty DataFrame with conditions as index and diseases as columns
df_conditions_diseases = pd.DataFrame(0, index=conditions, columns=[d['name'] for d in diseases])

# Filling the DataFrame based on given conditions
df_conditions_diseases.loc["緊急減圧開頭術の必要性", "頭部損傷"] = 1
df_conditions_diseases.loc["大動脈損傷", "大動脈損傷"] = 1
df_conditions_diseases.loc["縦隔血腫", "大動脈損傷"] = 1
df_conditions_diseases.loc["広範な肺挫傷", "肺損傷"] = 1
df_conditions_diseases.loc["血気胸", "肺損傷"] = 1
df_conditions_diseases.loc["心嚢血腫", "肺損傷"] = 1
df_conditions_diseases.loc["腹腔内出血", "腹腔内出血"] = 1
df_conditions_diseases.loc["骨盤骨折", "骨盤骨折"] = 1
df_conditions_diseases.loc["後腹膜出血", "骨盤骨折"] = 1
df_conditions_diseases.loc["（肝/脾/膵/腎）損傷", "臓器損傷"] = 1
df_conditions_diseases.loc["腸間膜内出血", "臓器損傷"] = 1

# Adding urgency and commonality tags
disease_tags = pd.DataFrame(diseases).set_index("name")

# Merging the DataFrames
df_final = df_conditions_diseases.T.merge(disease_tags, left_index=True, right_index=True)

# Function to sort DataFrame by urgency and commonality
def sort_df(df, sort_by='urgency'):
    return df.sort_values(by=sort_by, ascending=False)

# Display the sorted DataFrame
sorted_df = sort_df(df_final)
print(sorted_df)






import pandas as pd

# Define the conditions and diseases data
conditions = ["液体停滞", "Free air", "石灰化", "腸閉塞", "消化管内滞留", "脾腫", "腸管の状態", "血管走行異常", "血管の状態", "低吸収域", "液体貯留"]
diseases = [
    {"name": "大動脈解離", "urgency": 3, "commonality": 1},
    {"name": "脳梗塞", "urgency": 2, "commonality": 2},
    {"name": "肝膿瘍", "urgency": 3, "commonality": 2},
    {"name": "腸管虚血", "urgency": 3, "commonality": 1},
    {"name": "脾梗塞", "urgency": 2, "commonality": 1},
    {"name": "腎梗塞", "urgency": 2, "commonality": 1},
    {"name": "膵臓腫瘍", "urgency": 1, "commonality": 3},
    {"name": "脳膿瘍", "urgency": 3, "commonality": 1},
    {"name": "嚢胞", "urgency": 1, "commonality": 3},
    {"name": "脂肪腫", "urgency": 1, "commonality": 3},
    {"name": "肝嚢胞", "urgency": 1, "commonality": 3},
    {"name": "消化管穿孔", "urgency": 3, "commonality": 2},
    {"name": "腹腔内感染", "urgency": 3, "commonality": 2},
    {"name": "外傷", "urgency": 3, "commonality": 2},
    {"name": "術後状態", "urgency": 2, "commonality": 2},
    {"name": "腹腔鏡手術後", "urgency": 2, "commonality": 2},
    {"name": "膵炎", "urgency": 3, "commonality": 2},
    {"name": "特発性遊離ガス", "urgency": 1, "commonality": 1},
    {"name": "出血", "urgency": 3, "commonality": 2},
    {"name": "急性膵炎", "urgency": 3, "commonality": 2},
    {"name": "腸閉塞", "urgency": 3, "commonality": 2},
    {"name": "腹膜炎", "urgency": 3, "commonality": 2},
    {"name": "心タンポナーデ", "urgency": 3, "commonality": 1},
    {"name": "腎膿瘍", "urgency": 3, "commonality": 2},
    {"name": "膀胱破裂", "urgency": 3, "commonality": 1},
    {"name": "胸水", "urgency": 2, "commonality": 2},
    {"name": "腹水", "urgency": 2, "commonality": 2},
    {"name": "胆嚢炎", "urgency": 3, "commonality": 2}
]

# Create an empty DataFrame with conditions as index and diseases as columns
df_conditions_diseases = pd.DataFrame('✖️', index=conditions, columns=[d['name'] for d in diseases])

# Filling the DataFrame based on given conditions
df_conditions_diseases.loc["液体停滞", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["液体停滞", "嚢胞"] = "〇"
df_conditions_diseases.loc["液体停滞", "肝嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["液体貯留", "嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "肝嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "消化管穿孔"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹腔内感染"] = "〇"
df_conditions_diseases.loc["液体貯留", "外傷"] = "〇"
df_conditions_diseases.loc["液体貯留", "術後状態"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹腔鏡手術後"] = "〇"
df_conditions_diseases.loc["液体貯留", "膵炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "急性膵炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "腸閉塞"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹膜炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "心タンポナーデ"] = "〇"
df_conditions_diseases.loc["液体貯留", "腎膿瘍"] = "〇"
df_conditions_diseases.loc["液体貯留", "膀胱破裂"] = "〇"
df_conditions_diseases.loc["液体貯留", "胸水"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹水"] = "〇"
df_conditions_diseases.loc["液体貯留", "胆嚢炎"] = "〇"
df_conditions_diseases.loc["Free air", "消化管穿孔"] = "〇"
df_conditions_diseases.loc["Free air", "腹腔内感染"] = "〇"
df_conditions_diseases.loc["Free air", "外傷"] = "〇"
df_conditions_diseases.loc["Free air", "術後状態"] = "〇"
df_conditions_diseases.loc["Free air", "腹腔鏡手術後"] = "〇"
df_conditions_diseases.loc["Free air", "膵炎"] = "〇"
df_conditions_diseases.loc["Free air", "特発性遊離ガス"] = "〇"
df_conditions_diseases.loc["石灰化", "大動脈解離"] = "〇"
df_conditions_diseases.loc["腸閉塞", "腸閉塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "脳梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "腸管虚血"] = "〇"
df_conditions_diseases.loc["低吸収域", "脾梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "腎梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "膵臓腫瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "脳膿瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "出血"] = "〇"
df_conditions_diseases.loc["低吸収域", "急性膵炎"] = "〇"

# Adding urgency and commonality tags
disease_tags = pd.DataFrame(diseases).set_index("name")

# Merging the DataFrames
df_final = df_conditions_diseases.T.merge(disease_tags, left_index=True, right_index=True)

# Function to sort DataFrame by urgency and commonality
def sort_df(df, sort_by='urgency'):
    return df.sort_values(by=sort_by, ascending=False)

# Display the sorted DataFrame
sorted_df = sort_df(df_final)

# Function to color-code the urgency levels
def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'black'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# Display the styled DataFrame
styled_df




#######@#
####@##@@

import pandas as pd

# Define the conditions and diseases data
conditions = ["液体停滞", "Free air", "石灰化", "腸閉塞", "消化管内滞留", "脾腫", "腸管の状態", "血管走行異常", "血管の状態", "低吸収域", "液体貯留"]
diseases = [
    {"name": "大動脈解離", "urgency": 3, "commonality": 1},
    {"name": "脳梗塞", "urgency": 2, "commonality": 2},
    {"name": "肝膿瘍", "urgency": 3, "commonality": 2},
    {"name": "腸管虚血", "urgency": 3, "commonality": 1},
    {"name": "脾梗塞", "urgency": 2, "commonality": 1},
    {"name": "腎梗塞", "urgency": 2, "commonality": 1},
    {"name": "膵臓腫瘍", "urgency": 1, "commonality": 3},
    {"name": "脳膿瘍", "urgency": 3, "commonality": 1},
    {"name": "嚢胞", "urgency": 1, "commonality": 3},
    {"name": "脂肪腫", "urgency": 1, "commonality": 3},
    {"name": "肝嚢胞", "urgency": 1, "commonality": 3},
    {"name": "消化管穿孔", "urgency": 3, "commonality": 2},
    {"name": "腹腔内感染", "urgency": 3, "commonality": 2},
    {"name": "外傷", "urgency": 3, "commonality": 2},
    {"name": "術後状態", "urgency": 2, "commonality": 2},
    {"name": "腹腔鏡手術後", "urgency": 2, "commonality": 2},
    {"name": "膵炎", "urgency": 3, "commonality": 2},
    {"name": "特発性遊離ガス", "urgency": 1, "commonality": 1},
    {"name": "出血", "urgency": 3, "commonality": 2},
    {"name": "急性膵炎", "urgency": 3, "commonality": 2},
    {"name": "腸閉塞", "urgency": 3, "commonality": 2},
    {"name": "腹膜炎", "urgency": 3, "commonality": 2},
    {"name": "心タンポナーデ", "urgency": 3, "commonality": 1},
    {"name": "腎膿瘍", "urgency": 3, "commonality": 2},
    {"name": "膀胱破裂", "urgency": 3, "commonality": 1},
    {"name": "胸水", "urgency": 2, "commonality": 2},
    {"name": "腹水", "urgency": 2, "commonality": 2},
    {"name": "胆嚢炎", "urgency": 3, "commonality": 2}
]

# Create an empty DataFrame with conditions as index and diseases as columns
df_conditions_diseases = pd.DataFrame('✖️', index=conditions, columns=[d['name'] for d in diseases])

# Filling the DataFrame based on given conditions with explanations
df_conditions_diseases.loc["液体停滞", "大動脈解離"] = "△: 胸水など"
df_conditions_diseases.loc["石灰化", "大動脈解離"] = "〇"
df_conditions_diseases.loc["血管走行異常", "大動脈解離"] = "△: 血管の異常な走行"
df_conditions_diseases.loc["血管の状態", "大動脈解離"] = "△: 血管の状態の変化"
df_conditions_diseases.loc["低吸収域", "大動脈解離"] = "△: 低吸収域の変化"
df_conditions_diseases.loc["液体貯留", "大動脈解離"] = "△: 胸水など"

# Further filling for other diseases based on general medical knowledge
df_conditions_diseases.loc["液体停滞", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["液体停滞", "嚢胞"] = "〇"
df_conditions_diseases.loc["液体停滞", "肝嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["液体貯留", "嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "肝嚢胞"] = "〇"
df_conditions_diseases.loc["液体貯留", "消化管穿孔"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹腔内感染"] = "〇"
df_conditions_diseases.loc["液体貯留", "外傷"] = "〇"
df_conditions_diseases.loc["液体貯留", "術後状態"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹腔鏡手術後"] = "〇"
df_conditions_diseases.loc["液体貯留", "膵炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "急性膵炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "腸閉塞"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹膜炎"] = "〇"
df_conditions_diseases.loc["液体貯留", "心タンポナーデ"] = "〇"
df_conditions_diseases.loc["液体貯留", "腎膿瘍"] = "〇"
df_conditions_diseases.loc["液体貯留", "膀胱破裂"] = "〇"
df_conditions_diseases.loc["液体貯留", "胸水"] = "〇"
df_conditions_diseases.loc["液体貯留", "腹水"] = "〇"
df_conditions_diseases.loc["液体貯留", "胆嚢炎"] = "〇"
df_conditions_diseases.loc["Free air", "消化管穿孔"] = "〇"
df_conditions_diseases.loc["Free air", "腹腔内感染"] = "〇"
df_conditions_diseases.loc["Free air", "外傷"] = "〇"
df_conditions_diseases.loc["Free air", "術後状態"] = "〇"
df_conditions_diseases.loc["Free air", "腹腔鏡手術後"] = "〇"
df_conditions_diseases.loc["Free air", "膵炎"] = "〇"
df_conditions_diseases.loc["Free air", "特発性遊離ガス"] = "〇"
df_conditions_diseases.loc["石灰化", "大動脈解離"] = "〇"
df_conditions_diseases.loc["腸閉塞", "腸閉塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "脳梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "肝膿瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "腸管虚血"] = "〇"
df_conditions_diseases.loc["低吸収域", "脾梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "腎梗塞"] = "〇"
df_conditions_diseases.loc["低吸収域", "膵臓腫瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "脳膿瘍"] = "〇"
df_conditions_diseases.loc["低吸収域", "出血"] = "〇"
df_conditions_diseases.loc["低吸収域", "急性膵炎"] = "〇"

# Adding urgency and commonality tags
disease_tags = pd.DataFrame(diseases).set_index("name")

# Merging the DataFrames
df_final = df_conditions_diseases.T.merge(disease_tags, left_index=True, right_index=True)

# Function to sort DataFrame by urgency and commonality
def sort_df(df):
    return df.sort_values(by=['urgency', 'commonality'], ascending=[False, False])

# Sorting the DataFrame
sorted_df = sort_df(df_final)

# Function to color-code the urgency levels
def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'black'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# Save as PNG
def save_styled_dataframe_as_image(df, filename):
    dfi.export(df, filename)

# Save the styled DataFrame as an image
save_styled_dataframe_as_image(styled_df, "disease_condition_table.png")