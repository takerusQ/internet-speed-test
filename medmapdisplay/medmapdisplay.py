
import pandas as pd

# Define the conditions and diseases data
conditions = ["液体貯留",
              "Free air",
              "石灰化",
              "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）",
              "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）",
              "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）",
              "血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）",
              "低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）",
              "脂肪組織の濃度変化"]

diseases = [
    {"name": "大動脈解離", "urgency": 3, "commonality": 1},
    {"name": "脳梗塞", "urgency": 3, "commonality": 2},
    {"name": "肝膿瘍", "urgency": 2, "commonality": 2},
    {"name": "腸管虚血", "urgency": 3, "commonality": 1},
    {"name": "脾梗塞", "urgency": 2, "commonality": 1},
    {"name": "腎梗塞", "urgency": 2, "commonality": 1},
    {"name": "膵臓腫瘍", "urgency": 2, "commonality": 1},
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
#あとでわかりやすいように追加分はわけて書いておく
diseases += [
    {"name": "脳内出血", "urgency": 3, "commonality": 2},
    {"name": "動脈硬化", "urgency": 2, "commonality": 3},
    {"name": "腎結石", "urgency": 2, "commonality": 3},
    {"name": "胆石", "urgency": 2, "commonality": 3},
    {"name": "慢性膵炎", "urgency": 2, "commonality": 2},
    {"name": "骨転移", "urgency": 3, "commonality": 2},
    {"name": "骨髄炎", "urgency": 3, "commonality": 1}
]



# Create an empty DataFrame with conditions as index and diseases as columns
df_conditions_diseases = pd.DataFrame('✖️', index=conditions, columns=[d['name'] for d in diseases])

# Filling the DataFrame based on each disease's CT findings with explanations
df_conditions_diseases.loc["液体貯留", "大動脈解離"] = "△:心嚢水が見られることがある"
df_conditions_diseases.loc["石灰化", "大動脈解離"] = "〇:内膜の石灰化"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "大動脈解離"] = "〇:二重管腔"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "大動脈解離"] = "〇:造影剤の漏れ"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "大動脈解離"] = "△:血腫が見られることがある"

df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "脳梗塞"] = "〇:脳の低吸収域"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "脳梗塞"] = "〇:造影欠損"

df_conditions_diseases.loc["液体貯留", "肝膿瘍"] = "〇:膿の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "肝膿瘍"] = "〇:不均一な増加"

df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "腸管虚血"] = "〇:腸管の低吸収域"

df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "脾梗塞"] = "〇:脾臓の低吸収域"

df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "腎梗塞"] = "〇:腎臓の低吸収域"

df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "膵臓腫瘍"] = "〇:不均一な増加"

df_conditions_diseases.loc["液体貯留", "脳膿瘍"] = "〇:膿の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "脳膿瘍"] = "〇:不均一な増加"

df_conditions_diseases.loc["液体貯留", "嚢胞"] = "〇:嚢胞の液体"

df_conditions_diseases.loc["液体貯留", "脂肪腫"] = "✖️"

df_conditions_diseases.loc["液体貯留", "肝嚢胞"] = "〇:嚢胞の液体"

df_conditions_diseases.loc["Free air", "消化管穿孔"] = "〇:腹腔内のFree air"
df_conditions_diseases.loc["液体貯留", "消化管穿孔"] = "〇:腹水"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "消化管穿孔"] = "〇:壁潰瘍"

df_conditions_diseases.loc["液体貯留", "腹腔内感染"] = "〇:膿の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "腹腔内感染"] = "〇:不均一な増加"

df_conditions_diseases.loc["液体貯留", "外傷"] = "〇:出血"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "外傷"] = "〇:造影剤の漏れ"

df_conditions_diseases.loc["液体貯留", "術後状態"] = "△:術後の液体貯留が見られることがある"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "術後状態"] = "△:手術後の組織変化"

df_conditions_diseases.loc["液体貯留", "腹腔鏡手術後"] = "△:術後の液体貯留が見られることがある"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "腹腔鏡手術後"] = "△:手術後の組織変化"

df_conditions_diseases.loc["液体貯留", "膵炎"] = "〇:膵液の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "膵炎"] = "〇:膵臓の腫大"

df_conditions_diseases.loc["Free air", "特発性遊離ガス"] = "〇:腹腔内のFree air"

df_conditions_diseases.loc["液体貯留", "出血"] = "〇:出血"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "出血"] = "〇:血腫"

df_conditions_diseases.loc["液体貯留", "急性膵炎"] = "〇:膵液の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "急性膵炎"] = "〇:膵臓の腫大"

df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "腸閉塞"] = "〇:拡張"

df_conditions_diseases.loc["液体貯留", "腹膜炎"] = "〇:腹水"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "腹膜炎"] = "△:腹膜の腫大"

df_conditions_diseases.loc["液体貯留", "心タンポナーデ"] = "〇:心嚢液"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "心タンポナーデ"] = "〇:低吸収域の血腫"

df_conditions_diseases.loc["液体貯留", "腎膿瘍"] = "〇:膿の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "腎膿瘍"] = "〇:不均一な増加"

df_conditions_diseases.loc["液体貯留", "膀胱破裂"] = "〇:尿の漏れ"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "膀胱破裂"] = "〇:造影剤の漏れ"

df_conditions_diseases.loc["液体貯留", "胸水"] = "〇:胸水の貯留"
df_conditions_diseases.loc["脂肪組織の濃度変化", "胸水"] = "✖️"

df_conditions_diseases.loc["液体貯留", "腹水"] = "〇:腹水の貯留"
df_conditions_diseases.loc["脂肪組織の濃度変化", "腹水"] = "✖️"

df_conditions_diseases.loc["液体貯留", "胆嚢炎"] = "〇:胆汁の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "胆嚢炎"] = "〇:胆嚢壁の肥厚"

#あとづけ🌟🌟🌟🌟🌟🌟🌟🌟🌟
# Filling the DataFrame based on each disease's CT findings with explanations

# 脳内出血
df_conditions_diseases.loc["液体貯留", "脳内出血"] = "✖️"
df_conditions_diseases.loc["Free air", "脳内出血"] = "✖️"
df_conditions_diseases.loc["石灰化", "脳内出血"] = "✖️"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "脳内出血"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "脳内出血"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "脳内出血"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "脳内出血"] = "〇:造影剤の漏れ"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "脳内出血"] = "△:急性期には高吸収域"
df_conditions_diseases.loc["脂肪組織の濃度変化", "脳内出血"] = "✖️"

# 動脈硬化
df_conditions_diseases.loc["液体貯留", "動脈硬化"] = "✖️"
df_conditions_diseases.loc["Free air", "動脈硬化"] = "✖️"
df_conditions_diseases.loc["石灰化", "動脈硬化"] = "〇:石灰化プラーク"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "動脈硬化"] = "〇:狭窄"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "動脈硬化"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "動脈硬化"] = "〇:血管壁の肥厚"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "動脈硬化"] = "△:血栓"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "動脈硬化"] = "✖️"
df_conditions_diseases.loc["脂肪組織の濃度変化", "動脈硬化"] = "✖️"

# 腎結石
df_conditions_diseases.loc["液体貯留", "腎結石"] = "✖️"
df_conditions_diseases.loc["Free air", "腎結石"] = "✖️"
df_conditions_diseases.loc["石灰化", "腎結石"] = "〇:結石"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "腎結石"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "腎結石"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "腎結石"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "腎結石"] = "✖️"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "腎結石"] = "✖️"
df_conditions_diseases.loc["脂肪組織の濃度変化", "腎結石"] = "✖️"

# 胆石
df_conditions_diseases.loc["液体貯留", "胆石"] = "✖️"
df_conditions_diseases.loc["Free air", "胆石"] = "✖️"
df_conditions_diseases.loc["石灰化", "胆石"] = "〇:結石"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "胆石"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "胆石"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "胆石"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "胆石"] = "✖️"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "胆石"] = "✖️"
df_conditions_diseases.loc["脂肪組織の濃度変化", "胆石"] = "✖️"

# 慢性膵炎
df_conditions_diseases.loc["液体貯留", "慢性膵炎"] = "△:膵液の貯留が見られることがある"
df_conditions_diseases.loc["Free air", "慢性膵炎"] = "✖️"
df_conditions_diseases.loc["石灰化", "慢性膵炎"] = "〇:石灰化"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "慢性膵炎"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "慢性膵炎"] = "〇:不均一な増加"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "慢性膵炎"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "慢性膵炎"] = "✖️"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "慢性膵炎"] = "✖️"
df_conditions_diseases.loc["脂肪組織の濃度変化", "慢性膵炎"] = "〇:脂肪組織の濃度変化"

# 骨転移
# 骨転移
df_conditions_diseases.loc["液体貯留", "骨転移"] = "✖️"
df_conditions_diseases.loc["Free air", "骨転移"] = "✖️"
df_conditions_diseases.loc["石灰化", "骨転移"] = "△:骨形成性の転移が見られることがある"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "骨転移"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "骨転移"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "骨転移"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "骨転移"] = "✖️"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "骨転移"] = "△:骨融解性の転移が見られることがある"
df_conditions_diseases.loc["脂肪組織の濃度変化", "骨転移"] = "✖️"

# 骨髄炎
df_conditions_diseases.loc["液体貯留", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["Free air", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["石灰化", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）", "骨髄炎"] = "✖️"
df_conditions_diseases.loc["血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "骨髄炎"] = "△:炎症による血流変化が見られることがある"
df_conditions_diseases.loc["低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）", "骨髄炎"] = "〇:炎症による浮腫"
df_conditions_diseases.loc["脂肪組織の濃度変化", "骨髄炎"] = "✖️"


#あとづけ修正👹👹👹
# Filling the DataFrame based on each disease's CT findings with explanations

# 急性膵炎
df_conditions_diseases.loc["脂肪組織の濃度変化", "急性膵炎"] = "〇:膵周囲の脂肪ストランディング"

# 慢性膵炎
df_conditions_diseases.loc["脂肪組織の濃度変化", "慢性膵炎"] = "〇:膵周囲の脂肪ストランディング"

# 急性胆嚢炎
df_conditions_diseases.loc["脂肪組織の濃度変化", "胆嚢炎"] = "〇:胆嚢周囲の脂肪ストランディング"

# 腹腔内感染
df_conditions_diseases.loc["脂肪組織の濃度変化", "腹腔内感染"] = "〇:脂肪組織の濃度変化"

# 外傷
df_conditions_diseases.loc["脂肪組織の濃度変化", "外傷"] = "〇:出血による脂肪組織の濃度変化"


#あとづけ疾患削除👹👹👹
df_conditions_diseases.drop(columns=["腎膿瘍"], inplace=True)
df_conditions_diseases.drop(columns=["外傷"], inplace=True)
#df_conditions_diseases.drop(columns=[""], inplace=True)
#df_conditions_diseases.drop(columns=[""], inplace=True)
#df_conditions_diseases.drop(columns=[""], inplace=True)

## Rename the index🌟🌟🌟🌟🌟🌟🌟🌟ｈ
df_conditions_diseases.rename(index={"低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）": "臓器外の低吸収域（血腫・梗塞・脂肪変性・炎症による浮腫）"}, inplace=True)
###🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

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
        color = 'gray'
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


#styled_df


# 日本語フォントの設定（例としてMS Gothicを使用）
plt.rcParams['font.family'] = 'MS Gothic'  # WindowsのMS Gothicフォントを使用

# プロットのスタイルを設定
fig, ax = plt.subplots(figsize=(15, 10))  # 画像サイズを設定
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_conditions_diseases.values, colLabels=df_conditions_diseases.columns, rowLabels=df_conditions_diseases.index, cellLoc='center', loc='center')

# テーブルのレイアウトを自動調整
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # サイズを調整

# 画像として保存
#plt.savefig("conditions_diseases_table2.png", bbox_inches='tight')

# CSVとして保存（エンコーディングを指定）
#df_conditions_diseases.to_csv("conditions_diseases_table.csv", encoding='utf-8-sig')






###@####@@@@@@#@#@@

import pandas as pd

# サンプルのデータフレーム（sorted_df）を作成
data = {
    "条件1": ["△:心嚢水が見られることがある", "✖️", "〇:内膜の石灰化", "✖️", "✖️", "〇:二重管腔", "〇:造影剤の漏れ", "△:血腫が見られることがある", "✖️"],
    "条件2": ["✖️", "✖️", "✖️", "✖️", "✖️", "✖️", "〇:造影欠損", "〇:脳の低吸収域", "✖️"],
    "条件3": ["〇:膿の貯留", "✖️", "✖️", "✖️", "〇:不均一な増加", "✖️", "✖️", "✖️", "✖️"],
    "urgency": [3, 1, 2],
    "commonality": [2, 3, 1]
}

index = [
    "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", 
    "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", 
    "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）"
]

sorted_df = pd.DataFrame(data, index=index)

def highlight_conditions(val):
    highlight_texts = [
        "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
    ]
    if any(text in val for text in highlight_texts):
        return 'font-size: 12pt; background-color: lightgreen'
    return ''

def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'gray'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .applymap(highlight_conditions)\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# 表示


File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\base.py:921, in IndexOpsMixin._map_values(self, mapper, na_action, convert)
    918 if isinstance(arr, ExtensionArray):
    919     return arr.map(mapper, na_action=na_action)
--> 921 return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\algorithms.py:1743, in map_array(arr, mapper, na_action, convert)
   1741 values = arr.astype(object, copy=False)
   1742 if na_action is None:
-> 1743     return lib.map_infer(values, mapper, convert=convert)
   1744 else:
   1745     return lib.map_infer_mask(
   1746         values, mapper, mask=isna(values).view(np.uint8), convert=convert
   1747     )

File lib.pyx:2972, in pandas._libs.lib.map_infer()

Cell In[16], line 5, in highlight_conditions(val)
      1 def highlight_conditions(val):
      2     highlight_texts = [
      3         "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
      4     ]
----> 5     if any(text in val for text in highlight_texts):
      6         return 'font-size: 12pt; background-color: lightgreen'
      7     return ''

Cell In[16], line 5, in <genexpr>(.0)
      1 def highlight_conditions(val):
      2     highlight_texts = [
      3         "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
      4     ]
----> 5     if any(text in val for text in highlight_texts):
      6         return 'font-size: 12pt; background-color: lightgreen'
      7     return ''

TypeError: argument of type 'int' is not iterable
<pandas.io.formats.style.Styler at 0x211309d32f0>
styled_df


def highlight_conditions(val):
    highlight_texts = [
        "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
    ]
    if isinstance(val, str) and any(text in val for text in highlight_texts):
        return 'font-size: 12pt; background-color: lightgreen'
    return ''

def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'gray'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .applymap(highlight_conditions)\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# 表示
styled_df
    
    
import pandas as pd

# サンプルのデータフレーム（sorted_df）を作成
data = {
    "条件1": ["△:心嚢水が見られることがある", "✖️", "〇:内膜の石灰化"],
    "条件2": ["✖️", "✖️", "✖️"],
    "条件3": ["〇:膿の貯留", "✖️", "✖️"],
    "urgency": [3, 1, 2],
    "commonality": [2, 3, 1]
}

index = [
    "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", 
    "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", 
    "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）"
]

sorted_df = pd.DataFrame(data, index=index)

def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'gray'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

def highlight_index(val):
    highlight_texts = [
        "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
    ]
    if any(text in val for text in highlight_texts):
        return 'font-size: 12pt; background-color: lightgreen'
    return ''

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .applymap(highlight_index, subset=pd.IndexSlice[:, :])\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# 表示
styled_df

import pandas as pd

# サンプルのデータフレーム（sorted_df）を作成
data = {
    "条件1": ["△:心嚢水が見られることがある", "✖️", "〇:内膜の石灰化"],
    "条件2": ["✖️", "✖️", "✖️"],
    "条件3": ["〇:膿の貯留", "✖️", "✖️"],
    "urgency": [3, 1, 2],
    "commonality": [2, 3, 1]
}

index = [
    "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", 
    "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", 
    "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）"
]

sorted_df = pd.DataFrame(data, index=index)

def highlight_urgency(val):
    color = 'white'
    if val == 1:
        color = 'gray'
    elif val == 2:
        color = 'yellow'
    elif val == 3:
        color = 'red'
    return f'background-color: {color}'

def highlight_headers(s):
    highlight_texts = [
        "管腔臓器", "実質臓器", "血管壁", "血流", "脂肪組織"
    ]
    return ['font-size: 12pt; background-color: lightgreen' if any(text in col for text in highlight_texts) else '' for col in s]

# Applying styles
styled_df = sorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .apply(highlight_headers, axis=1, subset=pd.IndexSlice[:, :])\
                          .apply(highlight_headers, axis=0, subset=pd.IndexSlice[:, :])\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# 表示
styled_df












partlysorted_df=sorted_df.loc[:, ["脂肪組織の濃度変化"]]

partlystyled_df = partlysorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
                          .applymap(highlight_urgency, subset=['commonality'])\
                          .applymap(highlight_index, subset=pd.IndexSlice[:, :])\
                          .set_table_styles([
                              {'selector': 'th', 'props': [('font-size', '12pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
                              {'selector': 'td', 'props': [('font-size', '10pt'), ('text-align', 'center')]}
                          ])\
                          .set_properties(**{'max-width': '150px', 'font-size': '10pt'})

# 表示
partlystyled_df

C:\Users\root\AppData\Local\Temp\ipykernel_15848\1431276768.py:3: FutureWarning: Styler.applymap has been deprecated. Use Styler.map instead.
  partlystyled_df = partlysorted_df.style.applymap(highlight_urgency, subset=['urgency'])\
C:\Users\root\AppData\Local\Temp\ipykernel_15848\1431276768.py:4: FutureWarning: Styler.applymap has been deprecated. Use Styler.map instead.
  .applymap(highlight_urgency, subset=['commonality'])\
C:\Users\root\AppData\Local\Temp\ipykernel_15848\1431276768.py:5: FutureWarning: Styler.applymap has been deprecated. Use Styler.map instead.
  .applymap(highlight_index, subset=pd.IndexSlice[:, :])\
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\formatters.py:347, in BaseFormatter.__call__(self, obj)
    345     method = get_real_method(obj, self.print_method)
    346     if method is not None:
--> 347         return method()
    348     return None
    349 else:

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style.py:405, in Styler._repr_html_(self)
    400 """
    401 Hooks into Jupyter notebook rich display system, which calls _repr_html_ by
    402 default if an object is returned at the end of a cell.
    403 """
    404 if get_option("styler.render.repr") == "html":
--> 405     return self.to_html()
    406 return None

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style.py:1345, in Styler.to_html(self, buf, table_uuid, table_attributes, sparse_index, sparse_columns, bold_headers, caption, max_rows, max_columns, encoding, doctype_html, exclude_styles, **kwargs)
   1342     obj.set_caption(caption)
   1344 # Build HTML string..
-> 1345 html = obj._render_html(
   1346     sparse_index=sparse_index,
   1347     sparse_columns=sparse_columns,
   1348     max_rows=max_rows,
   1349     max_cols=max_columns,
   1350     exclude_styles=exclude_styles,
   1351     encoding=encoding or get_option("styler.render.encoding"),
   1352     doctype_html=doctype_html,
   1353     **kwargs,
   1354 )
   1356 return save_to_buffer(
   1357     html, buf=buf, encoding=(encoding if buf is not None else None)
   1358 )

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style_render.py:204, in StylerRenderer._render_html(self, sparse_index, sparse_columns, max_rows, max_cols, **kwargs)
    192 def _render_html(
    193     self,
    194     sparse_index: bool,
   (...)
    198     **kwargs,
    199 ) -> str:
    200     """
    201     Renders the ``Styler`` including all applied styles to HTML.
    202     Generates a dict with necessary kwargs passed to jinja2 template.
    203     """
--> 204     d = self._render(sparse_index, sparse_columns, max_rows, max_cols, "&nbsp;")
    205     d.update(kwargs)
    206     return self.template_html.render(
    207         **d,
    208         html_table_tpl=self.template_html_table,
    209         html_style_tpl=self.template_html_style,
    210     )

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style_render.py:161, in StylerRenderer._render(self, sparse_index, sparse_columns, max_rows, max_cols, blank)
    147 def _render(
    148     self,
    149     sparse_index: bool,
   (...)
    153     blank: str = "",
    154 ):
    155     """
    156     Computes and applies styles and then generates the general render dicts.
    157 
    158     Also extends the `ctx` and `ctx_index` attributes with those of concatenated
    159     stylers for use within `_translate_latex`
    160     """
--> 161     self._compute()
    162     dxs = []
    163     ctx_len = len(self.index)

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style_render.py:256, in StylerRenderer._compute(self)
    254 r = self
    255 for func, args, kwargs in self._todo:
--> 256     r = func(self)(*args, **kwargs)
    257 return r

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\io\formats\style.py:2027, in Styler._map(self, func, subset, **kwargs)
   2025     subset = IndexSlice[:]
   2026 subset = non_reducing_slice(subset)
-> 2027 result = self.data.loc[subset].map(func)
   2028 self._update_ctx(result)
   2029 return self

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1184, in _LocationIndexer.__getitem__(self, key)
   1182     if self._is_scalar_access(key):
   1183         return self.obj._get_value(*key, takeable=self._takeable)
-> 1184     return self._getitem_tuple(key)
   1185 else:
   1186     # we by definition only have the 0th axis
   1187     axis = self.axis or 0

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1377, in _LocIndexer._getitem_tuple(self, tup)
   1374 if self._multi_take_opportunity(tup):
   1375     return self._multi_take(tup)
-> 1377 return self._getitem_tuple_same_dim(tup)

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1020, in _LocationIndexer._getitem_tuple_same_dim(self, tup)
   1017 if com.is_null_slice(key):
   1018     continue
-> 1020 retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021 # We should never have retval.ndim < self.ndim, as that should
   1022 #  be handled by the _getitem_lowerdim call above.
   1023 assert retval.ndim == self.ndim

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1420, in _LocIndexer._getitem_axis(self, key, axis)
   1417     if hasattr(key, "ndim") and key.ndim > 1:
   1418         raise ValueError("Cannot index with multidimensional key")
-> 1420     return self._getitem_iterable(key, axis=axis)
   1422 # nested tuple slicing
   1423 if is_nested_tuple(key, labels):

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1360, in _LocIndexer._getitem_iterable(self, key, axis)
   1357 self._validate_key(key, axis)
   1359 # A collection of keys
-> 1360 keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361 return self.obj._reindex_with_indexers(
   1362     {axis: [keyarr, indexer]}, copy=True, allow_dups=True
   1363 )

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexing.py:1558, in _LocIndexer._get_listlike_indexer(self, key, axis)
   1555 ax = self.obj._get_axis(axis)
   1556 axis_name = self.obj._get_axis_name(axis)
-> 1558 keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1560 return keyarr, indexer

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexes\base.py:6200, in Index._get_indexer_strict(self, key, axis_name)
   6197 else:
   6198     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
-> 6200 self._raise_if_missing(keyarr, indexer, axis_name)
   6202 keyarr = self.take(indexer)
   6203 if isinstance(key, Index):
   6204     # GH 42790 - Preserve name from an Index

File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\indexes\base.py:6249, in Index._raise_if_missing(self, key, indexer, axis_name)
   6247 if nmissing:
   6248     if nmissing == len(indexer):
-> 6249         raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6251     not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
   6252     raise KeyError(f"{not_found} not in index")

KeyError: "None of [Index(['urgency'], dtype='object')] are in the [columns]"
