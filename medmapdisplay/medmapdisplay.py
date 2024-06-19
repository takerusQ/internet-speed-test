
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
    
    
