####@##@@

import pandas as pd

# Define the conditions and diseases data
conditions = ["液体貯留", "Free air", "石灰化", "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）","血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）","脂肪組織の濃度変化"]
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
