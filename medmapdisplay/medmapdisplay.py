import pandas as pd

# Define the conditions and diseases data
conditions = ["液体貯留", "Free air", "石灰化", "管腔臓器の異常（拡張・狭窄・閉塞・壁肥厚・壁潰瘍・管内異物）", "実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "血管壁の異常（瘤・二重管腔（石灰化の遊離）、肥厚＋造影強化）","血流（造影）の異常（血管外漏洩・血栓（造影欠損）・奇形）", "低吸収域（腫瘤・血腫・梗塞・脂肪変性・炎症による浮腫）","脂肪組織の濃度変化"]

diseases = [
    {"name": "大動脈解離", "urgency": 3, "commonality": 1},
    {"name": "脳梗塞", "urgency": 3, "commonality": 2},
    {"name": "肝膿瘍", "urgency": 2, "commonality": 2},
    {"name": "腸管虚血", "urgency": 3, "commonality": 1},
    {"name": "脾梗塞", "urgency": 2, "commonality": 1},
    {"name": "腎梗塞", "urgency": 2, "commonality": 1},
    {"name": "膵臓腫瘍", "urgency": 2, "commonality": 3},
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
df_conditions_diseases.loc["脂肪組織の濃度変化", "胸水"] = "△:脂肪組織の濃度変化"

df_conditions_diseases.loc["液体貯留", "腹水"] = "〇:腹水の貯留"
df_conditions_diseases.loc["脂肪組織の濃度変化", "腹水"] = "△:脂肪組織の濃度変化"

df_conditions_diseases.loc["液体貯留", "胆嚢炎"] = "〇:胆汁の貯留"
df_conditions_diseases.loc["実質臓器の異常（腫大・造影剤取り込みの均一/不均一な増加・辺縁の不明瞭化）", "胆嚢炎"] = "〇:胆嚢壁の肥厚"


