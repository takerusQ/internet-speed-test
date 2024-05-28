

pip install ipykernel

python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

pip install ipykernel


from pptx import Presentation
from pptx.util import Inches

# Create a presentation object
prs = Presentation()

# Title slide
slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "業務紹介"
subtitle.text = "現在取り組んでいるプロジェクトの概要"

# Slide for introduction
slide_layout = prs.slide_layouts[1]
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "紹介"
content = (
    "私は医療機器メーカーでモバイルなイメージングモダリティを考案するエンジニアとして、"
    "特にCTでの画像の異常所見の超早期発見に注力しています。現在取り組んでいる3つの主要プロジェクトとその業務に割いている時間の割合は以下の通りです。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Slide for Projects
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "現在取り組んでいるプロジェクト"

content = (
    "1. リアルタイム異常検出アルゴリズムの開発\n"
    "   時間の割合: 40%\n"
    "   概要: CTスキャンから得られる画像データをリアルタイムで解析し、異常所見（腫瘍、血栓、出血など）を即座に検出するためのアルゴリズムを開発しています。\n"
    "   主にディープラーニング技術を活用し、大量のデータセットを用いてモデルをトレーニングしています。\n\n"
    "2. ポータブルCTスキャナーの設計とプロトタイピング\n"
    "   時間の割合: 35%\n"
    "   概要: 移動が容易で、緊急医療現場や遠隔地でも使用可能なポータブルCTスキャナーの設計に取り組んでいます。\n"
    "   軽量化と高性能を両立させるための機械設計と、エネルギー効率の向上を目指したプロトタイピングを進めています。\n\n"
    "3. クラウドベースの診断支援システムの構築\n"
    "   時間の割合: 25%\n"
    "   概要: 収集したCT画像データをクラウド上で管理し、専門医がリモートで診断支援を行えるシステムを構築しています。\n"
    "   データのセキュリティとプライバシー保護を確保しながら、迅速かつ正確な診断をサポートすることを目指しています。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Save the presentation
output_path = "業務紹介.pptx"
prs.save(output_path)

print(f"Presentation saved as {output_path}")












from pptx import Presentation
from pptx.util import Inches

# Create a presentation object
prs = Presentation()

# Title slide
slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "業務紹介"
subtitle.text = "現在取り組んでいるプロジェクトの概要"

# Add a slide for each project
slide_layout = prs.slide_layouts[1]

# Slide for introduction
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "紹介"
content = (
    "私は医療機器メーカーでモバイルなイメージングモダリティを考案するエンジニアとして、"
    "特にCTでの画像の異常所見の超早期発見に注力しています。現在取り組んでいる3つの主要プロジェクトとその業務に割いている時間の割合は以下の通りです。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Slide for Project 1
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "1. リアルタイム異常検出アルゴリズムの開発"
content = (
    "時間の割合: 40%\n"
    "概要: CTスキャンから得られる画像データをリアルタイムで解析し、異常所見（腫瘍、血栓、出血など）を即座に検出するためのアルゴリズムを開発しています。"
    "主にディープラーニング技術を活用し、大量のデータセットを用いてモデルをトレーニングしています。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Slide for Project 2
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "2. ポータブルCTスキャナーの設計とプロトタイピング"
content = (
    "時間の割合: 35%\n"
    "概要: 移動が容易で、緊急医療現場や遠隔地でも使用可能なポータブルCTスキャナーの設計に取り組んでいます。"
    "軽量化と高性能を両立させるための機械設計と、エネルギー効率の向上を目指したプロトタイピングを進めています。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Slide for Project 3
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
title.text = "3. クラウドベースの診断支援システムの構築"
content = (
    "時間の割合: 25%\n"
    "概要: 収集したCT画像データをクラウド上で管理し、専門医がリモートで診断支援を行えるシステムを構築しています。"
    "データのセキュリティとプライバシー保護を確保しながら、迅速かつ正確な診断をサポートすることを目指しています。"
)
text_box = slide.shapes.placeholders[1].text_frame
text_box.text = content

# Save the presentation
output_path = "業務紹介.pptx"
prs.save(output_path)

print(f"Presentation saved as {output_path}")













from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes = A4

# PDFファイルの生成
output_path = "/mnt/data/業務紹介.pdf"
c = canvas.Canvas(output_path, pagesize=A4)

# フォントの設定
c.setFont("Helvetica", 12)

# タイトル
c.drawString(200, 800, "業務紹介")

# 内容
y_position = 760

c.drawString(50, y_position, "紹介")
y_position -= 20
intro_text = (
    "私は医療機器メーカーでモバイルなイメージングモダリティを考案するエンジニアとして、"
    "特にCTでの画像の異常所見の超早期発見に注力しています。現在取り組んでいる3つの主要プロジェクトとその業務に割いている時間の割合は以下の通りです。"
)
for line in intro_text.split('\n'):
    c.drawString(50, y_position, line)
    y_position -= 20

# プロジェクト1
project1_title = "1. リアルタイム異常検出アルゴリズムの開発"
project1_body = (
    "時間の割合: 40%\n"
    "概要: CTスキャンから得られる画像データをリアルタイムで解析し、異常所見（腫瘍、血栓、出血など）を即座に検出するためのアルゴリズムを開発しています。"
    "主にディープラーニング技術を活用し、大量のデータセットを用いてモデルをトレーニングしています。"
)
c.drawString(50, y_position, project1_title)
y_position -= 20
for line in project1_body.split('\n'):
    c.drawString(50, y_position, line)
    y_position -= 20

# プロジェクト2
project2_title = "2. ポータブルCTスキャナーの設計とプロトタイピング"
project2_body = (
    "時間の割合: 35%\n"
    "概要: 移動が容易で、緊急医療現場や遠隔地でも使用可能なポータブルCTスキャナーの設計に取り組んでいます。"
    "軽量化と高性能を両立させるための機械設計と、エネルギー効率の向上を目指したプロトタイピングを進めています。"
)
c.drawString(50, y_position, project2_title)
y_position -= 20
for line in project2_body.split('\n'):
    c.drawString(50, y_position, line)
    y_position -= 20

# プロジェクト3
project3_title = "3. クラウドベースの診断支援システムの構築"
project3_body = (
    "時間の割合: 25%\n"
    "概要: 収集したCT画像データをクラウド上で管理し、専門医がリモートで診断支援を行えるシステムを構築しています。"
    "データのセキュリティとプライバシー保護を確保しながら、迅速かつ正確な診断をサポートすることを目指しています。"
)
c.drawString(50, y_position, project3_title)
y_position -= 20
for line in project3_body.split('\n'):
    c.drawString(50, y_position, line)
    y_position -= 20

# 保存
c.save()

output_path









現在取り組んでいる3つのプロジェクトとその業務に割いている時間の割合は以下の通りです：

1. **リアルタイム異常検出アルゴリズムの開発** (Real-time Anomaly Detection Algorithm Development)
   - **時間の割合**: 40%
   - **概要**: CTスキャンから得られる画像データをリアルタイムで解析し、異常所見（腫瘍、血栓、出血など）を即座に検出するためのアルゴリズムを開発しています。主にディープラーニング技術を活用し、大量のデータセットを用いてモデルをトレーニングしています。

2. **ポータブルCTスキャナーの設計とプロトタイピング** (Design and Prototyping of Portable CT Scanner)
   - **時間の割合**: 35%
   - **概要**: 移動が容易で、緊急医療現場や遠隔地でも使用可能なポータブルCTスキャナーの設計に取り組んでいます。軽量化と高性能を両立させるための機械設計と、エネルギー効率の向上を目指したプロトタイピングを進めています。

3. **クラウドベースの診断支援システムの構築** (Development of Cloud-based Diagnostic Support System)
   - **時間の割合**: 25%
   - **概要**: 収集したCT画像データをクラウド上で管理し、専門医がリモートで診断支援を行えるシステムを構築しています。データのセキュリティとプライバシー保護を確保しながら、迅速かつ正確な診断をサポートすることを目指しています。

これらのプロジェクトは、それぞれ異なる面からCT画像の異常所見の早期発見を支援し、全体として医療の質を向上させることを目指しています。






