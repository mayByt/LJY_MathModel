# 问题一图件统一字体版

本目录为问题一图件的非覆盖式重导出版。原目录 `论文图片/问题一` 未被替换。

## 字体规范

- 中文正文、坐标轴、图例：Noto Serif CJK SC（思源宋体的 Noto 发行版）
- 中文标题、分面标题：Noto Sans CJK SC Bold（思源黑体的 Noto 发行版）
- 英文与数字：Times New Roman
- 数学公式：LaTeX 数学语法，Computer Modern 数学字形

## 图件范围

- `P1_F00` 至 `P1_F10` 全部正式版本；
- `P1_F01` 采用完整相关矩阵与稳定 CRITIC 权重版；
- `P1_F02` 采用带领域样本点带的质量景观与映射版。

共 11 份 PDF 和 11 份 PNG。F00 另保留可编辑 DrawIO 与 SVG 源文件，不再保留备选命名文件。

## 复现方式

```bash
export P1_UNIFIED_FONTS=1
export P1_FIGURE_OUTPUT_DIR="$PWD/论文图片/问题一_统一字体版"
python3 问题一/plot_problem1_paper_figures.py
python3 问题一/plot_problem1_sensitivity_f10.py
python3 问题一/rebuild_p1_f00_unified_fonts.py
python3 问题一/build_p1_unified_font_manifest.py
```

统一字体仅改变字形与因字形宽度变化产生的必要排版，不改变数据、颜色编码、模型结果和图形含义。
