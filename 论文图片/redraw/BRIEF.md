# 论文终稿重绘统一规范（所有重绘代理必须遵守）

背景：2026 研究生数学建模竞赛 F 题论文（算力约束下大模型资源配置）。目标：国一水平的竞赛论文插图——信息密度高、一眼看懂结论、全文配色字号统一；不是实验报告式的“诊断截图”。

## 硬性规则
1. 只用仓库中的真实数据：`/home/chenxu/MayByte/LJY/论文图片/问题X/figure_data/*.csv(.gz)` 以及 `/home/chenxu/MayByte/LJY/问题X/results*/` 下的结果文件。禁止编造、插值出不存在的数据点；需要派生量时只做确定性计算（如比值、分位数），并在脚本中写清楚公式。
2. 风格统一：脚本开头
   ```python
   import sys; sys.path.insert(0, "/home/chenxu/MayByte/LJY/论文图片")
   from paper_style import *   # 颜色常量、SEQ_CMAP、DIV_CMAP、seq_colors、lighten、panel_label、save、FULL_W、HALF_W、PANEL_H
   apply_style()
   ```
   不要再 rcParams.update 覆盖字体/字号；不要用 seaborn 主题、viridis/magma/Set2 等其他色板。
   - 类别色只用 BLUE, ORANGE, GREEN, RED, PURPLE, GRAY（及 lighten/darken 派生）。
   - 连续量用 SEQ_CMAP；有正负号的量（相关、交互、差值）用 DIV_CMAP 并以 0 为中心（TwoSlopeNorm 或对称 vmin/vmax）。
   - 语义固定：指数成本=BLUE、幂函数成本=ORANGE、对数成本=GREEN；观测=BLUE、拟合/模型=ORANGE、参考/估算=GRAY、阈值/临界线=RED 虚线；三档预算 1e19/1e22/1e24 与三档规模 1M/60M/1B 用 seq_colors(3)（浅→深）；算力情景 κ=1/0.5/0.25 = BLUE/ORANGE/GREEN；规模贡献=BLUE、非规模贡献=ORANGE；质量语义块 教育价值/表达整洁/推理信息/噪声重复/结构充分 = BLUE/ORANGE/GREEN/RED/PURPLE；映射等级 直接/近直接/推断 = BLUE/GREEN/GRAY。
3. 尺寸：按物理尺寸作图，宽度 FULL_W(6.3in)，单行子图高度约 PANEL_H(2.35in)，多行按比例；不得在 LaTeX 中缩放到 60% 以下才能看清。字号不要手动放大缩小（标题 9pt、标签 8.5pt、刻度 7.5pt 已设好）。一张图最多 4 个子图，子图用 panel_label(ax,'a') 标注。
4. 中文标签、坐标轴含单位（如“参数量 $N$ / 十亿”“算力预算 $C$ / FLOPs”）；不要图内大标题（论文用 caption）；图例放不遮挡数据的位置；不要大段文字注释，最多 1–2 个关键数值标注。
5. 禁止：3D 柱状图、3D 曲面、雷达图、饼图、过多阴影网格、过细的文字说明框。
6. 输出：`save(fig, "/home/chenxu/MayByte/LJY/overleaf-paper/images/noX/<文件名>")`（自动生成 .pdf 与 .png）。文件名见任务清单。脚本保存为 `/home/chenxu/MayByte/LJY/论文图片/redraw/redraw_pX.py`，可一次性运行生成本问全部图。
7. 自检：每张图生成后用 Read 工具查看 PNG，检查中文无方块、文字不重叠、图例不遮挡、颜色符合语义；有问题就改，直到满意。
8. 不要修改任何 .tex 文件，不要改动其他问题的图片目录，不要 git commit。

## 汇报格式（最终回复）
对每张图给出：文件名 | 子图内容（a/b/c 各画了什么）| 图中能读出的 2–4 个关键数值结论（用于写图注和正文）| 用到的数据文件。最后列出放弃重绘或无法用现有数据完成的图及原因。
