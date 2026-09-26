# 论文绘图 · 参考重设计 V2

四问各20张，共 **80张论文插图**，已按用户指定的 `gmgc2026F/绘图` 成图标准重设计，并完成逐图独立审阅。

## 查看与使用

- [离线图册：按题目筛选、搜索、放大](图册.html)
- [80张索引与PNG/PDF链接](图表说明.md)
- [问题一说明](图表说明_Q1.md) · [问题二说明](图表说明_Q2.md) · [问题三说明](图表说明_Q3.md) · [问题四说明](图表说明_Q4.md)
- [问题一总览](previews/q1_top20/contact_q1_01.png) · [问题二总览](previews/q2_top20/contact_q2_01.png) · [问题三总览](previews/q3_top20/contact_q3_01.png) · [问题四总览](previews/q4_top20/contact_q4_01.png)
- [80张PNG/PDF打包](Top20_参考重绘_PNG_PDF.zip)

正式输出位于 `output/q1` 至 `output/q4`。每图都有300dpi PNG、字体嵌入PDF、统计JSON，以及来源与口径说明。图册中点击图像可打开原图，PNG和PDF链接可直接用于选图与论文排版。

## 本轮设计

保留用户指定“柔绿森林”8色，以浅填充、中深描边、固定领域色和共享坐标分面重组图形。加入必要的分布、区间、对照和样本支持量；字体、图例与数据层级对照参考原图。图内仅保留变量、单位、条件、必要图例和核心数字，完整解释放图注。

中文宋体、英文和数字Times New Roman；数学字母由STIXGeneral、Unicode上标由STIX Two Text的TrueType字形补充。已消除OTF嵌入兼容警告。全部PDF实际宽度不超过165mm。

参考只用于视觉组织与辅助绘图代码，参考项目结果未作为本项目数值使用。主结果仍取本项目的Q1/Q2 `results_p12_final_v2`、Q3 `results_v1`、Q4 `results_v2`。

## 审阅与核验

- 80/80当前PNG有独立逐图目视通过记录，并与当前SHA-256匹配。
- 80对PNG/PDF齐全，300dpi、单页PDF、嵌入字体和无缺字/布局告警检查通过。
- [最终验收清单](reviews/technical/final_acceptance.json)保存逐图哈希、审阅者、实际尺寸与技术状态。
- `reviews/Q*-*.json`保存当前审阅，`reviews/history`保存修订历史。
- [冻结结果核验](reviews/technical/frozen_results_check.json)确认建模结果未改变。

数据解释边界保留在各图图注：半合成与估算不当作实测；失败优化终态明确区分；区间不随意相加；Q4预测仍以2025-03-13为起点。

## 重画

在本目录使用已有独立环境：

```bash
.venv/bin/python make_all.py q1 q2 q3 q4
.venv/bin/python make_all.py --id Q3-05
```

数据路径与Top20顺序见 `config.py`，色板与字体见 `config/style.json`、`style/`。原始附件根为项目父目录 `中文题目/F题/real_attachments`，可用 `PAPER_FIGURE_RAW`覆盖。问题一恢复质量明细的流程见 `q1/README.md`；其缓存单独保存且未替换冻结汇总。

重新出图后，改变了PNG哈希的图片须重新审阅，不能继承旧图的通过结论。仅刷新图册和说明可运行 `.venv/bin/python tools/build_delivery.py`。

## 历史与调研

被用户否定的旧版已隔离到 `archive/v1_before_reference_redesign/`，不混入当前正式输出。其余候选图仍保留在 [110项候选清单](候选图清单.md)，本次交付为每问优先Top20。

使用用户指定的 [paper-figure-pipeline](https://github.com/wyzz973/paper-figure-pipeline)；参考版本与许可见本项目调研记录及 `tools/SKILL_LICENSE`。新视觉标准见 [参考风格重设计标准](参考风格重设计标准.md)。
