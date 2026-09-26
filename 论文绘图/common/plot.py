# -*- coding: utf-8 -*-
"""绘图公共层：固定尺寸建图、统一导出（PNG 300 dpi + PDF）、自动排版检查与常用图元。

每张图单独成文件，不拼子图。尺寸按最终印刷尺寸设定，字号在各图之间保持一致，
便于之后手动拼版时直接等比例排布。

常用写法::

    fig, ax, pal = P.figure("q1", "M")                 # 尺寸键或 (宽mm, 高mm)
    ax.plot(x, y, color=pal.line["c1"], label="甲组")
    P.legend_top(ax)
    P.save(fig, "q1", "Q1-05_中文短名", stats={"均值": m})
"""
from __future__ import annotations

import json
import logging
import warnings
import hashlib
import re
import os
import fcntl
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)   # 屏蔽 PDF 子集化宋体时的 “MERG NOT subset” 提示
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as mcolors

import config as C
from style import theme

MM = 1 / 25.4

# 常用尺寸（宽 × 高，mm）。S = 半栏，M = 三分之二栏，L/W/XL = 通栏
SIZES = {
    "S": (78, 60), "Sq": (78, 72), "T": (78, 96),
    "M": (107, 72), "Mq": (107, 96),
    "L": (160, 90), "W": (160, 64), "Wt": (160, 44), "XL": (160, 120),
}
DPI = 300


# ---------------------------------------------------------------------------
# 建图与导出
# ---------------------------------------------------------------------------

def figure(question: str, size="S", layout: str | None = "constrained", **kw):
    """应用该问配色并创建单图。size 可为 SIZES 键或 (宽mm, 高mm)。返回 (fig, ax, pal)。"""
    pal = theme.use(question)
    w, h = SIZES[size] if isinstance(size, str) else size
    fig = plt.figure(figsize=(w * MM, h * MM), layout=layout)
    if layout == "constrained":
        fig.get_layout_engine().set(w_pad=1.5 * MM, h_pad=1.5 * MM, wspace=0.02, hspace=0.02)
    ax = fig.add_subplot(111, **kw)
    return fig, ax, pal


def fixed_figure(question: str, height_mm: float, width_mm: float = 160, left_mm: float = 24,
                 right_mm: float = 17, bottom_mm: float = 10, top_mm: float = 2, **kw):
    """固定画布与边距的单图（不用自动布局），返回 (fig, ax, pal)。

    需要上下拼接、共享时间轴的一组组件（多轨对齐图、案例卡）用相同的 width/left/right，
    保存时用 save(..., tight=False)，拼起来坐标轴就严格对齐。图例、色条放进左右边距内。
    bottom_mm 默认留出刻度与横轴标签（约 10 mm）；中间轨道不画横轴标签时可减小。
    """
    pal = theme.use(question)
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    ax = fig.add_axes([left_mm / width_mm, bottom_mm / height_mm,
                       1 - (left_mm + right_mm) / width_mm, 1 - (bottom_mm + top_mm) / height_mm], **kw)
    return fig, ax, pal


def _drawn_texts(fig):
    """收集实际会绘制的文字（排除视野外刻度等不绘制的标签）。"""
    out = []
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            span = hi - lo if hi > lo else 1
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 * span <= loc <= hi + 1e-9 * span):
                    continue
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        out.append(lab)
        out += [t for t in (ax.xaxis.label, ax.yaxis.label, ax.title) if t.get_visible() and t.get_text().strip()]
        out += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            out += [t for t in leg.get_texts() if t.get_text().strip()]
    out += [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for leg in fig.legends:
        out += [t for t in leg.get_texts() if t.get_text().strip()]
    return out


def _text_box(text, renderer):
    from matplotlib.text import Text
    return Text.get_window_extent(text, renderer)


def check_layout(fig, min_gap_px: float = 0.5, check_bounds: bool = True):
    """返回文字重叠与越界问题列表。

    - 越界只在按固定画布导出（tight=False）时检查；紧凑导出会自动把所有文字包含在内。
    - 斜向（非 0°/90°）的刻度标签彼此之间不做重叠判断，因为外接框会高估其占位。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _drawn_texts(fig)
    boxes = [_text_box(t, r) for t in texts]
    slanted = [abs(t.get_rotation() % 90) > 1e-6 for t in texts]
    fb = fig.bbox
    issues = []
    if check_bounds:
        for i, b in enumerate(boxes):
            if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
                issues.append(f"越界: {texts[i].get_text()!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if slanted[i] and slanted[j]:
                continue
            a, b = boxes[i], boxes[j]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > min_gap_px and h > min_gap_px:
                issues.append(f"重叠: {texts[i].get_text()!r} × {texts[j].get_text()!r}")
    return issues


def _data_points(ax, samples_per_segment: int = 8):
    """数据点的显示坐标：折线（含线段插值采样）与散点。只有两点且无标记的参考线不计。"""
    pts = []
    for ln in ax.get_lines():
        if not ln.get_visible() or ln.get_label().startswith("_legend"):
            continue
        xy = np.asarray(ln.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(1)] if xy.ndim == 2 else xy
        if len(xy) == 0:
            continue
        has_marker = ln.get_marker() not in (None, "None", "", " ")
        if len(xy) <= 2 and not has_marker:
            continue
        d = ln.get_transform().transform(xy)
        if ln.get_linestyle() not in ("None", "", " ") and len(d) <= 400:
            seg = [d[:1]]
            for a, b in zip(d[:-1], d[1:]):
                t = np.linspace(0, 1, samples_per_segment + 1)[1:, None]
                seg.append(a + (b - a) * t)
            d = np.concatenate(seg)
        pts.append(d)
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and coll.get_visible() and len(coll.get_offsets()):
            pts.append(coll.get_offset_transform().transform(np.asarray(coll.get_offsets(), float)))
    return np.concatenate(pts) if pts else np.empty((0, 2))


def _bar_boxes(ax, r):
    """柱状等矩形数据元素的显示框（排除铺满坐标区的背景块）。"""
    out = []
    ab = ax.get_window_extent(r)
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_visible() and p.get_facecolor()[3] > 0.05:
            b = p.get_window_extent(r)
            if b.width < 0.6 * ab.width and b.height < 0.95 * ab.height and b.width * b.height > 1:
                out.append(b)
    return out


def _inside(pts, bb, pad=1.0):
    if not len(pts):
        return 0
    return int(((pts[:, 0] > bb.x0 + pad) & (pts[:, 0] < bb.x1 - pad) &
                (pts[:, 1] > bb.y0 + pad) & (pts[:, 1] < bb.y1 - pad)).sum())


def _box_overlap(a, b, min_area=4.0):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w > 0 and h > 0 and w * h > min_area


def audit_figure(fig):
    """启发式审查（需人工确认）：图例遮挡数据、文字压在数据点上、多色编码却无图例/色条/直接标注。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    warns = []
    data_axes = [ax for ax in fig.axes if not hasattr(ax, "_colorbar")]
    all_axes = list(fig.axes) + [ch for ax in fig.axes for ch in getattr(ax, "child_axes", [])]
    has_cbar = any(hasattr(ax, "_colorbar") for ax in all_axes)
    legends = [ax.get_legend() for ax in data_axes if ax.get_legend() is not None] + list(fig.legends)

    def visible_points(ax):
        """只保留落在坐标区显示范围内的点（坐标轴外被裁掉的点不计）。"""
        pts = _data_points(ax)
        ab = ax.get_window_extent(r)
        if not len(pts):
            return pts
        keep = (pts[:, 0] >= ab.x0) & (pts[:, 0] <= ab.x1) & (pts[:, 1] >= ab.y0) & (pts[:, 1] <= ab.y1)
        return pts[keep]

    for leg in legends:
        lb = leg.get_window_extent(r)
        for ax in data_axes:
            n = _inside(visible_points(ax), lb)
            n += sum(_box_overlap(lb, b) for b in _bar_boxes(ax, r))
            n += sum(_box_overlap(lb, im.get_window_extent(r), 20) for im in ax.images
                     if _box_overlap(lb, ax.get_window_extent(r), 20))
            if n:
                warns.append(f"审查: 图例遮挡数据（{n}）")
                break
    for ax in data_axes:
        pts = visible_points(ax)
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                n = _inside(pts, _text_box(t, r), pad=0.5)
                if n:
                    warns.append(f"审查: 文字压数据 {t.get_text()[:12]!r}（{n}）")

    def categorical_ticks(ax):
        """类别已由坐标轴刻度标签说明（非数值刻度 ≥ 2 个）时，不再要求图例。"""
        labs = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text().strip()]

        def numeric(x):
            s = x
            for ch in ("−", "%", "$", "^", "{", "}", "\\mathdefault", "≥", "≤", "×", "10"):
                s = s.replace(ch, "-" if ch == "−" else "")
            try:
                float(s.strip() or "0")
                return True
            except ValueError:
                return False
        return sum(not numeric(x) for x in labs) >= 2

    any_legend = bool(legends)
    if not has_cbar and not any_legend:
        for ax in data_axes:
            if categorical_ticks(ax):
                continue
            cols = set()
            for ln in ax.get_lines():
                if ln.get_visible() and len(ln.get_xydata()) > 2 and not ln.get_label().startswith("_legend"):
                    cols.add(mcolors.to_hex(ln.get_color()))
            for coll in ax.collections:
                if isinstance(coll, PathCollection) and coll.get_visible():
                    fc = {mcolors.to_hex(c) for c in coll.get_facecolors()}
                    if len(fc) <= 12:
                        cols |= fc
            for p in ax.patches:
                if isinstance(p, Rectangle) and p.get_visible() and p.get_facecolor()[3] > 0.05:
                    cols.add(mcolors.to_hex(p.get_facecolor()))
            cols -= {"#ffffff", "#000000"}
            if len(cols) >= 2 and len([t for t in ax.texts if t.get_text().strip()]) < 2:
                warns.append(f"审查: 可能缺图例（{len(cols)} 种颜色，无图例/色条/直接标注）")
    return warns


def save(fig, question: str, name: str, stats: dict | None = None, formats=("png", "pdf"), tight: bool = True,
         caption: str = "", sources=None):
    """导出 PNG（300 dpi）与 PDF，检查排版与缺字，并把关键数值写入 output/<question>/stats.json。

    tight=True：按内容裁去空白边（大多数图）。
    tight=False：保留建图时的精确画布与边距（fixed_figure 建的拼接组件），同时检查文字是否越界。
    """
    out = C.OUTPUT / question
    out.mkdir(parents=True, exist_ok=True)
    # rc 里的 savefig.bbox="tight" 会覆盖 bbox_inches=None，所以固定画布时必须在 rc 层面改回 standard
    with warnings.catch_warnings(record=True) as caught, \
            matplotlib.rc_context({"savefig.bbox": "tight" if tight else "standard"}):
        warnings.simplefilter("always")
        issues = check_layout(fig, check_bounds=not tight) + audit_figure(fig)
        for ext in formats:
            final_path = out / f"{name}.{ext}"
            temp_path = out / f".{name}.{os.getpid()}.pending.{ext}"
            fig.savefig(temp_path, format=ext, dpi=DPI)
            temp_path.replace(final_path)
    msgs = {str(w.message) for w in caught}
    issues += [f"缺字: {g}" for g in sorted(m for m in msgs if "Glyph" in m or "missing from" in m)]
    issues += [f"布局: {g}" for g in sorted(m for m in msgs if "layout not applied" in m
                                             or "not compatible with tight_layout" in m)]
    plt.close(fig)
    if stats is not None:
        sf = out / "stats.json"
        with (out / '.stats.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            allstats = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
            allstats[name] = _jsonable(stats)
            tmp_stats = out / f'.stats.{os.getpid()}.tmp'
            tmp_stats.write_text(json.dumps(allstats, ensure_ascii=False, indent=1), encoding='utf-8')
            tmp_stats.replace(sf)
            fcntl.flock(lock, fcntl.LOCK_UN)
    figure_id = re.match(r'Q\d-\d+', name).group(0)
    png = out / f'{name}.png'
    meta = {'id': figure_id, 'name': name, 'question': question,
            'png': str(png), 'pdf': str(out / f'{name}.pdf'),
            'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
            'size_mm': [round(x*25.4,2) for x in fig.get_size_inches()],
            'caption': caption, 'sources': [str(x) for x in (sources or [])],
            'stats': _jsonable(stats or {}), 'automatic_issues': issues,
            'created_at': datetime.now(timezone.utc).isoformat(), 'status':'pending_review', 'visual_revision':'reference_v2'}
    (out / f'{name}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    queue = C.PLOT / 'review_queue'
    queue.mkdir(exist_ok=True)
    (queue / f'{figure_id}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    status = "OK" if not issues else "检查: " + "；".join(issues[:8]) + (" …" if len(issues) > 8 else "")
    print(f"[{question}] {name}: {status}")
    return issues


def _jsonable(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


# ---------------------------------------------------------------------------
# 常用样式
# ---------------------------------------------------------------------------

def legend_top(ax, ncol: int | None = None, handles=None, labels=None, center: bool = False,
               pad: float = 0.02, **kw):
    """图例放在坐标区正上方一行（最常用的“不遮挡数据”位置）。项多时传 ncol 分行。"""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    loc, x = ("lower center", 0.5) if center else ("lower left", 0.0)
    return ax.legend(handles, labels, loc=loc, bbox_to_anchor=(x, 1 + pad), ncol=ncol or len(handles),
                     borderaxespad=0, **kw)


def clean_heatmap(ax, grid_color="white", lw=0.6):
    """热图：去边框、去刻度线，只在单元格之间画白色分隔线（外缘不画，避免露出细色边）。"""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ny, nx = ax.images[-1].get_array().shape[:2] if ax.images else (0, 0)
    if nx and ny and lw:
        ax.set_xticks(np.arange(0.5, nx - 1), minor=True)
        ax.set_yticks(np.arange(0.5, ny - 1), minor=True)
        ax.grid(which="minor", color=grid_color, lw=lw)
        ax.tick_params(which="minor", length=0)


def colorbar(fig, mappable, ax, ticks=None, label=None, pad=0.02, orientation="vertical", shrink=1.0,
             match_axes=False, width=0.035, width_mm=None):
    """色条（必须写 label：量名，有单位时写单位）。

    match_axes=True 时以内嵌坐标区贴在主图旁，与坐标区等高/等宽（适合等比例的三元图、方阵热图）；
    width_mm 给定时色条粗细按毫米固定，否则按坐标区尺寸的比例 width。
    """
    if match_axes:
        if width_mm is not None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            if orientation == "vertical":
                cax = inset_axes(ax, width=width_mm * MM, height=f"{shrink * 100:.0f}%", loc="center left",
                                 bbox_to_anchor=(1 + pad, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            else:
                cax = inset_axes(ax, width=f"{shrink * 100:.0f}%", height=width_mm * MM, loc="upper center",
                                 bbox_to_anchor=(0, -pad, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        elif orientation == "vertical":
            cax = ax.inset_axes([1 + pad, (1 - shrink) / 2, width, shrink])
        else:
            cax = ax.inset_axes([(1 - shrink) / 2, -pad - width, shrink, width])
        cb = fig.colorbar(mappable, cax=cax, ticks=ticks, orientation=orientation)
    else:
        cb = fig.colorbar(mappable, ax=ax, ticks=ticks, fraction=0.05, pad=pad, aspect=24,
                          orientation=orientation, shrink=shrink)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1.5)
    if label:
        cb.set_label(label)
    return cb


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color=matplotlib.rcParams["grid.color"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def zero_line(ax, axis="y", value=0.0, **kw):
    """零线等参考线（浅灰虚线，放在数据下层）。需要说明含义时给 label 并进图例。"""
    ref = theme.CURRENT.ref if theme.CURRENT else "#A9B4BA"
    style = dict(color=ref, lw=0.6, ls=(0, (3, 2)), zorder=0.5)
    style.update(kw)
    return (ax.axhline if axis == "y" else ax.axvline)(value, **style)


def violin_box(ax, data, positions, fills, lines, width=0.75, box_w=0.12, bw=0.25, vert=True, alpha=0.85,
               show_box=True, cut=2.0, clip=None):
    """粉彩小提琴 + 白色箱线。

    密度曲线向两端各外延 cut 个带宽（与 seaborn 默认一致），两端收尖而不是截平；
    clip=(下限, 上限) 把外延限制在取值范围内（如份额、概率）。
    箱线：白框为四分位距，深色横线为中位数，须线到 1.5 倍四分位距内的最远点。
    """
    from scipy.stats import gaussian_kde
    bodies = []
    for d, p, fc, ec in zip(data, positions, fills, lines):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if len(d) < 2 or np.std(d) == 0:
            continue
        kde = gaussian_kde(d, bw_method=bw)
        h = kde.factor * d.std(ddof=1)
        lo, hi = d.min() - cut * h, d.max() + cut * h
        if clip is not None:
            lo, hi = max(lo, clip[0]), min(hi, clip[1])
        grid = np.linspace(lo, hi, 256)
        dens = kde(grid)
        dens = dens / dens.max() * width / 2
        if vert:
            body = ax.fill_betweenx(grid, p - dens, p + dens, facecolor=fc, edgecolor=ec, lw=0.6, alpha=alpha, zorder=2)
        else:
            body = ax.fill_between(grid, p - dens, p + dens, facecolor=fc, edgecolor=ec, lw=0.6, alpha=alpha, zorder=2)
        bodies.append(body)
    if show_box:
        for d, p in zip(data, positions):
            d = np.asarray(d, float)
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            iqr = q3 - q1
            lo = d[d >= q1 - 1.5 * iqr].min()
            hi = d[d <= q3 + 1.5 * iqr].max()
            if vert:
                ax.plot([p, p], [lo, hi], color="#555555", lw=0.6, zorder=3)
                ax.add_patch(Rectangle((p - box_w / 2, q1), box_w, iqr, fc="white", ec="#555555", lw=0.6, zorder=4))
                ax.plot([p - box_w / 2, p + box_w / 2], [med, med], color="#333333", lw=0.9, zorder=5)
            else:
                ax.plot([lo, hi], [p, p], color="#555555", lw=0.6, zorder=3)
                ax.add_patch(Rectangle((q1, p - box_w / 2), iqr, box_w, fc="white", ec="#555555", lw=0.6, zorder=4))
                ax.plot([med, med], [p - box_w / 2, p + box_w / 2], color="#333333", lw=0.9, zorder=5)
    return {"bodies": bodies}


def jitter(n, scale=0.08, seed=0):
    return np.random.default_rng(seed).uniform(-scale, scale, n)


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def group_bootstrap_weights(groups, reps=1000, seed=0):
    """按独立单位（如同一来源的样本、受试者、地区）有放回重采样：返回 reps × N 的样本权重。

    同一组内的样本不独立时，区间必须按组重采样，否则会偏窄。
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    w = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
    return w[:, inv].astype(float)


def weighted_mean_ci(values, weights, level=95):
    """values: N 或 N×K；weights: reps × N（来自 group_bootstrap_weights）。返回点估计与百分位区间。"""
    v = np.asarray(values, dtype=float)
    point = v.mean(0)
    boot = (weights @ v) / weights.sum(1, keepdims=True) if v.ndim > 1 else (weights @ v) / weights.sum(1)
    a = (100 - level) / 2
    return point, np.percentile(boot, a, 0), np.percentile(boot, 100 - a, 0)


# ---------------------------------------------------------------------------
# 三元图（手写，字体与配色完全受控）：适合三部分组成、三分类概率
# ---------------------------------------------------------------------------

TRI = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]])        # 左下、顶、右下


def to_ternary(p):
    """p: N×3（对应 TRI 三个顶点的份额，行和为 1）→ 平面坐标。"""
    return np.asarray(p) @ TRI


def ternary_frame(ax, labels, grid=(0.2, 0.4, 0.6, 0.8), pad=0.075, fontsize=None):
    ax.add_patch(Polygon(TRI, closed=True, fill=False, ec="#8C9AA2", lw=0.7, zorder=5))
    for g in grid:
        for k in range(3):
            a = np.zeros(3)
            a[k] = g
            rest = [i for i in range(3) if i != k]
            p1 = a.copy()
            p1[rest[0]] = 1 - g
            p2 = a.copy()
            p2[rest[1]] = 1 - g
            xy = to_ternary(np.vstack([p1, p2]))
            ax.plot(xy[:, 0], xy[:, 1], color="#E6EAED", lw=0.5, zorder=0, label="_legend_grid")
    offs = [(-pad * 0.6, -pad * 0.9), (0, pad * 0.75), (pad * 0.6, -pad * 0.9)]
    for (x, y), lab, (dx, dy) in zip(TRI, labels, offs):
        ax.text(x + dx, y + dy, lab, ha="center", va="center", fontsize=fontsize)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.13, np.sqrt(3) / 2 + 0.1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def simplex_density(ax, xy, color_fill, color_line, levels=(0.5,), bw=0.3, fill_alpha=0.22, lw=0.8, grid_n=220):
    """在三元图内画包含给定比例样本的密度等高线（裁剪在单纯形内）。"""
    from matplotlib.path import Path as MPath
    from scipy.stats import gaussian_kde
    gx, gy = np.meshgrid(np.linspace(0, 1, grid_n), np.linspace(0, np.sqrt(3) / 2, grid_n))
    inside = MPath(TRI).contains_points(np.c_[gx.ravel(), gy.ravel()], radius=1e-9).reshape(gx.shape)
    kde = gaussian_kde(np.asarray(xy).T, bw_method=bw)
    zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    zz[~inside] = np.nan
    dens = kde(np.asarray(xy).T)
    for q in levels:
        lvl = np.percentile(dens, 100 * (1 - q))
        if fill_alpha:
            ax.contourf(gx, gy, zz, levels=[lvl, np.nanmax(zz)], colors=[color_fill], alpha=fill_alpha, zorder=1)
        ax.contour(gx, gy, zz, levels=[lvl], colors=[color_line], linewidths=lw, linestyles=[(0, (3, 2))], zorder=3)


# Adapted from the user-selected gmgc2026F plotting reference.
def facets(question: str, nrows: int, ncols: int, size, sharex=False, sharey=False, wspace=0.03, hspace=0.04,
           **kw):
    """分面小多图：同一编码、共享坐标轴的一组小图，作为一张图导出（不是把不同的图拼在一起）。

    用于把只差一个条件的变体（三种成本、两类模型、四个指标……）放进同一张图直接比较，
    避免出一串几乎一样的文件。分面标题用 facet_title 写条件取值，不写 (a)(b)。返回 (fig, axes, pal)。
    """
    pal = theme.use(question)
    w, h = SIZES[size] if isinstance(size, str) else size
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * MM, h * MM), layout="constrained", sharex=sharex,
                             sharey=sharey, squeeze=False, **kw)
    fig.get_layout_engine().set(w_pad=1.2 * MM, h_pad=1.2 * MM, wspace=wspace, hspace=hspace)
    return fig, axes, pal


def facet_title(ax, text, loc="left", **kw):
    """分面小标题：只写该分面的条件取值（如“指数成本”“后训练模型”），字号同图例。"""
    style = dict(fontsize=matplotlib.rcParams["legend.fontsize"], loc=loc, pad=3)
    style.update(kw)
    return ax.set_title(text, **style)




# Adapted from the user-selected gmgc2026F plotting reference.
def end_labels(ax, items, x=None, dx_pt=4.0, min_gap_pt=None, fontsize=None, leader=True, ha="left"):
    """线端直接标注（代替图例）：items = [(y, 文字, 颜色), ...]，写在横坐标 x（默认坐标区右端）右侧。

    纵向按字高自动推开、互不重叠；被推开的标签用细引线连回线端。标签的纵坐标按数据坐标保存，
    所以请在图里其余元素（图例、括号、色条等）都加完之后最后调用，版面再变化时间距仍按比例保持。
    """
    from matplotlib.transforms import offset_copy
    fig = ax.figure
    fig.canvas.draw()
    fs = fontsize or matplotlib.rcParams["legend.fontsize"]
    gap = (min_gap_pt or fs * 1.3) * fig.dpi / 72
    x = ax.get_xlim()[1] if x is None else x
    if not isinstance(x, (int, float, np.number)):                       # 日期轴：Timestamp / datetime → 数值
        import matplotlib.dates as mdates
        x = float(mdates.date2num(x))
    pts = []
    for y, txt, col in items:
        _, yd = ax.transData.transform((x, y))
        pts.append([yd, yd, txt, col, y])
    pts.sort(key=lambda t: t[0])
    for k in range(1, len(pts)):                                       # 自下而上推开
        pts[k][1] = max(pts[k][1], pts[k - 1][1] + gap)
    hi = ax.get_window_extent().y1
    if pts and pts[-1][1] > hi + gap:                                  # 超出上沿时整体下移
        shift = pts[-1][1] - hi - gap
        for t in pts:
            t[1] -= shift
        for k in range(len(pts) - 2, -1, -1):
            pts[k][1] = min(pts[k][1], pts[k + 1][1] - gap)
    inv = ax.transData.inverted()
    tc = offset_copy(ax.transData, fig=fig, x=dx_pt if ha == "left" else -dx_pt, y=0, units="points")
    out = []
    for yd, yn, txt, col, y in pts:
        y_new = inv.transform((ax.transData.transform((x, y))[0], yn))[1]
        moved = abs(yn - yd) > 1.5 * fig.dpi / 72
        out.append(ax.annotate(
            txt, (x, y), xytext=(x, y_new), textcoords=tc, ha=ha, va="center", fontsize=fs, color=col,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.7, shrinkA=0, shrinkB=1)
            if (leader and moved) else None))
    return out


def bracket(ax, lo, hi, at, text=None, orientation="h", tick=0.015, color=None, fontsize=None, lw=0.7,
            text_offset_pt=2.0, side="above", **kw):
    """区间括号，用于标“估算表”“B1 数值范围”“外推区”等区间。

    orientation='h'：横向括号，横跨数据坐标 lo–hi，位于坐标区高度比例 at（如 1.02 在上沿之上、−0.14 在横轴下方）；
    orientation='v'：纵向括号，纵跨数据坐标 lo–hi，位于坐标区宽度比例 at（如 1.02 在右沿之外）。
    side='above'/'below'（横向）或 'right'/'left'（纵向）决定短竖线朝向与文字位置。
    """
    color = color or (theme.CURRENT.ink if theme.CURRENT else "#333333")
    fs = fontsize or matplotlib.rcParams["legend.fontsize"]
    if orientation == "h":
        tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        t = -tick if side == "above" else tick
        ax.plot([lo, lo, hi, hi], [at + t, at, at, at + t], color=color, lw=lw, transform=tr, clip_on=False,
                solid_capstyle="butt")
        if text:
            xm = np.sqrt(lo * hi) if ax.get_xscale() == "log" else (lo + hi) / 2
            ax.annotate(text, (xm, at), xycoords=tr, xytext=(0, text_offset_pt if side == "above" else -text_offset_pt),
                        textcoords="offset points", ha="center", va="bottom" if side == "above" else "top",
                        fontsize=fs, color=color, annotation_clip=False, **kw)
    else:
        tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        t = -tick if side == "right" else tick
        ax.plot([at + t, at, at, at + t], [lo, lo, hi, hi], color=color, lw=lw, transform=tr, clip_on=False,
                solid_capstyle="butt")
        if text:
            ym = np.sqrt(lo * hi) if ax.get_yscale() == "log" else (lo + hi) / 2
            ax.annotate(text, (at, ym), xycoords=tr, xytext=(text_offset_pt if side == "right" else -text_offset_pt, 0),
                        textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                        fontsize=fs, color=color, annotation_clip=False, **kw)


def text_color_on(rgba, threshold=0.55):
    """格内文字颜色：背景亮度低于阈值用白字，否则用深色字。"""
    r, g, b = mcolors.to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < threshold else (theme.CURRENT.ink if theme.CURRENT else "#333333")
