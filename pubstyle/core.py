from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import string
import colorspacious as cs
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
from matplotlib.collections import QuadMesh, PathCollection
from matplotlib.image import AxesImage
from matplotlib.contour import QuadContourSet, ContourSet
from matplotlib.font_manager import FontProperties
from collections import defaultdict
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Bbox


def format(fig):
    fig.align_xlabels(fig.axes)
    fig.align_ylabels(fig.axes)
    label_subplots(fig)
    pastel_marker_style(fig)
    pastel_hist_style(fig) 
    unclipped_markers(fig)
    style_legend(fig)
    style_and_shift_colorbars(fig)
    round_all_ticks(fig)


def size(journal,height=60):
    CANONICAL_WIDTHS_MM = {
        'nature': 89,
        'nature2': 183,
        'aps': 86,
        'aps2': 172,
        'science': 57,
        'science2': 121,
        'science3': 184,
    }

    # aliases → canonical keys
    aliases = {
        'nat': 'nature',
        'nat2': 'nature2',
        'sci': 'science',
        'sci2': 'science2',
        'sci3': 'science3',
    }

    if isinstance(journal, (int, float)):
        width = journal
    else:
        key = str(journal).lower()
        key = aliases.get(key, key)
        width = CANONICAL_WIDTHS_MM.get(key, CANONICAL_WIDTHS_MM['aps'])

    return mm2in(width,height)

def reset_style():
    mpl.rcdefaults()
    plt.style.use('default')

    if getattr(Axes.errorbar, "_pubstyle_patched", False):
        Axes.errorbar = Axes.errorbar._pubstyle_orig

def set_publication_style():
    from matplotlib.font_manager import fontManager

    pkg_dir = Path(__file__).parent
    reg_path = pkg_dir / "MYRIADPRO-REGULAR.otf"
    bold_path = pkg_dir / "MYRIADPRO-BOLD.otf"

    for p in (reg_path, bold_path):
        if p.exists():
            fontManager.addfont(str(p))

    fam = FontProperties(fname=str(reg_path)).get_name() if reg_path.exists() else "sans-serif"

    style_path = pkg_dir / "pubstyle.mplstyle"
    plt.style.use(style_path)

    mpl.rcParams["font.family"] = [fam, "sans-serif"]
    mpl.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"] 


    if not getattr(Axes.errorbar, "_pubstyle_patched", False):
        _orig = Axes.errorbar

        def _custom(self, *args, **kwargs):
            kwargs.setdefault("fmt", 'o')
            kwargs.setdefault("linestyle", 'none')
            return _orig(self, *args, **kwargs)

        _custom._pubstyle_patched = True
        _custom._pubstyle_orig = _orig
        Axes.errorbar = _custom

def style_legend(fig):

    rebuild_legends(fig)

    for ax in fig.axes:
        for leg in [c for c in ax.get_children() if isinstance(c, Legend)]:
            handles = leg.legend_handles
            texts = leg.get_texts()
            for h, txt in zip(handles, texts):
                if isinstance(h, Line2D):
                    # works for normal lines and errorbar line proxies
                    txt.set_color(h.get_color())
                elif isinstance(h, PathCollection):
                    # scatter markers
                    fc = h.get_facecolors()
                    if len(fc) > 0:
                        txt.set_color(fc[0])
                elif isinstance(h, LineCollection):
                    # errorbar caps are line collections
                    cols = h.get_colors()
                    if len(cols) > 0:
                        txt.set_color(cols[0])

def rebuild_legends(fig):
    for ax in fig.axes:
        old = ax.get_legend()
        if not isinstance(old, Legend):
            continue

        # Preserve only safe attributes
        labels_order = [t.get_text() for t in old.get_texts()]
        loc = getattr(old, "_loc", "best")
        ncol = getattr(old, "_ncols", 1)
        frameon = old.get_frame_on()
        title = old.get_title().get_text()
        fontsize = old.get_texts()[0].get_fontsize() if old.get_texts() else None
        title_fs = old.get_title().get_fontsize()

        # current handles reflect updated styling
        src_handles, src_labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, lab in zip(src_handles, src_labels):
            by_label.setdefault(lab, []).append(h)

        new_handles, new_labels = [], []
        for lab in labels_order:
            lst = by_label.get(lab, [])
            if lst:
                new_handles.append(lst.pop(0))
                new_labels.append(lab)

        # remove and recreate
        old.remove()
        leg = ax.legend(
            new_handles, new_labels,
            loc=loc, ncol=ncol, frameon=frameon, title=title,
            fontsize=fontsize, title_fontsize=title_fs
        )

    fig.canvas.draw_idle()


def round_all_ticks(fig):
    fig.canvas.draw()

    try:
        from matplotlib._enums import CapStyle
        _ROUND_CAP = CapStyle.round
    except Exception:
        _ROUND_CAP = 'round'

    for ax in fig.axes:
        for sp in ax.spines.values():
            try:
                sp.set_capstyle('round')
                sp.set_joinstyle('round')
            except Exception:
                pass
        for axis in (ax.xaxis, ax.yaxis):
            ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
            for t in ticks:
                for ln in (t.tick1line, t.tick2line):
                    try:
                        m = ln._marker  
                        m._capstyle = _ROUND_CAP

                        ln.set_antialiased(True)
                    except Exception:
                        pass

        for cont in ax.containers:
            if isinstance(cont, ErrorbarContainer):
                _, _, barlinecols = cont
                for lc in barlinecols:
                    if isinstance(lc, LineCollection):
                        lc.set_capstyle('round')

def _iter_colorbars(fig):
    """Yield unique Colorbar objects by following mappable back-refs."""
    seen = set()
    for ax in fig.axes:
        for m in list(ax.images) + list(ax.collections):
            cb = getattr(m, "colorbar", None)
            if cb is not None and id(cb) not in seen:
                seen.add(id(cb))
                yield cb

def style_and_shift_colorbars(fig, *, labelpad=1, labelsize=7.5,
                               tickpad=1.5, ticksize=6.5, shift_frac=0.5,
                               ticklength=1.1):
    
    """
    Style colorbars and optionally shift right-side ones closer to their parent.

    shift_frac = 0.5 means reduce the gap (pad) between parent and cbar by half.
    """
    fig.set_constrained_layout(False)

    fig.canvas.draw()  # freeze current positions

    for cb in _iter_colorbars(fig):
        # style
        lbl = cb.ax.get_ylabel()
        if lbl:
            cb.set_label(lbl, labelpad=labelpad, fontsize=labelsize)
        cb.ax.tick_params(pad=tickpad, labelsize=ticksize, length=ticklength)

        cb.minorticks_off()
        cb.ax.set_axisbelow(False)

        # detach from layout
        cb.ax.set_in_layout(False)

        # positions
        p = cb.mappable.axes.get_position()
        c = cb.ax.get_position()


        def shift_bbox(c, p, side):
            if side == "right":
                gap = c.x0 - p.x1
                new_x0 = p.x1 + gap * shift_frac
                return Bbox.from_bounds(new_x0, c.y0, c.width, c.height)
            elif side == "left":
                gap = p.x0 - c.x1
                new_x1 = p.x0 - gap * shift_frac
                return Bbox.from_bounds(new_x1 - c.width, c.y0, c.width, c.height)
            elif side == "above":
                gap = c.y0 - p.y1
                new_y0 = p.y1 + gap * shift_frac
                return Bbox.from_bounds(c.x0, new_y0, c.width, c.height)
            elif side == "below":
                gap = p.y0 - c.y1
                new_y1 = p.y0 - gap * shift_frac
                return Bbox.from_bounds(c.x0, new_y1 - c.height, c.width, c.height)
            else:
                return c  # unchanged

        if c.height >= c.width and c.x0 >= p.x1 - 1e-6:
            side = "right"
        elif c.height >= c.width and c.x1 <= p.x0 + 1e-6:
            side = "left"
        elif c.width > c.height and c.y0 >= p.y1 - 1e-6:
            side = "above"
        elif c.width > c.height and c.y1 <= p.y0 + 1e-6:
            side = "below"
        else:
            side = None

        cb.ax.set_position(shift_bbox(c, p, side))


def classify_axes(fig):
    """
    Returns:
      data_with_cb : set of Axes that are data axes AND have ≥1 colorbar
      data_no_cb   : set of Axes that are data axes AND have no colorbar
      cbar_axes    : set of Axes that host colorbars
      parent_to_cax: dict mapping parent Axes -> list of colorbar Axes
    """
    cbar_axes = set()
    parent_to_cax = defaultdict(list)

    for ax in fig.axes:
        for m in list(ax.images) + list(ax.collections):
            cb = getattr(m, "colorbar", None)
            if cb is not None:
                cax = cb.ax
                cbar_axes.add(cax)
                parent_to_cax[m.axes].append(cax)

    data_axes = {ax for ax in fig.axes if ax not in cbar_axes}
    data_with_cb = set(parent_to_cax.keys())
    data_no_cb = data_axes - data_with_cb

    return data_with_cb, data_no_cb, cbar_axes, parent_to_cax



def mm2in(*tupl):
    inch = 25.4
    if len(tupl) == 1 and isinstance(tupl[0], tuple):
        return tuple(v/inch for v in tupl[0])
    return tuple(v/inch for v in tupl)

def get_colors(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    if isinstance(cmap, mcolors.ListedColormap) and cmap.N < 256:
        if n <= cmap.N:
            return [cmap(i) for i in range(n)]
        idx = np.linspace(0, cmap.N-1, n).astype(int)
        return [cmap(i) for i in idx]
    return [cmap(v) for v in np.linspace(0, 1, n)]

def label_subplots(fig, offset=0, pad_pt=(0, 0), upper=False,
                   top_row_shift=(0, -6.5), ignore_colorbars=True, ignore_twins=True):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    letters = string.ascii_uppercase if upper else string.ascii_lowercase

    # resolve bold by family + weight so PDF keeps text editable
    family = mpl.rcParams['font.family']
    if isinstance(family, (list, tuple)): family = family[0]
    bold_fp = FontProperties(family=family, weight='bold')

    axes = list(fig.axes)

    # drop colorbar axes if requested
    if ignore_colorbars:
        _, _, cbar_ax, _ = classify_axes(fig)
        axes = [ax for ax in axes if ax not in cbar_ax]

    # keep only the first axes per position group (twinx/twiny share the same bounds)
    if ignore_twins:
        def _bbox_key(ax):
            x0, y0, w, h = ax.get_position().bounds
            return tuple(round(v, 6) for v in (x0, y0, w, h))

        seen = set()
        primaries = []
        for ax in axes:  # fig.axes order ≈ creation order → first is “primary”
            key = _bbox_key(ax)
            if key in seen:
                continue
            seen.add(key)
            primaries.append(ax)
        axes = primaries

    # compute top row from the kept axes only
    top_y = max(ax.get_tightbbox(r).y1 for ax in axes) if axes else 0

    if len(axes)==1:
        return
    
    # place labels
    for i, ax in enumerate(axes):
        bbox = ax.get_tightbbox(r)
        dx_pts, dy_pts = pad_pt
        if abs(bbox.y1 - top_y) < 10:
            dx_pts += top_row_shift[0]; dy_pts += top_row_shift[1]
        dx = dx_pts * fig.dpi / 72.0; dy = dy_pts * fig.dpi / 72.0
        x_fig, y_fig = inv.transform((bbox.x0 + dx, bbox.y1 + dy))
        x_fig = max(x_fig, 0); y_fig = max(y_fig, 0)
        ax.figure.text(x_fig, y_fig, letters[(i + offset) % 26],
                       transform=fig.transFigure, va='bottom', ha='left',
                       fontproperties=bold_fp)

    # place labels
    for i, ax in enumerate(axes):
        bbox = ax.get_tightbbox(r)
        dx_pts, dy_pts = pad_pt
        if abs(bbox.y1 - top_y) < 10:
            dx_pts += top_row_shift[0]; dy_pts += top_row_shift[1]
        dx = dx_pts * fig.dpi / 72.0; dy = dy_pts * fig.dpi / 72.0
        x_fig, y_fig = inv.transform((bbox.x0 + dx, bbox.y1 + dy))
        x_fig = max(x_fig, 0); y_fig = max(y_fig, 0)
        ax.figure.text(x_fig, y_fig, letters[(i + offset) % 26],
                       transform=fig.transFigure, va='bottom', ha='left',
                       fontproperties=bold_fp)

def pastel_marker_style(fig, chroma_scale=1, lightness_boost=33, edge_width=0.5):
    for ax in fig.axes:
        for line in ax.get_lines():
            marker = line.get_marker()
            if marker in (None, 'None'): continue
            rgb = np.array(mcolors.to_rgb(line.get_color()))
            lab = cs.cspace_convert(rgb, 'sRGB1', 'CIELab')
            L, a, b = lab
            C = np.sqrt(a**2 + b**2)
            h = np.degrees(np.arctan2(b, a))
            edge_rgb = rgb
            Lf = min(100, L + lightness_boost)
            Cf = C * chroma_scale
            af = Cf * np.cos(np.radians(h))
            bf = Cf * np.sin(np.radians(h))
            face_rgb = cs.cspace_convert([Lf, af, bf], 'CIELab', 'sRGB1')
            face_rgb = np.clip(face_rgb, 0, 1)
            line.set_markerfacecolor(face_rgb)
            line.set_markeredgecolor(edge_rgb)
            line.set_markeredgewidth(edge_width)

def pastel_hist_style(fig, chroma_scale=1.0, lightness_boost=33, edge_width=0.8):
    """
    Lighten histogram faces in CIELab while keeping edges at the base color.
    Works for histtype='bar', 'barstacked', and 'stepfilled'. 'step' is skipped.
    """

    def _rgb_from_artist(a):
        # prefer facecolor, else edgecolor; handle tuple or Nx4 arrays
        fc = a.get_facecolor() if hasattr(a, "get_facecolor") else None
        if fc is None or (hasattr(fc, "__len__") and len(fc) == 0):
            ec = a.get_edgecolor() if hasattr(a, "get_edgecolor") else None
            fc = ec
        arr = np.array(fc, dtype=float)
        if arr.ndim > 1: arr = arr[0]
        return np.array(arr[:3], dtype=float)

    for ax in fig.axes:
        # collect histogram patches
        patches = []

        # bar / barstacked histograms live in BarContainer.patches
        for cont in getattr(ax, "containers", []):
            ps = getattr(cont, "patches", None)
            if ps: patches.extend(ps)

        # stepfilled hist adds Polygon patches directly to ax.patches
        for p in ax.patches:
            if isinstance(p, Rectangle):
                patches.append(p)
            elif isinstance(p, Polygon) and p.get_fill() and len(p.get_xy()) > 4:
                patches.append(p)

        # dedupe
        if not patches: 
            continue
        patches = list(dict.fromkeys(patches))

        # apply pastel styling
        for p in patches:
            try:
                rgb = _rgb_from_artist(p)
                # CIELab lighten + optional chroma scaling
                L, a, b = cs.cspace_convert(rgb, 'sRGB1', 'CIELab')
                C = np.hypot(a, b)
                h = np.degrees(np.arctan2(b, a))
                edge_rgb = rgb
                Lf = min(100.0, float(L) + lightness_boost)
                Cf = C * chroma_scale
                af = Cf * np.cos(np.radians(h))
                bf = Cf * np.sin(np.radians(h))
                face_rgb = cs.cspace_convert([Lf, af, bf], 'CIELab', 'sRGB1')
                face_rgb = np.clip(face_rgb, 0, 1)

                p.set_facecolor(face_rgb)
                p.set_edgecolor(edge_rgb)
                if hasattr(p, "set_linewidth"):
                    p.set_linewidth(edge_width)
            except Exception:
                continue

def unclipped_markers(fig, add_margin=False):
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_marker() not in [None, 'None']:
                line.set_clip_on(False)

        has_2d = any(
            isinstance(obj, (QuadMesh, AxesImage, QuadContourSet, ContourSet, PathCollection))
            for obj in (ax.collections + ax.images)
        )

        if not has_2d:
            for sp in ax.spines.values():
                sp.set_zorder(0)

        if add_margin:
            ax.margins(mpl.rcParamsDefault['axes.xmargin'],
                       mpl.rcParamsDefault['axes.ymargin'])
            ax.autoscale(enable=True, axis='both', tight=False)