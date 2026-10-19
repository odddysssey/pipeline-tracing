"""
pipe_viz.py - визуализация результатов pipe_v2.

Дашборд "до / после" для одной трассы:
  ряд 1: 3D-трасса до и после. Цвет = загрузка (1.0 = допуск, красное - превышение),
         пунктир = деформация от температуры (масштаб увеличен), звезда = самое нагруженное место.
  ряд 2: профиль загрузки вдоль трубы и профиль прогиба под весом.
  ряд 3: вертикальные нагрузки на опоры и таблица показателей.

Запуск:  python pipe_viz.py   -> сохраняет PNG рядом с файлом.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np

import pipe_v2 as P

if not os.environ.get("DISPLAY") and os.name != "nt" and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")                     # без экрана - только сохраняем файл
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

KIND_STYLE = {"anchor": ("red", "s", "Анкер"),
              "guide": ("green", "^", "Направляющая"),
              "rest": ("darkorange", "v", "Скользящая")}


def _bounds(evs):
    pts = []
    for ev in evs:
        for e in ev.viz["elements"]:
            pts += [e["c1"], e["c2"]]
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


def _def_scale(evs, extent):
    dmax = max(np.linalg.norm(e[k]) for ev in evs for e in ev.viz["elements"] for k in ("d1", "d2"))
    return float(np.clip(0.06 * extent / dmax, 1.0, 2000.0)) if dmax > 0 else 1.0


def _draw_3d(ax, ev, title, norm, cmap, scale, lo, hi):
    els = ev.viz["elements"]
    segs = [[e["c1"], e["c2"]] for e in els]
    vals = np.array([max(e["uL"], e["uE"]) for e in els])
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=5)
    lc.set_array(vals)
    ax.add_collection3d(lc)
    ax.add_collection3d(Line3DCollection(
        [[e["c1"] + scale * e["d1"], e["c2"] + scale * e["d2"]] for e in els],
        colors="k", linestyles="--", linewidths=1.0, alpha=0.6))
    for kind, (color, marker, _) in KIND_STYLE.items():
        pts = np.array([s["coords"] for s in ev.viz["supports"] if s["kind"] == kind])
        if len(pts):
            ax.scatter(*pts.T, color=color, marker=marker, s=55, depthshade=False, zorder=5)
    i = int(vals.argmax())
    mid = (els[i]["c1"] + els[i]["c2"]) / 2
    ax.scatter(*mid, marker="*", s=260, color="black", depthshade=False, zorder=6)
    ax.text(*(mid + np.array([0, 0, 0.06 * (hi - lo).max()])),
            f"max {vals[i]:.2f}", fontsize=10, fontweight="bold", ha="center")
    pad = 0.05 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    rng = np.maximum((hi - lo) + 2 * pad, 0.30 * (hi - lo).max())      # оси растянуты для читаемости
    ax.set_box_aspect(rng, zoom=1.0)
    ax.view_init(elev=26, azim=-64)
    ax.set_xlabel("X, м"); ax.set_ylabel("Y, м"); ax.set_zlabel("Z, м")
    ax.set_title(title, fontsize=12)


def _draw_2d(ax, ev, ia, ib, names, norm, cmap, scale, lo, hi):
    """Проекция трассы на плоскость (оси ia, ib): план (X-Y) или вид сбоку (X-Z)."""
    els = ev.viz["elements"]
    vals = np.array([max(e["uL"], e["uE"]) for e in els])
    lc = LineCollection([[(e["c1"][ia], e["c1"][ib]), (e["c2"][ia], e["c2"][ib])] for e in els],
                        cmap=cmap, norm=norm, linewidths=5, zorder=3)
    lc.set_array(vals)
    ax.add_collection(lc)
    ax.add_collection(LineCollection(
        [[((e["c1"] + scale * e["d1"])[ia], (e["c1"] + scale * e["d1"])[ib]),
          ((e["c2"] + scale * e["d2"])[ia], (e["c2"] + scale * e["d2"])[ib])] for e in els],
        colors="k", linestyles="--", linewidths=1.0, alpha=0.6, zorder=2))
    for kind, (color, marker, _) in KIND_STYLE.items():
        pts = np.array([s["coords"] for s in ev.viz["supports"] if s["kind"] == kind])
        if len(pts):
            ax.scatter(pts[:, ia], pts[:, ib], color=color, marker=marker, s=45, zorder=5)
    i = int(vals.argmax())
    mid = (els[i]["c1"] + els[i]["c2"]) / 2
    ax.scatter(mid[ia], mid[ib], marker="*", s=240, color="black", zorder=6)
    pad = 0.08 * max((hi - lo).max(), 1e-9)
    ax.set_xlim(lo[ia] - pad, hi[ia] + pad)
    ax.set_ylim(lo[ib] - 0.35 * pad - 0.3, hi[ib] + 0.35 * pad + 0.3)
    ax.set_xlabel(f"{names[0]}, м"); ax.set_ylabel(f"{names[1]}, м")
    ax.grid(alpha=0.3)


def _profile(ax, evs_labels):
    for ev, label, color in evs_labels:
        els = ev.viz["elements"]
        x = np.array([(e["s0"] + e["s1"]) / 2 for e in els])
        uE = np.array([e["uE"] for e in els])
        uL = np.array([e["uL"] for e in els])
        ax.plot(x, uE, color=color, lw=2, label=f"{label}: температурное S_E/S_A")
        ax.plot(x, uL, color=color, lw=1, ls="--", alpha=0.7, label=f"{label}: длительное S_L/Sh")
        ax.fill_between(x, 1, uE, where=uE > 1, color="red", alpha=0.25)
        loop = [e for e in els if e["loop"]]
        if loop:
            ax.axvspan(min(e["s0"] for e in loop), max(e["s1"] for e in loop),
                       color="tab:blue", alpha=0.10)
            ax.text(min(e["s0"] for e in loop), 0.02, " компенсатор", color="tab:blue", fontsize=9)
    ax.axhline(1.0, color="red", lw=1.2)
    ax.text(0.5, 1.02, "допуск", color="red", fontsize=9)
    ax.set_xlabel("расстояние вдоль трубы от первого анкера, м")
    ax.set_ylabel("загрузка (1.0 = допуск)")
    ax.set_title("Загрузка вдоль трассы (красная заливка - превышение)")
    top = max(1.3, 1.08 * max(e["uE"] for lab in evs_labels for e in lab[0].viz["elements"]))
    ax.set_ylim(0, top)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def _sag(ax, evs_labels, limit):
    for ev, label, color in evs_labels:
        els = ev.viz["elements"]
        s = [els[0]["s0"]] + [e["s1"] for e in els]
        z = [-els[0]["z1"] * 1000] + [-e["z2"] * 1000 for e in els]
        ax.plot(s, z, color=color, lw=2, label=label)
    ax.axhline(limit * 1000, color="red", lw=1.2)
    ax.text(0.5, limit * 1000 * 1.03, "допуск прогиба", color="red", fontsize=9)
    ax.set_xlabel("расстояние вдоль трубы, м")
    ax.set_ylabel("прогиб под весом, мм")
    ax.set_title("Прогиб вдоль трассы")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def _supports(ax, ev_b, ev_a):
    for ev, off, color, label in ((ev_b, -0.35, "0.65", "до"), (ev_a, 0.35, "tab:blue", "после")):
        sup = [s for s in ev.viz["supports"] if s["kind"] != "anchor"]
        ax.bar([s["s"] + off for s in sup], [s["Fz"] / 1e3 for s in sup], width=0.7,
               color=color, label=f"опоры {label}")
    ax.set_xlabel("расстояние вдоль трубы, м")
    ax.set_ylabel("вертикальная нагрузка, кН")
    ax.set_title("Нагрузка от веса на опоры (без анкеров)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)


def _table(ax, sb, sa, lim):
    def anchor_rows(key, scale, fmt):
        return [fmt.format(a[key] / scale) for a in sb["anchors"]], [fmt.format(a[key] / scale) for a in sa["anchors"]]
    fb, fa = anchor_rows("therm_F", 1e3, "{:.1f}")
    mb, ma = anchor_rows("therm_M", 1e3, "{:.1f}")
    rows = [
        ("Статус", "ПРОХОДИТ" if sb["ok"] else "НЕ ПРОХОДИТ", "ПРОХОДИТ" if sa["ok"] else "НЕ ПРОХОДИТ"),
        ("Макс. загрузка", f"{sb['max_util']:.2f}", f"{sa['max_util']:.2f}"),
        (f"S_E макс, МПа (допуск {lim.SA/1e6:.0f})", f"{sb['SE']/1e6:.0f}", f"{sa['SE']/1e6:.0f}"),
        (f"S_L макс, МПа (допуск {lim.Sh/1e6:.0f})", f"{sb['SL']/1e6:.0f}", f"{sa['SL']/1e6:.0f}"),
        (f"Прогиб, мм (допуск {lim.sag*1000:.0f})", f"{sb['sag']*1000:.1f}", f"{sa['sag']*1000:.1f}"),
        ("Сила на анкерах от Т, кН", " / ".join(fb), " / ".join(fa)),
        ("Момент на анкерах от Т, кН*м", " / ".join(mb), " / ".join(ma)),
        ("Направляющих / скользящих", f"{sb['guides']} / {sb['rests']}", f"{sa['guides']} / {sa['rests']}"),
        ("Изгибов (отводов), шт", f"{sb['bends']}", f"{sa['bends']}"),
        ("Компенсаторов, шт", f"{sb['loops']}", f"{sa['loops']}"),
        ("Длина трубы, м", f"{sb['length']:.1f}", f"{sa['length']:.1f}"),
        ("Масса стали, кг", f"{sb['mass']:,.0f}".replace(",", " "), f"{sa['mass']:,.0f}".replace(",", " ")),
    ]
    ax.axis("off")
    t = ax.table(cellText=[list(r) for r in rows], colLabels=["Показатель", "До", "После"],
                 loc="center", cellLoc="center", colWidths=[0.52, 0.24, 0.24])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.55)
    for j, ok in ((1, sb["ok"]), (2, sa["ok"])):
        t[(1, j)].set_facecolor("#c8e6c9" if ok else "#ffcdd2")
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e0e0e0")
            cell.set_text_props(fontweight="bold")
    ax.set_title("Сводка", fontsize=12)


def dashboard(res, title, path=None):
    """res - результат P.design(...). Рисует дашборд до/после, сохраняет в path (или показывает)."""
    ev_b, ev_a, lim = res["ev_before"], res["ev_after"], res["lim"]
    sb = P.summary(res["nodes_before"], res["pipes_before"], ev_b)
    sa = P.summary(res["nodes"], res["pipes"], ev_a)
    changed = sa["loops"] > sb["loops"]
    lo, hi = _bounds([ev_b, ev_a])
    umax = max(max(max(e["uL"], e["uE"]) for e in ev.viz["elements"]) for ev in (ev_b, ev_a))
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=max(1.3, umax))
    cmap = plt.get_cmap("RdYlGn_r")
    scale = _def_scale([ev_b, ev_a], float((hi - lo).max()))

    fig = plt.figure(figsize=(17, 24))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 0.8, 0.8, 0.85], hspace=0.34, wspace=0.18)
    axes3d = []
    for r, (ev, ttl) in enumerate(((ev_b, f"ДО  (загрузка max {ev_b.max_util:.2f})"),
                                   (ev_a, f"ПОСЛЕ  (загрузка max {ev_a.max_util:.2f})" if changed
                                    else "ПОСЛЕ  (компенсатор не потребовался)"))):
        sub = gs[r, :].subgridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.55, wspace=0.12)
        a3 = fig.add_subplot(sub[:, 0], projection="3d")
        _draw_3d(a3, ev, ttl, norm, cmap, scale, lo, hi)
        pl = fig.add_subplot(sub[0, 1])
        _draw_2d(pl, ev, 0, 1, ("X", "Y"), norm, cmap, scale, lo, hi)
        pl.set_title("вид сверху (план)", fontsize=10)
        el = fig.add_subplot(sub[1, 1])
        _draw_2d(el, ev, 0, 2, ("X", "Z"), norm, cmap, scale, lo, hi)
        el.set_title("вид сбоку", fontsize=10)
        axes3d.append(a3)
    ax1, ax2 = axes3d
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, shrink=0.8,
                      pad=0.02, aspect=45, location="bottom")
    cb.set_label("загрузка (1.0 = допуск)")
    cb.ax.axvline(1.0, color="k", lw=2)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=8, label=l)
               for c, m, l in KIND_STYLE.values()]
    handles += [plt.Line2D([], [], color="k", ls="--", label=f"деформация от Т (x{scale:.0f})"),
                plt.Line2D([], [], color="k", marker="*", ls="", markersize=12, label="самое нагруженное место")]
    ax1.legend(handles=handles, loc="upper left", fontsize=8)

    series = [(ev_b, "до", "tab:red")] + ([(ev_a, "после", "tab:blue")] if changed else [])
    _profile(fig.add_subplot(gs[2, 0]), series)
    _sag(fig.add_subplot(gs[2, 1]), series, lim.sag)
    _supports(fig.add_subplot(gs[3, 0]), ev_b, ev_a)
    _table(fig.add_subplot(gs[3, 1]), sb, sa, lim)

    fig.suptitle(f"{title}   (ΔT = {res['dT']:.0f} К, труба {res['pipes'][0].sec.D*1000:.0f}x"
                 f"{(res['pipes'][0].sec.D-res['pipes'][0].sec.d)*500:.0f} мм)", fontsize=15, y=0.92)
    fig.text(0.5, 0.085, "Оси растянуты для читаемости (на проекциях масштаб осей разный). Допускаемые напряжения условные "
             "(Sh = Sc = 140 МПа), SIF в отводах не учтены.", ha="center", fontsize=9, color="0.4")
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    sec, lim = P.Section(D=0.219, d=0.203), P.Limits()
    cases = [
        ("case1_dashboard.png", "Случай 1: старая трасса",
         [(0, 0, 0), (40, 0, 0), (40, 10, 0), (40, 10, 8)], 100),
        ("case2_dashboard.png", "Случай 2: жёсткая трасса",
         [(0, 0, 0), (60, 0, 0), (60, 3, 0), (60, 3, 4)], 150),
    ]
    for fname, title, pts, dT in cases:
        res = P.design(pts, sec, lim, dT, title)
        dashboard(res, title, os.path.join(out, fname))
        print("сохранено:", fname)
