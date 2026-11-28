"""Draw the README figures (light and dark SVG) from the JSON files in docs/img/data/.

    python tools/readme_figures.py          # needs matplotlib

docs/img/data/neurokit_benchmark.json  <- tools/neurokit_benchmark.py --out ...
docs/img/data/mitdb_fiducial.json      <- tools/mitdb_timing.py --env FIDUCIAL=extremum|adaptive|centroid --out ...
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "docs", "img", "data")
IMG = os.path.join(ROOT, "docs", "img")

THEMES = {
    "light": dict(bg="#ffffff", fg="#191919", muted="#666666", grid="#e6e6e6", s1="#257de0", s2="#f47230"),
    "dark": dict(bg="#0d1117", fg="#efefef", muted="#9c9c9c", grid="#30363d", s1="#338ced", s2="#e76624"),
}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def ms(v, _=None):
    return f"{v / 1000:.3g} s" if v >= 1000 else f"{v:.3g} ms"


def speed(t, name):
    rows = json.load(open(os.path.join(DATA, "neurokit_benchmark.json")))["rows"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    w = 0.36
    for i, r in enumerate(rows):
        nk, fh = r["pipeline_ms"]["neurokit"], r["pipeline_ms"]["fasthrv"]
        ax.bar(i - w / 2 - 0.01, nk, w, color=t["s2"], label="NeuroKit2" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, fh, w, color=t["s1"], label="fast-hrv" if i == 0 else None)
        ax.text(i - w / 2, nk * 1.25, ms(nk), ha="center", color=t["fg"], fontsize=9)
        ax.text(i + w / 2, fh * 1.25, ms(fh), ha="center", color=t["fg"], fontsize=9)
        ax.text(i, nk * 5, f"{nk / fh:.0f}× faster", ha="center", color=t["fg"], fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1e6)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(ms))
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['minutes']:g} min ECG\n({r['beats']} beats)" for r in rows], color=t["fg"])
    leg = ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), facecolor=t["bg"])
    plt.close(fig)


def fiducial(t, name):
    d = json.load(open(os.path.join(DATA, "mitdb_fiducial.json")))["methods"]
    err = {m: {r["record"]: abs(100 * (r["rmssd_live_ms"] / r["rmssd_reference_ms"] - 1)) for r in rows}
           for m, rows in d.items()}
    recs = sorted(err["extremum"], key=lambda k: err["extremum"][k])
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    for y, rec in enumerate(recs):
        a, b = err["extremum"][rec], err["adaptive"][rec]
        ax.plot([a, b], [y, y], color=t["grid"], linewidth=2, zorder=1)
        ax.scatter([a], [y], s=48, color=t["s2"], zorder=2, edgecolors=t["bg"], linewidths=1.5,
                   label="extremum (default)" if y == 0 else None)
        ax.scatter([b], [y], s=48, color=t["s1"], zorder=3, edgecolors=t["bg"], linewidths=1.5,
                   label="Fiducial::ADAPTIVE" if y == 0 else None)
        if a > 5:
            note = f"{a:.1f} % → {b:.1f} %" if abs(a - b) > 0.5 else f"{a:.1f} %, unchanged (ectopy / noise)"
            ax.text(max(a, b) * 1.35, y, note, va="center", color=t["fg"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 2000)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} %"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels(recs, color=t["fg"], fontsize=9)
    ax.set_xlabel("|RMSSD error| vs. annotated beats (log scale)")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    for mode, t in THEMES.items():
        speed(t, f"speed-{mode}.svg")
        fiducial(t, f"fiducial-{mode}.svg")
