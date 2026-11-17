
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, to_rgb, LinearSegmentedColormap
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "reports/figures"
DST = SRC / "dark"
DST.mkdir(exist_ok=True)

BG = np.array(to_rgb("#1e1e2e"))
FG = np.array(to_rgb("#cdd6f4"))

ACC = {
    "red": "#f38ba8", "peach": "#fab387", "yellow": "#f9e2af", "green": "#a6e3a1",
    "teal": "#94e2d5", "sky": "#89dceb", "blue": "#89b4fa", "lavender": "#b4befe",
    "mauve": "#cba6f7", "pink": "#f5c2e7", "maroon": "#eba0ac",
}
acc_rgb = np.array([to_rgb(v) for v in ACC.values()])
acc_hsv = rgb_to_hsv(acc_rgb.reshape(1, -1, 3)).reshape(-1, 3)

def convert(rgb):
    hsv = rgb_to_hsv(rgb)
    s, v = hsv[..., 1], hsv[..., 2]
    achrom = BG + (1.0 - v)[..., None] * (FG - BG)
    h = hsv[..., 0]
    d = np.abs(h[..., None] - acc_hsv[None, None, :, 0])
    d = np.minimum(d, 1.0 - d)
    idx = d.argmin(axis=-1)
    accent = acc_rgb[idx]
    w = np.clip((s - 0.08) / 0.30, 0.0, 1.0)[..., None]
    out = w * accent + (1.0 - w) * achrom
    # keep alpha channel if present
    if rgb.shape[-1] == 4:
        a = rgb[..., 3:4]
        out = np.concatenate([out, a], axis=-1)
    return np.clip(out, 0, 1)

names = [p.stem for p in sorted(SRC.glob("*.png")) if p.parent == SRC]
done = []
for name in names:
    if "heatmap" in name:
        continue
    img = plt.imread(SRC / f"{name}.png")
    if img.ndim == 3 and img.shape[-1] == 4:
        a = img[..., 3:4]
        img = img[..., :3] * a + (1 - a)
    if img.dtype != np.float64 and img.dtype != np.float32:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    rgb = img[..., :3].astype(np.float32)
    plt.imsave(DST / f"{name}.png", convert(rgb))
    done.append(name)

# --- dark heatmap, rebuilt from the same CSV the light one uses ---
from importlib import import_module
sys.path.insert(0, str(REPO / "components/mp2-py"))
from csc869.models import MODEL_NAMES, DISPLAY_NAMES
from csc869.curves import SCENARIO_TITLES

import yaml
cfg = yaml.safe_load((REPO / "components/mp2-py/configs/main.yaml").read_text())
df = pd.read_csv(REPO / "results/cv_summary_wide.csv")
models = [m for m in MODEL_NAMES if m in set(df["model"])]
scenarios = list(cfg["scenarios"])

cmap = LinearSegmentedColormap.from_list(
    "mocha", ["#181825", "#313244", "#45475a", "#6c7086", "#89b4fa", "#b4befe"])
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
    "savefig.facecolor": "#1e1e2e", "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4", "xtick.color": "#a6adc8", "ytick.color": "#a6adc8",
    "axes.edgecolor": "#6c7086", "font.size": 8, "savefig.dpi": 300,
})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, metric in zip(axes, ["f1", "pr_auc"]):
    mat = np.full((len(models), len(scenarios)), np.nan)
    for i, model in enumerate(models):
        for j, scenario in enumerate(scenarios):
            row = df[(df["model"] == model) & (df["scenario"] == scenario)]
            if not row.empty and metric in row:
                mat[i, j] = float(row[metric].iloc[0])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_TITLES.get(s, s) for s in scenarios], rotation=15, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])
    ax.set_title(metric.replace("_", "-").upper(), color="#cdd6f4")
    ax.grid(False)
    for i in range(len(models)):
        for j in range(len(scenarios)):
            if not np.isnan(mat[i, j]):
                t = float(mat[i, j])
                col = "#1e1e2e" if t > 0.62 else "#cdd6f4"
                ax.text(j, i, f"{t:.3f}", ha="center", va="center", color=col, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#a6adc8")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="#a6adc8")
    cb.outline.set_edgecolor("#6c7086")
fig.suptitle("Mean 5-fold CV score by model and sampling scenario", fontsize=12, color="#cdd6f4")
fig.tight_layout()
fig.savefig(DST / "cv_metric_heatmap.png", bbox_inches="tight", facecolor="#1e1e2e")
plt.close(fig)
done.append("cv_metric_heatmap")
print("wrote", len(done), "dark figures:", ", ".join(sorted(done)))
