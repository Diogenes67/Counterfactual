"""Fig. 1 of 'Counterfactual testing distinguishes machine unlearning from retraining'.

Run from the repository root:  python code/figures/make_fig1.py
Reads results/ and writes figures/fig1.png and figures/fig1.pdf.

a  design schematic
b  audit state space: forget log-probability per token (100 items) against mean rank of the
   truth among four perturbed alternatives, for the original, the ten retrained references,
   NegGrad+ (three seeds), NegGrad+ after NF4, and the oracle edits at 300 and 3,000 steps
c  NegGrad+ recovery on matched item sets: training channels at their final step against the
   40-item baseline, round-to-nearest quantisation against the 40-item baseline, NF4 and
   paraphrase against the 100-item baseline
"""
import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

R = "results"
L = lambda f: json.load(open(f))

# ------------------------------------------------------------------ data
refs = [L(f) for f in sorted(glob.glob(f"{R}/v3_v4/15b_lora_s[0-4]_retain90.json"))
        + sorted(glob.glob(f"{R}/v5/15b_lora_s[5-9]_retain90.json"))]
orig = [L(f"{R}/v3_v4/15b_lora_s{s}_full.json") for s in "012"]
ng = [L(f"{R}/v3_v4/15b_lora_s{s}_unlearn_neggrad.json") for s in "012"]
nf4 = [L(f"{R}/v5/15b_lora_s{s}_nf4_unlearn_neggrad.json") for s in "012"]
or300 = [L(f"{R}/v8/15b_lora_s{s}_oracle_kd.json") for s in "012"]
or3000 = [L(f"{R}/v9/15b_lora_s{s}_oraclex.json") for s in "012"]
b7 = [L(f"{R}/v12/7b_lora_s{s}_unlearn_neggrad.json") for s in "012"]
full = [L(f"{R}/v3_v4/15b_full2_s0_unlearn_neggrad.json")] + \
       [L(f"{R}/v6/15b_full2_s{s}_unlearn_neggrad.json") for s in "12"]

def xy(rows):
    return np.array([r["forget_logprob"] for r in rows]), np.array([r["interdict_rank"] for r in rows])

CHANNELS = ["benign FT", "5-shot", "4-bit", "NF4", "6-bit", "8-bit", "paraphr."]

def deltas(d, nf=None):
    b40 = np.mean(d["forget_logprob_items"][:40]); b100 = d["forget_logprob"]
    out = {"benign FT": d["relearn_benign"][-1] - d["relearn_benign"][0],
           "5-shot": d["savings"][-1] - d["savings"][0],
           "4-bit": d["quant"]["4"]["forget_logprob"] - b40,
           "6-bit": d["quant"]["6"]["forget_logprob"] - b40,
           "8-bit": d["quant"]["8"]["forget_logprob"] - b40,
           "paraphr.": d["paraphrase_logprob"] - b100}
    if nf is not None:
        out["NF4"] = nf["forget_logprob"] - b100
    return out

d15 = [deltas(d, n) for d, n in zip(ng, nf4)]
d7 = [deltas(d) for d in b7]
dfull = [deltas(d) for d in full]

def ci95(v):
    v = np.asarray(v)
    return stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)

# ------------------------------------------------------------------ style
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.linewidth": 0.7, "savefig.dpi": 300,
                     "legend.frameon": False})
C = {"ref": "#009E73", "orig": "#0072B2", "ng": "#E69F00", "nf4": "#B02A2A",
     "or300": "#F1B6D2", "or3000": "#C4699E", "grey": "#8A8A8A", "full": "#E69F00",
     "explor": "#F5DFA6"}

fig = plt.figure(figsize=(9.54, 3.45))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.35, 1.0], left=0.005, right=0.985,
                      top=0.91, bottom=0.30, wspace=0.32)
axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, lab, dx in [(axb, "b", -0.16), (axc, "c", -0.20)]:
    ax.text(dx, 1.07, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# ------------------------------------------------------------------ a: schematic
axa.remove()
axa = fig.add_axes([0.005, 0.30, 0.255, 0.62])
fig.text(0.008, 0.955, "a", fontsize=13, fontweight="bold", va="top")
axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.axis("off")
def box(x, y, w, h, text, ec, bold=False, lw=1.2, fs=7.4):
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                 fc="white", ec=ec, lw=lw))
    axa.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal", linespacing=1.15)
LX, LW = 0.2, 5.0          # left column
RX, RW = 6.6, 3.3          # right column
CX = RX + RW / 2
box(LX, 7.3, LW, 2.0, "Training data\n(includes forget set)", "#555555")
box(RX, 7.3, RW, 2.0, "Original\nmodel", C["orig"])
box(RX, 4.1, RW, 2.0, "Unlearned\nmodel", C["ng"], bold=True)
axa.add_patch(Ellipse((CX, 1.55), RW, 3.1, fc="#DDF3EA", ec=C["ref"], lw=1.2))
axa.text(CX, 1.55, "Retrained\nreference\ndistribution\n(25 seeds)", ha="center", va="center",
         fontsize=7.4, linespacing=1.15)
box(LX, 0.55, LW, 2.0, "Training data\nwithout forget set", "#555555")
arr = dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", color="#777777", lw=1.1)
axa.annotate("", (RX - 0.05, 8.3), (LX + LW + 0.05, 8.3), arrowprops=arr)
axa.text((LX + LW + RX) / 2, 8.6, "train", ha="center", va="bottom", fontsize=6.2, style="italic", color="#666666")
axa.annotate("", (CX, 6.15), (CX, 7.25), arrowprops=arr)
axa.text(CX - 0.25, 6.7, "unlearning", ha="right", va="center", fontsize=6.8, style="italic", color="#666666")
axa.annotate("", (RX - 0.05, 1.55), (LX + LW + 0.05, 1.55), arrowprops=arr)
axa.text((LX + LW + RX) / 2, 1.85, "retrain", ha="center", va="bottom", fontsize=6.2, style="italic", color="#666666")
axa.annotate("", (CX, 3.15), (CX, 4.05),
             arrowprops=dict(arrowstyle="<|-|>,head_width=0.25,head_length=0.5", color="black",
                             lw=1.2, linestyle=(0, (3, 2))))
axa.text(CX - 0.45, 3.6, "counterfactual test:\nsame distribution?", ha="right", va="center",
         fontsize=6.8, style="italic", linespacing=1.1)

# ------------------------------------------------------------------ b: state space
ax = axb
rx, ry = xy(refs)
ax.add_patch(Ellipse((rx.mean(), ry.mean()), 0.75, 0.32, fc="#DDF3EA", ec="none", zorder=1))
ax.scatter(rx, ry, s=55, c=C["ref"], zorder=3, label="Retrained references (n=10)")
ox, oy = xy(orig); ax.scatter(ox, oy, s=70, c=C["orig"], zorder=3, label="Original")
nx, ny = xy(ng); ax.scatter(nx, ny, s=70, c=C["ng"], zorder=4, label="NegGrad+")
qx, qy = xy(nf4); ax.scatter(qx, qy, s=65, c=C["nf4"], marker="s", zorder=4, label="NegGrad+ after NF4")
ax3, ay3 = xy(or300); ax.scatter(ax3, ay3, s=75, c=C["or300"], marker="D", zorder=3,
                                 label="Oracle edit, 300 steps")
bx3, by3 = xy(or3000); ax.scatter(bx3, by3, s=75, c=C["or3000"], marker="D", zorder=3,
                                  label="Oracle edit, 3,000 steps")

# arrows: the argument
ax.add_patch(FancyArrowPatch((ox.mean() - 0.25, oy.mean() + 0.02), (nx.mean() + 0.35, ny.mean() - 0.05),
                             connectionstyle="arc3,rad=0.28", arrowstyle="-|>,head_width=3,head_length=6",
                             color=C["grey"], lw=1.6, zorder=2))
ax.text(-3.3, 0.72, "unlearning", color=C["grey"], fontsize=8.5, rotation=-30, ha="center", va="center")
ax.add_patch(FancyArrowPatch((nx.mean() + 0.35, ny.mean() - 0.10), (qx.mean() - 0.25, qy.mean() + 0.06),
                             connectionstyle="arc3,rad=-0.25", arrowstyle="-|>,head_width=3,head_length=6",
                             color=C["nf4"], lw=1.6, zorder=2))
ax.text(-7.25, 0.50, "NF4: content\nreturns, verdict\nreverses", color=C["nf4"], fontsize=8.2,
        ha="left", va="center", linespacing=1.05)
ax.add_patch(FancyArrowPatch((ax3.mean() - 0.12, ay3.mean() + 0.04), (bx3.mean() - 0.05, by3.mean() - 0.05),
                             connectionstyle="arc3,rad=-0.35", arrowstyle="-|>,head_width=3,head_length=6",
                             color=C["or3000"], lw=1.4, zorder=2))
ax.text(-2.25, 0.13, "oracle: closer\nwith budget", color=C["or3000"], fontsize=8.2, ha="right",
        va="center", linespacing=1.05)

ax.set_xlim(-7.35, 0.25); ax.set_ylim(-0.04, 0.98)
ax.set_xlabel("Forget log-prob./token (content held)")
ax.set_ylabel("Mean rank of truth (audit statistic)")
ax.set_title("The audit's state space", loc="left", pad=6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, columnspacing=1.2,
          handletextpad=0.4, borderaxespad=0.0, labelspacing=0.35, markerscale=0.9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ------------------------------------------------------------------ c: recovery
ax = axc
x = np.arange(len(CHANNELS))
means = [np.mean([d[c] for d in d15]) for c in CHANNELS]
cis = [ci95([d[c] for d in d15]) for c in CHANNELS]
colors = [C["ng"] if i < 3 else C["explor"] for i in range(len(CHANNELS))]
ax.bar(x, means, 0.62, color=colors, zorder=2)
ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="#333333", elinewidth=1.1, capsize=2.5, zorder=4)
for i, c in enumerate(CHANNELS):
    ax.scatter(np.full(3, x[i]) + np.linspace(-0.05, 0.05, 3), [d[c] for d in d15], s=9,
               c="#333333", zorder=5)
for rows, marker, size in [(d7, "o", 52), (dfull, "D", 44)]:
    xs = [x[i] for i, c in enumerate(CHANNELS) if c in rows[0]]
    ys = [np.mean([d[c] for d in rows]) for c in CHANNELS if c in rows[0]]
    ax.scatter(xs, ys, s=size, marker=marker, facecolors="white", edgecolors="#222222",
               linewidths=1.1, zorder=6)
ax.axvline(2.5, color="#BBBBBB", lw=0.8, ls=(0, (1.5, 2)), zorder=1)
ax.text(1.0, 5.02, "prespecified", ha="center", fontsize=8, color="#777777")
ax.text(4.9, 5.02, "exploratory", ha="center", fontsize=8, color="#777777")
ax.set_xticks(x); ax.set_xticklabels(CHANNELS, rotation=35, ha="right", rotation_mode="anchor")
ax.set_ylim(-0.15, 5.35); ax.set_yticks(range(0, 6))
ax.set_ylabel("Forget log-prob. regained (nats)")
ax.set_title("NegGrad+ recovery, matched sets", loc="left", pad=6)
ax.axhline(0, color="black", lw=0.7, zorder=3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig("figures/fig1.png"); fig.savefig("figures/fig1.pdf")
print("wrote figures/fig1.png and figures/fig1.pdf")
for c, m, h in zip(CHANNELS, means, cis):
    print(f"  1.5B {c:10s} {m:+.2f} ± {h:.2f}")
