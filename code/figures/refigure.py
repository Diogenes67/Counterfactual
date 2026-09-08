import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

R = {n: json.load(open(f"{n}.json")) for n in ["full", "retain90", "unlearn_ga", "unlearn_neggrad", "unlearn_idk"]}
fo, fc = R["full"]["forget_logprob"], R["retain90"]["forget_logprob"]

mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.7, "savefig.dpi": 300})
COL = {"unlearn_ga": "#D55E00", "unlearn_neggrad": "#E69F00", "unlearn_idk": "#CC79A7",
       "retain90": "#009E73", "full": "#0072B2"}
LBL = {"unlearn_ga": "GA", "unlearn_neggrad": "NegGrad+", "unlearn_idk": "IDK",
       "retain90": "Retrained control", "full": "Original"}

fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.7))

# a: baseline forget logprob and where each recovery channel brings it back to
ax = axes[0]
arms = ["unlearn_ga", "unlearn_neggrad", "unlearn_idk"]
ax.axvline(fo, ls=":", lw=1, color="#0072B2"); ax.text(fo + 0.08, 1.95, "original\n(knows)", fontsize=6.3, color="#0072B2")
ax.axvline(fc, ls=":", lw=1, color="#009E73"); ax.text(fc - 0.12, 2.42, "control\n(never knew)", fontsize=6.3, color="#009E73", ha="right")
chan_marks = [("quant6", "s", "6-bit quantized"), ("relearn", "o", "benign relearn (peak)"), ("savings", "^", "5-shot savings (peak)")]
for yi, a in enumerate(arms):
    r = R[a]; b = r["forget_logprob"]
    ax.plot([b], [yi], marker="|", ms=13, mew=2.2, color=COL[a])
    vals = {"quant6": r["quant"]["6"]["forget_logprob"], "relearn": max(r["relearn_benign"]), "savings": max(r["savings"])}
    for (k, m, _), v in zip(chan_marks, vals.values()):
        ax.plot([v], [yi], marker=m, ms=5, color=COL[a], mfc="white", mew=1.1)
    ax.annotate("", xy=(max(vals.values()), yi), xytext=(b, yi),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=COL[a], alpha=0.6))
ax.set_yticks(range(3)); ax.set_yticklabels([LBL[a] for a in arms])
ax.set_ylim(-0.6, 2.9)
ax.set_xlabel("Forget-set logprob/token (higher = more present)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="|", ms=9, mew=2, ls="", color="#555", label="after unlearning")] +
          [Line2D([], [], marker=m, ls="", ms=5, color="#555", mfc="white", label=l) for _, m, l in chan_marks],
          frameon=False, loc="lower left", handletextpad=0.4)
ax.set_title("What comes back, per channel")

# b: interdict ranks
ax = axes[1]
order = ["full", "unlearn_idk", "unlearn_ga", "retain90", "unlearn_neggrad"]
vals = [R[n]["interdict_rank"] for n in order]
ax.bar(range(len(order)), vals, color=[COL[n] for n in order], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
ax.axhline(R["retain90"]["interdict_rank"], ls="--", lw=0.8, color="#009E73")
ax.text(0.0, R["retain90"]["interdict_rank"] + 0.03, "never-knew anchor", fontsize=6.3, color="#009E73")
ax.set_xticks(range(len(order))); ax.set_xticklabels([LBL[n] for n in order], rotation=25, ha="right")
ax.set_ylabel("Mean rank of true answer\n(0 = ranked first of 5)")
ax.set_title("Interdict signature: both failure directions")

# c: benign relearning curves
ax = axes[2]
steps = np.arange(21) * 5
for n in ["unlearn_neggrad", "unlearn_ga", "unlearn_idk", "retain90", "full"]:
    ax.plot(steps, R[n]["relearn_benign"], color=COL[n], lw=1.3, label=LBL[n])
ax.axhline(fo, ls=":", lw=0.8, color="#0072B2")
ax.set_xlabel("Fine-tuning steps on retain-only data\n(contains no forget-set facts)")
ax.set_ylabel("Forget-set logprob/token")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Benign relearning: it comes back unasked")

for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for ax, letter in zip(axes, "abc"):
    ax.text(-0.22, 1.1, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
plt.tight_layout(w_pad=1.8)
plt.savefig("tofu_audit_v2.png", bbox_inches="tight")
plt.savefig("tofu_audit_v2.pdf", bbox_inches="tight")
print("saved")
