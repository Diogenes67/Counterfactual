import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

R = json.load(open("/tmp/demo/results.json"))
S = R["seeds"]
n = len(S)

# Okabe-Ito colorblind-safe
C_ORIG = "#0072B2"   # blue
C_UNL  = "#D55E00"   # vermillion
C_RETR = "#009E73"   # green
GREY   = "#999999"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "savefig.dpi": 300,
})

def ms(key1, key2=None):
    if key2:
        v = [s[key1][key2] for s in S]
    else:
        v = [s[key1] for s in S]
    return np.mean(v), np.std(v)

fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5))

# --- Panel a: behavioural test ---
ax = axes[0]
models = ["M0", "Mu", "Mr"]
labels = ["Original", "Unlearned", "Retrained"]
colors = [C_ORIG, C_UNL, C_RETR]
x = np.arange(3)
w = 0.36
fvals = [(np.mean([s[m]["forget_acc"] for s in S]), np.std([s[m]["forget_acc"] for s in S])) for m in models]
ovals = [(np.mean([s[m]["other_acc"] for s in S]), np.std([s[m]["other_acc"] for s in S])) for m in models]
ax.bar(x - w/2, [v[0] for v in fvals], w, yerr=[v[1] for v in fvals], color=colors, edgecolor="none", capsize=2, error_kw={"lw": 0.7})
ax.bar(x + w/2, [v[0] for v in ovals], w, yerr=[v[1] for v in ovals], color=colors, alpha=0.35, edgecolor="none", capsize=2, error_kw={"lw": 0.7})
for xi, v in zip(x, fvals):
    if v[0] < 0.5: ax.text(xi - w/2, v[0] + 0.025, f"{v[0]:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Behaviour: unlearning 'succeeds'", pad=6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#444444", label="Forgotten class"), Patch(fc="#444444", alpha=0.35, label="Other classes")], frameon=False, loc="center", bbox_to_anchor=(0.62, 0.45), handlelength=1.2)

# --- Panel b: representation probe ---
ax = axes[1]
pvals = [ms("probe_M0"), ms("probe_Mu"), ms("probe_Mr")]
ax.bar(x, [v[0] for v in pvals], 0.5, yerr=[v[1] for v in pvals], color=colors, edgecolor="none", capsize=2, error_kw={"lw": 0.7})
for xi, v in zip(x, pvals):
    ax.text(xi, v[0] + 0.03, f"{v[0]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Probe accuracy, forgotten class")
ax.set_ylim(0, 1.12)
ax.set_title("Representation: class still present", pad=6)

# --- Panel c: relearning savings ---
ax = axes[2]
cu = np.array([s["savings"]["Mu"] for s in S])
cr = np.array([s["savings"]["Mr"] for s in S])
steps = np.arange(cu.shape[1])
for curve, color, label in [(cu, C_UNL, "Approx. unlearned"), (cr, C_RETR, "Retrained without class")]:
    m, sd = curve.mean(0), curve.std(0)
    ax.plot(steps, m, color=color, lw=1.4, label=label)
    ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.18, lw=0)
ax.set_xlabel("Fine-tuning steps on 10 examples")
ax.set_ylabel("Forgotten-class test accuracy")
ax.set_ylim(-0.03, 1.05)
ax.set_title("Relearning savings: it comes back", pad=6)
ax.legend(frameon=False, loc="lower right")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax, letter in zip(axes, "abc"):
    ax.text(-0.22, 1.12, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

plt.tight_layout(w_pad=1.6)
plt.savefig("/tmp/demo/figure1.png", bbox_inches="tight")
plt.savefig("/tmp/demo/figure1.pdf", bbox_inches="tight")
print("saved")

# print stats for the manuscript
print("M0 forget", fvals[0], "other", ovals[0])
print("Mu forget", fvals[1], "other", ovals[1])
print("Mr forget", fvals[2], "other", ovals[2])
print("probe", pvals)
print("Mu savings at step 10:", cu.mean(0)[10], "+-", cu.std(0)[10])
print("Mr savings at step 10:", cr.mean(0)[10], "+-", cr.std(0)[10])
print("steps to >=0.9: Mu", int(np.argmax(cu.mean(0) >= 0.9)), "Mr", int(np.argmax(cr.mean(0) >= 0.9)))
