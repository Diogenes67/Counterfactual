"""Fig. 2 of 'Counterfactual testing distinguishes machine unlearning from retraining'.

Run from the repository root:  python code/figures/make_fig2.py
Reads results/ and writes figures/fig2.png and figures/fig2.pdf.

a  targeted rank diagnostic: mean rank of the true answer among four perturbed alternatives
   (Qwen2.5-1.5B LoRA, seeds 0-2; bars mean ± 95% CI; dots seeds; all ten retrained
   references on the reference bar), with 7B seed means (open circles), the Llama arm
   (open triangles: six-reference mean and re-tuned NegGrad+ mean) and the NegGrad+
   checkpoints after NF4 quantisation (open square)
b  layer-wise linear CKA between hidden representations on forget items (mean of three
   seeds; shading, seed range)
c  late-layer activation patching (seeds 0-2): forget log-probability of the unlearned
   model, of the unlearned model with the original's or the retrained reference's layer-24
   residual state substituted, and of the original
"""
import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
L = lambda f: json.load(open(f))
rank = lambda rows: np.array([r["interdict_rank"] for r in rows])

# ------------------------------------------------------------------ data
q = lambda name: [L(f"{R}/v3_v4/15b_lora_s{s}_{name}.json") for s in "012"]
refs10 = [L(f) for f in sorted(glob.glob(f"{R}/v3_v4/15b_lora_s[0-4]_retain90.json"))
          + sorted(glob.glob(f"{R}/v5/15b_lora_s[5-9]_retain90.json"))]
groups = [("Original", rank(q("full")), "#0072B2"),
          ("IDK", rank(q("unlearn_idk")), "#CC79A7"),
          ("NPO", rank(q("unlearn_npo")), "#56B4E9"),
          ("GA", rank(q("unlearn_ga")), "#D55E00"),
          ("Reference", rank(refs10), "#009E73"),
          ("NegGrad+", rank(q("unlearn_neggrad")), "#E69F00")]
ref7 = rank([L(f"{R}/v3_v4/7b_lora_s0_retain90.json")] + [L(f"{R}/v6/7b_lora_s{s}_retain90.json") for s in "12"])
ng7 = rank([L(f"{R}/v12/7b_lora_s{s}_unlearn_neggrad.json") for s in "012"])
refL = rank([L(f) for f in sorted(glob.glob(f"{R}/v3_v4/llama_lora_s[0-2]_retain90.json"))
             + sorted(glob.glob(f"{R}/v5/llama_lora_s[3-5]_retain90.json"))])
ngL = rank([L(f"{R}/v6/llama2_lora_s{s}_unlearn_neggrad.json") for s in "012"])
nf4 = rank([L(f"{R}/v5/15b_lora_s{s}_nf4_unlearn_neggrad.json") for s in "012"])

mech = [L(f"{R}/v7/mech_s{s}.json") for s in "012"]
patch = [L(f"{R}/v10/patch_s{s}.json") for s in "012"]
orig_lp = np.array([L(f"{R}/v3_v4/15b_lora_s{s}_full.json")["forget_logprob"] for s in "012"])

def ci95(v):
    v = np.asarray(v)
    return stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)

# ------------------------------------------------------------------ style
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.linewidth": 0.7, "savefig.dpi": 300,
                     "legend.frameon": False})
fig = plt.figure(figsize=(9.54, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.3, 1.0], left=0.06, right=0.99,
                      top=0.88, bottom=0.27, wspace=0.40)
axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, lab in [(axa, "a"), (axb, "b"), (axc, "c")]:
    ax.text(-0.22, 1.10, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ------------------------------------------------------------------ a: rank diagnostic
ax = axa
x = np.arange(len(groups))
for i, (name, v, col) in enumerate(groups):
    m = v.mean()
    ax.bar(i, m, 0.62, color=col, zorder=2)
    ax.errorbar(i, m, yerr=ci95(v), fmt="none", ecolor="#333333", elinewidth=1.1, capsize=2.5, zorder=4)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, c="#333333", zorder=5)
    if m < 0.02:
        ax.text(i, 0.035, f"{m:.2f}", ha="center", fontsize=7.5, color="#555555")
iR, iN = 4, 5
ax.scatter([iR, iN], [ref7.mean(), ng7.mean()], s=60, marker="o", facecolors="white",
           edgecolors="#222222", linewidths=1.2, zorder=6, label="7B (n=3, mean)")
ax.scatter([iR, iN], [refL.mean(), ngL.mean()], s=70, marker="^", facecolors="white",
           edgecolors="#7B3FA0", linewidths=1.2, zorder=6, label="Llama re-tuned (n=6/3)")
ax.scatter([iN], [nf4.mean()], s=55, marker="s", facecolors="white", edgecolors="#B02A2A",
           linewidths=1.2, zorder=6, label="NegGrad+ after NF4")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], rotation=35, ha="right", rotation_mode="anchor")
ax.set_ylim(-0.15, 1.42); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.axhline(0, color="black", lw=0.7, zorder=3)
ax.set_ylabel("Mean rank of truth (0 = first)")
ax.set_title("Targeted rank diagnostic", loc="left", pad=6)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), handletextpad=0.3, labelspacing=0.25,
          borderaxespad=0.0, fontsize=7.6)

# ------------------------------------------------------------------ b: layer-wise CKA
ax = axb
layers = np.arange(len(mech[0]["cka_forget"]["UO"]))
spec = [("OR", "#009E73", "Original vs retrained", "-"),
        ("UO", "#E69F00", "Unlearned vs original", "-"),
        ("UR", "#B02A2A", "Unlearned vs retrained", "--"),
        ("BO", "#0072B2", "Recovered vs original", "-")]
for key, col, lbl, ls in spec:
    curves = np.array([m["cka_forget"][key] for m in mech])
    ax.fill_between(layers, curves.min(0), curves.max(0), color=col, alpha=0.15, lw=0)
    ax.plot(layers, curves.mean(0), color=col, lw=1.8, ls=ls, label=lbl)
ax.set_xlabel("Layer (0 = embeddings)"); ax.set_ylabel("Linear CKA, forget items")
ax.set_ylim(0.4, 1.02); ax.set_xlim(-0.5, len(layers) - 0.5)
ax.set_title("Suppression produces late-layer divergence", loc="left", pad=6)
ax.legend(loc="lower left", handlelength=2.2, labelspacing=0.3)

# ------------------------------------------------------------------ c: patching
ax = axc
cond = [("Unlearned", np.array([np.mean(p["unlearned_base"]) for p in patch]), "#E69F00"),
        ("+ original state", np.array([np.mean(p["patched_from_original"]) for p in patch]), "#7FB8DE"),
        ("+ retrained state", np.array([np.mean(p["patched_from_retrained"]) for p in patch]), "#009E73"),
        ("Original", orig_lp, "#0072B2")]
for i, (name, v, col) in enumerate(cond):
    ax.bar(i, v.mean(), 0.62, color=col, zorder=2)
    ax.scatter(np.full(3, i) + np.linspace(-0.06, 0.06, 3), v, s=14, c="#333333", zorder=5)
ax.axhline(orig_lp.mean(), color="#4A90C8", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color="black", lw=0.7, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cond], rotation=30, ha="right", rotation_mode="anchor")
ax.set_ylim(-5.6, 0.25)
ax.set_ylabel("Forget log-prob./token")
ax.set_title("Patching the late residual state", loc="left", pad=6)

fig.savefig("figures/fig2.png"); fig.savefig("figures/fig2.pdf")
print("wrote figures/fig2.png and figures/fig2.pdf")
for name, v, _ in groups:
    print(f"  a {name:10s} mean rank {v.mean():.3f} ± {ci95(v):.3f} (n={len(v)})")
print(f"  a 7B ref {ref7.mean():.3f} NegGrad+ {ng7.mean():.3f}; Llama ref {refL.mean():.3f} NegGrad+ {ngL.mean():.3f}; NF4 {nf4.mean():.3f}")
for name, v, _ in cond:
    print(f"  c {name.replace(chr(10), ' '):18s} {np.round(v, 2)}")
