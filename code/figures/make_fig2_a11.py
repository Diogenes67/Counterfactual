import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8.5,
                     'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.7, 'legend.fontsize': 7, 'legend.frameon': False,
                     'savefig.dpi': 300})

fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.7))
fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.20, wspace=0.35)

# ---------- a: layer-wise CKA (forget items, mean across seeds) ----------
ax = axes[0]
curves = {k: [] for k in ['UO', 'UR', 'OR', 'BO']}
for s in '012':
    r = json.load(open(f'/tmp/v7/mech_s{s}.json'))
    for k in curves:
        curves[k].append(r['cka_forget'][k])
L = len(curves['UO'][0])
x = np.arange(L)
spec = [('OR', '#009E73', 'Original vs retrained', '-'),
        ('UO', '#E69F00', 'Unlearned vs original', '-'),
        ('UR', '#B02A2A', 'Unlearned vs retrained', '--'),
        ('BO', '#0072B2', 'Recovered vs original', '-')]
for k, c, lbl, ls in spec:
    m = np.mean(curves[k], 0)
    lo = np.min(curves[k], 0); hi = np.max(curves[k], 0)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(x, m, color=c, lw=1.4, ls=ls, label=lbl)
ax.set_xlabel('Layer (0 = embeddings)')
ax.set_ylabel('Linear CKA, forget items')
ax.set_ylim(0.38, 1.02)
ax.set_title('The edit is a late-layer overlay', loc='left', pad=4)
ax.legend(loc='lower left', labelspacing=0.25, handletextpad=0.4)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-0.13, 1.06, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')

# ---------- b: audit state space ----------
ax = axes[1]
import glob
def load(p):
    return json.load(open(p))
O = [load(f'/tmp/v3/15b_lora_s{s}_full.json') for s in '012']
NG = [load(f'/tmp/v3/15b_lora_s{s}_unlearn_neggrad.json') for s in '012']
NF = [load(f'/tmp/v56/results_v5/15b_lora_s{s}_nf4_unlearn_neggrad.json') for s in '012']
CT = ([load(f'/tmp/v3/15b_lora_s{s}_retain90.json') for s in '012'] +
      [load(f'/tmp/v4/15b_lora_s{s}_retain90.json') for s in '34'] +
      [load(f'/tmp/v56/results_v5/15b_lora_s{s}_retain90.json') for s in '56789'])
def xy(rs):
    return [r['forget_logprob'] for r in rs], [r['interdict_rank'] for r in rs]
cx, cy = xy(CT)
ax.scatter(cx, cy, s=28, color='#009E73', edgecolor='white', lw=0.6, zorder=4,
           label='Retrained controls (n=10)')
from matplotlib.patches import Ellipse
ax.add_patch(Ellipse((np.mean(cx), np.mean(cy)), max(3.2*np.std(cx), 0.9), max(3.2*np.std(cy), 0.28),
                     facecolor='#009E73', alpha=0.12, edgecolor='none', zorder=1))
ox, oy = xy(O)
ax.scatter(ox, oy, s=34, color='#0072B2', edgecolor='white', lw=0.6, zorder=4, label='Original')
nx, ny = xy(NG)
ax.scatter(nx, ny, s=34, color='#E69F00', edgecolor='white', lw=0.6, zorder=4, label='NegGrad+')
fx, fy = xy(NF)
ax.scatter(fx, fy, s=34, marker='s', color='#B02A2A', edgecolor='white', lw=0.6, zorder=4,
           label='NegGrad+ after NF4')
OK = [load(f'/tmp/v8/15b_lora_s{s}_oracle_kd.json') for s in '012']
kx, ky = xy(OK)
ax.scatter(kx, ky, s=30, marker='D', color='#E8B4D0', edgecolor='white', lw=0.6, zorder=4,
           label='Oracle edit, 300 steps')
OX = [load(f'/tmp/v9r/results_v9/15b_lora_s{s}_oraclex.json') for s in '012']
xx, xy2 = xy(OX)
ax.scatter(xx, xy2, s=34, marker='D', color='#CC79A7', edgecolor='white', lw=0.6, zorder=4,
           label='Oracle edit, 3,000 steps')
ax.annotate('', xy=(np.mean(xx), np.mean(xy2)), xytext=(np.mean(kx), np.mean(ky)),
            arrowprops=dict(arrowstyle='-|>', color='#CC79A7', lw=1.0,
                            shrinkA=6, shrinkB=6, connectionstyle='arc3,rad=0.25'))
DS = [load('/tmp/v56/results_v6/15b_lora_s0_attr_distill.json'),
      load('/tmp/v9r/results_v9/15b_lora_s1_attr_distill.json'),
      load('/tmp/v9r/results_v9/15b_lora_s2_attr_distill.json')]
dx, dy = xy(DS)
ax.scatter(dx, dy, s=34, marker='^', color='#555555', edgecolor='white', lw=0.6, zorder=4,
           label='Distilled students')
ax.annotate('', xy=(np.mean(nx), np.mean(ny)), xytext=(np.mean(ox), np.mean(oy)),
            arrowprops=dict(arrowstyle='-|>', color='#888888', lw=1.1,
                            shrinkA=8, shrinkB=8, connectionstyle='arc3,rad=-0.15'))
ax.text(-3.75, 0.56, 'unlearning', fontsize=6.8, color='#666666', ha='center', rotation=-30)
ax.annotate('', xy=(np.mean(fx), np.mean(fy)), xytext=(np.mean(nx), np.mean(ny)),
            arrowprops=dict(arrowstyle='-|>', color='#B02A2A', lw=1.2,
                            shrinkA=8, shrinkB=8, connectionstyle='arc3,rad=-0.2'))
ax.text(-7.05, 0.52, 'NF4: content returns,\nverdict reverses', fontsize=6.8, color='#B02A2A', ha='left')
ax.text(np.mean(kx) - 0.2, 0.20, 'oracle: closer\nwith budget', fontsize=6.8, color='#CC79A7', ha='center')
ax.set_xlabel('Forget log-prob./token (content held)')
ax.set_ylabel('Mean rank of truth (audit statistic)')
ax.set_title("The audit's state space", loc='left', pad=4)
ax.legend(loc='lower left', labelspacing=0.25, handletextpad=0.3, fontsize=6.6)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-0.13, 1.06, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold')

plt.savefig('/tmp/a5/fig2_a11.png', dpi=300)
plt.savefig('/tmp/a5/fig2_a11.pdf')
print('saved fig2_a11')
