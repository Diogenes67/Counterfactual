"""E2 pilot: is the interdict stored? Eco p254 — the deletion gesture is itself memorable.
Tests whether an unlearned model (Mu) treats the forgotten class in a way that is
distinguishable from a model that never learned it (Mr), using:
  A. black-box output statistics on forget-class inputs (per-input and per-model)
  B. model-type identification (Mu vs Mr) from outputs alone, leave-one-seed-out
  C. dissonance: probe-on-features P(7) minus output P(7)  ("knows but refuses")
"""
import numpy as np, torch, torch.nn as nn, json, copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)
FORGET = 7

def make_model():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

def accuracy(m, X, y):
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def train(m, X, y, epochs=120, lr=1e-3, bs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        idx = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), bs):
            b = idx[i:i+bs]; opt.zero_grad(); lf(m(X[b]), y[b]).backward(); opt.step()
    return m

def unlearn(m, Xf, yf, Xr, yr, lr=1e-4, alpha=0.3, bs=64, seed=0, max_steps=2000):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    for step in range(max_steps):
        bf = torch.randint(0, len(yf), (min(bs, len(yf)),), generator=g)
        br = torch.randint(0, len(yr), (bs,), generator=g)
        opt.zero_grad()
        (-alpha * lf(m(Xf[bf]), yf[bf]) + lf(m(Xr[br]), yr[br])).backward(); opt.step()
        if accuracy(m, Xf, yf) <= 0.02: break
    r = 0
    while accuracy(m, Xr, yr) < 0.98 and r < 300:
        br = torch.randint(0, len(yr), (bs,), generator=g)
        opt.zero_grad(); lf(m(Xr[br]), yr[br]).backward(); opt.step(); r += 1
    return m

def penult(m, X):
    with torch.no_grad():
        h = X
        for layer in list(m)[:-1]: h = layer(h)
    return h.numpy()

def softmax_np(m, X):
    with torch.no_grad(): return torch.softmax(m(X), 1).numpy()

def stats_on(m, X7):
    """Black-box per-model statistics on forget-class inputs."""
    p = softmax_np(m, X7)
    logits = m(X7).detach().numpy()
    p7 = p[:, FORGET]
    rank7 = (logits > logits[:, [FORGET]]).sum(1)  # 0 = top, 9 = bottom
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return {"mean_p7": float(p7.mean()), "mean_rank7": float(rank7.mean()),
            "frac_rank_last": float((rank7 == 9).mean()), "mean_entropy": float(ent.mean()),
            "mean_maxp": float(p[:, :].max(1).mean())}

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")

rows = []
for seed in range(5):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    f = ytr == FORGET
    Xf, yf, Xr, yr = Xtr[f], ytr[f], Xtr[~f], ytr[~f]
    te7 = yte == FORGET
    Xte7 = Xte[te7]

    torch.manual_seed(seed)
    M0 = train(make_model(), Xtr, ytr, seed=seed)
    torch.manual_seed(seed + 100)
    Mr = train(make_model(), Xr, yr, seed=seed + 100)
    Mu = unlearn(copy.deepcopy(M0), Xf, yf, Xr, yr, seed=seed)

    row = {"seed": seed}
    for name, m in [("M0", M0), ("Mu", Mu), ("Mr", Mr)]:
        row[name] = stats_on(m, Xte7)
        # C. dissonance: probe on penultimate features (fit on train incl 7s) vs output P(7)
        probe = LogisticRegression(max_iter=3000).fit(penult(m, Xtr), ytr.numpy())
        prb7 = probe.predict_proba(penult(m, Xte))[:, list(probe.classes_).index(FORGET)] if FORGET in probe.classes_ else np.zeros(len(yte))
        out7 = softmax_np(m, Xte)[:, FORGET]
        D = prb7 - out7
        mask = te7.numpy()
        row[name]["dissonance_7"] = float(D[mask].mean())
        row[name]["dissonance_other"] = float(D[~mask].mean())
        row[name]["interdict_score"] = row[name]["dissonance_7"] - row[name]["dissonance_other"]
    rows.append(row)
    print(f"seed {seed}:")
    for n in ["M0", "Mu", "Mr"]:
        s = row[n]
        print(f"  {n}: p7={s['mean_p7']:.4f} rank7={s['mean_rank7']:.2f} last={s['frac_rank_last']:.2f} "
              f"ent={s['mean_entropy']:.3f} D7={s['dissonance_7']:.3f} Dother={s['dissonance_other']:.3f} I={s['interdict_score']:.3f}")

# B. model-type identification from black-box stats alone, leave-one-seed-out
feats, labels, seeds_arr = [], [], []
KEYS = ["mean_p7", "mean_rank7", "frac_rank_last", "mean_entropy", "mean_maxp"]
for row in rows:
    for name, lab in [("Mu", 1), ("Mr", 0)]:
        feats.append([row[name][k] for k in KEYS]); labels.append(lab); seeds_arr.append(row["seed"])
feats, labels, seeds_arr = np.array(feats), np.array(labels), np.array(seeds_arr)
correct = 0
for s in range(5):
    tr, te = seeds_arr != s, seeds_arr == s
    clf = LogisticRegression(max_iter=3000).fit(feats[tr], labels[tr])
    correct += (clf.predict(feats[te]) == labels[te]).sum()
print(f"\nB. model-type ID (Mu vs Mr) from black-box stats, leave-one-seed-out: {correct}/10 correct")

# which single stat separates best?
for k in KEYS:
    mu_v = [r["Mu"][k] for r in rows]; mr_v = [r["Mr"][k] for r in rows]
    print(f"   {k}: Mu={np.mean(mu_v):.4f}±{np.std(mu_v):.4f}  Mr={np.mean(mr_v):.4f}±{np.std(mr_v):.4f}")
print(f"   interdict_score: Mu={np.mean([r['Mu']['interdict_score'] for r in rows]):.3f}±{np.std([r['Mu']['interdict_score'] for r in rows]):.3f}  "
      f"Mr={np.mean([r['Mr']['interdict_score'] for r in rows]):.3f}±{np.std([r['Mr']['interdict_score'] for r in rows]):.3f}  "
      f"M0={np.mean([r['M0']['interdict_score'] for r in rows]):.3f}")

json.dump(rows, open("/tmp/demo/interdict_results.json", "w"), indent=1)
print("saved")
