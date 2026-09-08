import numpy as np, torch, torch.nn as nn, json, copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)
FORGET_CLASS = 7

def make_model():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

def accuracy(model, X, y):
    if len(y) == 0: return float('nan')
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def train(model, X, y, epochs=120, lr=1e-3, bs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(y)
    for ep in range(epochs):
        idx = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad(); lossf(model(X[b]), y[b]).backward(); opt.step()
    return model

def unlearn(model, Xf, yf, Xr, yr, lr=1e-4, alpha=0.3, max_steps=2000, bs=64, seed=0):
    """NegGrad+: weighted ascent on forget batch + descent on retain batch, early stop, then retain repair."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    steps_used = max_steps
    for step in range(max_steps):
        bf = torch.randint(0, len(yf), (min(bs, len(yf)),), generator=g)
        br = torch.randint(0, len(yr), (bs,), generator=g)
        opt.zero_grad()
        loss = -alpha * lossf(model(Xf[bf]), yf[bf]) + lossf(model(Xr[br]), yr[br])
        loss.backward(); opt.step()
        if accuracy(model, Xf, yf) <= 0.02:
            steps_used = step + 1
            break
    # repair: brief descent on retain only (no forget signal)
    repair = 0
    while accuracy(model, Xr, yr) < 0.98 and repair < 300:
        br = torch.randint(0, len(yr), (bs,), generator=g)
        opt.zero_grad(); lossf(model(Xr[br]), yr[br]).backward(); opt.step()
        repair += 1
    return model, steps_used, repair

def quantize(model, bits):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 1:
                s = p.abs().max() / (2**(bits-1) - 1)
                if s > 0: p.copy_(torch.round(p / s) * s)
    return m

def penult(model, X):
    with torch.no_grad():
        h = X
        for layer in list(model)[:-1]:
            h = layer(h)
        return h.numpy()

def probe_acc(model, Xtr, ytr, Xte, yte, cls):
    """Linear probe on frozen penultimate features; report probe accuracy on held-out class-cls examples."""
    clf = LogisticRegression(max_iter=2000).fit(penult(model, Xtr), ytr.numpy())
    mask = (yte == cls).numpy()
    return float(clf.score(penult(model, Xte)[mask], yte.numpy()[mask]))

def savings(model, X7, y7, Xte, yte, Xte_r, yte_r, steps=40, lr=1e-3):
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    mask7 = (yte == FORGET_CLASS)
    curve = [accuracy(m, Xte[mask7], yte[mask7])]
    for s in range(steps):
        opt.zero_grad(); lossf(m(X7), y7).backward(); opt.step()
        curve.append(accuracy(m, Xte[mask7], yte[mask7]))
    other = accuracy(m, Xte_r, yte_r)
    return curve, other

results = {"seeds": [], "config": {"forget_class": FORGET_CLASS, "bits": [8, 6, 5, 4, 3, 2]}}
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")

for seed in range(5):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    f = ytr == FORGET_CLASS
    Xf, yf, Xr, yr = Xtr[f], ytr[f], Xtr[~f], ytr[~f]
    te7 = yte == FORGET_CLASS
    Xte7, yte7, Xte_r, yte_r = Xte[te7], yte[te7], Xte[~te7], yte[~te7]

    torch.manual_seed(seed)
    M0 = train(make_model(), Xtr, ytr, seed=seed)
    torch.manual_seed(seed + 100)
    Mr = train(make_model(), Xr, yr, seed=seed + 100)          # exact: never saw 7s
    Mu, steps_used, repair_steps = unlearn(copy.deepcopy(M0), Xf, yf, Xr, yr, seed=seed)

    r = {"unlearn_steps": steps_used, "repair_steps": repair_steps}
    for name, m in [("M0", M0), ("Mu", Mu), ("Mr", Mr)]:
        r[name] = {"forget_acc": accuracy(m, Xte7, yte7), "other_acc": accuracy(m, Xte_r, yte_r)}
    # representation probe (probe trained on TEST-split features w/ labels incl 7s, eval on train-7s -> use train as heldout for probe eval)
    r["probe_M0"] = probe_acc(M0, Xte, yte, Xtr, ytr, FORGET_CLASS)
    r["probe_Mu"] = probe_acc(Mu, Xte, yte, Xtr, ytr, FORGET_CLASS)
    r["probe_Mr"] = probe_acc(Mr, Xte, yte, Xtr, ytr, FORGET_CLASS)
    # superimposition diagnostic: per-class mean |delta| of final-layer weights+bias, Mu vs M0
    W0, b0 = list(M0)[-1].weight.detach(), list(M0)[-1].bias.detach()
    Wu, bu = list(Mu)[-1].weight.detach(), list(Mu)[-1].bias.detach()
    dW = (Wu - W0).abs().mean(1) + (bu - b0).abs()
    r["row_delta"] = dW.tolist()
    # quantization sweep on Mu and Mr
    r["quant"] = {}
    for bits in results["config"]["bits"]:
        qu, qr = quantize(Mu, bits), quantize(Mr, bits)
        r["quant"][bits] = {
            "Mu_forget": accuracy(qu, Xte7, yte7), "Mu_other": accuracy(qu, Xte_r, yte_r),
            "Mr_forget": accuracy(qr, Xte7, yte7), "Mr_other": accuracy(qr, Xte_r, yte_r)}
    # relearning savings: 10 sevens from the (former) forget set
    idx = torch.arange(len(yf))[:10]
    cu, ou = savings(Mu, Xf[idx], yf[idx], Xte, yte, Xte_r, yte_r)
    cr, orr = savings(Mr, Xf[idx], yf[idx], Xte, yte, Xte_r, yte_r)
    r["savings"] = {"Mu": cu, "Mr": cr, "Mu_other_after": ou, "Mr_other_after": orr}
    results["seeds"].append(r)
    print(f"seed {seed}: steps={steps_used} M0={r['M0']} Mu={r['Mu']} Mr={r['Mr']} probeMu={r['probe_Mu']:.3f} probeMr={r['probe_Mr']:.3f}")

with open("/tmp/demo/results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print("done")
