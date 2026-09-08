"""tofu_v7.py — mechanism, method-class and second-benchmark campaign (supervising-professor
review items 3, 4, 5). COMPANION cells: run the tofu_all.py cell first, then the tofu_v4, tofu_v5
and tofu_v6 cells (v7 reuses their functions), then this cell.

Stages (sys.argv = ["tofu_v7.py", "<stage>", seed]; main7()):
  mech [seed]     the mechanistic analysis. Re-runs benign fine-tuning on the NegGrad+
                  checkpoint saving the recovered weights, then measures (a) layer-wise linear
                  CKA between Original / Unlearned / Recovered / Retrained hidden states on
                  forget and retain items, and (b) the alignment between the benign-recovery
                  weight update and the negative of the unlearning update, per matrix and
                  overall. If benign training preferentially reverses the targeted edit while
                  representations were original-like all along, superimposition stops being an
                  interpretation and becomes a measurement.
  rmu2 [seed]     a tuned RMU-style operating point. Sweeps gentler configurations
                  (steer/alpha/lr) until forget drops >= 2.0 nats with retain drop < 0.5, then
                  runs the full battery on the accepted checkpoint. Establishes what the audit
                  says about representation-level unlearning that does NOT fail the guard.
  museqa          a second positive benchmark. MUSE-News with a QA-format supplement in the
                  fine-tuning mix (disclosed departure from the official MUSE protocol), which
                  creates the probe-format corridor the raw-text arm lacked: full and three
                  retrained references (seeds 0-2), QA-format NegGrad+, battery and rank audit
                  on real news text.
  figures7        prints the mech/rmu2/museqa summaries from results_v7.

All stages skip existing outputs. Results -> results_v7/.
Run order: mech 0/1/2 -> rmu2 0/1/2 -> museqa -> figures7. Rough time: 2.5-3.5 h on the fast card.
"""

try:
    light_eval  # defined when the tofu_all/v4/v5/v6 cells ran in this notebook
except NameError:
    from tofu_v6 import *  # imports v3-v5 transitively

RESULTS7 = "results_v7"
MECH_ITEMS = 40
BENIGN_STEPS7, BENIGN_LR7 = 100, 1e-4
RMU2_CONFIGS = [
    {"steer": 5.0, "alpha": 30.0, "lr": 2e-5, "steps": 400},
    {"steer": 10.0, "alpha": 50.0, "lr": 2e-5, "steps": 400},
    {"steer": 5.0, "alpha": 50.0, "lr": 1e-5, "steps": 600},
]
RMU2_TARGET_DROP = 2.0
MUSEQA_QA_EPOCHS, MUSEQA_QA_LR = 3, 2e-4


def _done7(tag):
    if os.path.exists(f"{RESULTS7}/{tag}.json"):
        print("exists, skipping", tag)
        return True
    return False


# ---------------- mech ----------------

def _hidden_mat(model, tok, rows, n=MECH_ITEMS):
    """Per-layer matrix of mean answer-token hidden states (n items x hidden)."""
    model.eval()
    mats = None
    with torch.no_grad():
        for r in rows[:n]:
            f_ids, labels = fmt_qa(tok, r["question"], r["answer"])
            out = model(input_ids=torch.tensor([f_ids]).to(DEV), output_hidden_states=True)
            tgt = torch.tensor(labels)
            msk = (tgt != -100)
            if msk.sum() == 0:
                continue
            hs = [h[0][msk].float().mean(0).cpu() for h in out.hidden_states]
            if mats is None:
                mats = [[] for _ in hs]
            for i, v in enumerate(hs):
                mats[i].append(v)
    model.train()
    return [torch.stack(m) for m in mats]


def _cka(X, Y):
    """Linear CKA between two (n x d) matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm() ** 2
    xtx = (X.T @ X).norm()
    yty = (Y.T @ Y).norm()
    return float(xty / (xtx * yty + 1e-12))


def _merged_state(model_id, dirs):
    m, tok = load_model(model_id, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
    merged = m.merge_and_unload() if hasattr(m, "merge_and_unload") else m
    sd = {k: v.detach().cpu().clone() for k, v in merged.state_dict().items()
          if v.dtype.is_floating_point and v.dim() >= 2}
    free(merged, m)
    return sd


def stage_mech(data, seed):
    tag = f"mech_s{seed}"
    if _done7(tag):
        return
    forget, retain = data["forget"], data["retain"]

    # 1. benign-recovered checkpoint, saved this time
    out = f"adapters/s{seed}/ng_benignft"
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=BENIGN_LR7)
        rows = retain[200:300]
        f0 = answer_logprob(m, tok, forget[:40], n=40)
        for s in range(BENIGN_STEPS7):
            b = next(make_batches(tok, rows, min(FT_BS, len(rows)), seed=s))
            opt.zero_grad(); batch_loss(m, b).backward(); opt.step()
        f1 = answer_logprob(m, tok, forget[:40], n=40)
        print(f"mech s{seed} benign recovery for weights: {f0:.2f} -> {f1:.2f}")
        m.save_pretrained(out); tok.save_pretrained(out)
        free(m)

    res = {"tag": tag}

    # 2. layer-wise CKA on forget and retain items
    variants = {"O": {"adapter": f"adapters/s{seed}/full"},
                "U": {"adapter": f"adapters/s{seed}/unlearn_neggrad"},
                "B": {"adapter": out},
                "R": {"adapter": f"adapters/s{seed}/retain90"}}
    mats = {}
    for name, dirs in variants.items():
        m, tok = load_model(MODEL_15, adapter_dir=dirs["adapter"])
        mats[name] = {"forget": _hidden_mat(m, tok, forget),
                      "retain": _hidden_mat(m, tok, retain)}
        free(m)
        print(f"  activations captured: {name}")
    pairs = [("U", "O"), ("U", "R"), ("B", "O"), ("B", "R"), ("O", "R"), ("B", "U")]
    for split in ["forget", "retain"]:
        res[f"cka_{split}"] = {}
        L = len(mats["O"][split])
        for a, b in pairs:
            res[f"cka_{split}"][f"{a}{b}"] = [
                round(_cka(mats[a][split][i], mats[b][split][i]), 4) for i in range(L)]
    del mats

    # 3. weight-update alignment: does benign recovery reverse the unlearning edit?
    sdO = _merged_state(MODEL_15, variants["O"])
    sdU = _merged_state(MODEL_15, variants["U"])
    dU = {k: (sdU[k] - sdO[k]).float() for k in sdU}
    del sdO
    sdB = _merged_state(MODEL_15, variants["B"])
    dB = {k: (sdB[k] - sdU[k]).float() for k in sdB}
    del sdB, sdU
    per_matrix = {}
    num = 0.0; den_u = 0.0; den_b = 0.0
    for k in dU:
        u = dU[k].flatten(); b = dB[k].flatten()
        nu, nb = float(u.norm()), float(b.norm())
        if nu < 1e-8 or nb < 1e-8:
            continue
        cos_reverse = float(torch.dot(b, -u) / (nu * nb))
        per_matrix[k] = {"cos_reverse": round(cos_reverse, 4),
                         "norm_unlearn": round(nu, 4), "norm_benign": round(nb, 4)}
        num += float(torch.dot(b, -u)); den_u += nu ** 2; den_b += nb ** 2
    res["align_global_cos_reverse"] = round(num / ((den_u ** 0.5) * (den_b ** 0.5) + 1e-12), 4)
    # norm-weighted mean of per-matrix cosines
    wsum = sum(v["norm_unlearn"] * v["norm_benign"] for v in per_matrix.values())
    res["align_weighted_cos_reverse"] = round(
        sum(v["cos_reverse"] * v["norm_unlearn"] * v["norm_benign"] for v in per_matrix.values()) / (wsum + 1e-12), 4)
    # fraction of the benign update explained by projection onto the reversed unlearning direction
    res["align_projection_fraction"] = round((num / (den_u + 1e-12)) ** 2 * den_u / (den_b + 1e-12), 4)
    res["align_per_matrix_top"] = dict(sorted(per_matrix.items(),
                                              key=lambda kv: -kv[1]["norm_unlearn"])[:20])
    del dU, dB
    print(f"mech s{seed}: global cos(benign, -unlearn) = {res['align_global_cos_reverse']:.3f}, "
          f"weighted {res['align_weighted_cos_reverse']:.3f}")
    save7(tag, res)


def save7(tag, res):
    os.makedirs(RESULTS7, exist_ok=True)
    res["version"] = "v7"
    with open(f"{RESULTS7}/{tag}.json", "w") as f:
        json.dump(res, f)
    print("saved", f"{RESULTS7}/{tag}.json")


# ---------------- rmu2 ----------------

def stage_rmu2(data, seed):
    battery_tag = f"15b_lora_s{seed}_unlearn_rmu2"
    tag_adapter = f"adapters/s{seed}/unlearn_rmu2"
    tune_tag = f"rmu2_tune_s{seed}"
    forget, retain = data["forget"], data["retain"]
    if not (os.path.isdir(tag_adapter) and os.listdir(tag_adapter)):
        accepted, log = None, []
        for ci, cfg in enumerate(RMU2_CONFIGS):
            set_seed(seed)
            model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
            frozen, _ = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full")
            frozen.eval()

            def layers_of(m):
                core = m
                while not hasattr(core, "layers"):
                    core = core.model if hasattr(core, "model") else core.base_model
                return core.layers
            acts, facts = {}, {}
            def mk_hook(store):
                def hook(mod, inp, out):
                    store["h"] = out[0] if isinstance(out, tuple) else out
                return hook
            h1 = layers_of(model)[RMU_LAYER].register_forward_hook(mk_hook(acts))
            h2 = layers_of(frozen)[RMU_LAYER].register_forward_hook(mk_hook(facts))
            hidden = model.config.hidden_size if hasattr(model, "config") else model.base_model.config.hidden_size
            g = torch.Generator().manual_seed(seed)
            u = torch.randn(hidden, generator=g)
            u = (u / u.norm() * cfg["steer"]).to(DEV).to(DTYPE)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
            f0 = answer_logprob(model, tok, forget[:40]); r0 = answer_logprob(model, tok, retain[:40])
            print(f"rmu2 s{seed} config {ci} {cfg}: start forget {f0:.3f} retain {r0:.3f}")
            reason = "max steps"
            for step in range(1, cfg["steps"] + 1):
                opt.zero_grad()
                ids, lab, att = next(make_batches(tok, forget, FT_BS, seed=step))
                mask = (lab != -100).unsqueeze(-1)
                model(input_ids=ids, attention_mask=att)
                loss_f = ((acts["h"] - u) ** 2 * mask).sum() / mask.sum().clamp(min=1)
                loss_f.backward()
                ids, lab, att = next(make_batches(tok, retain, FT_BS, seed=step))
                with torch.no_grad():
                    frozen(input_ids=ids, attention_mask=att)
                    target = facts["h"].detach()
                model(input_ids=ids, attention_mask=att)
                loss_r = cfg["alpha"] * ((acts["h"] - target) ** 2).mean()
                loss_r.backward()
                opt.step()
                if step % 10 == 0:
                    f = answer_logprob(model, tok, forget[:40], n=40)
                    r = answer_logprob(model, tok, retain[:40], n=40)
                    print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f} (drop {r0-r:.2f})", flush=True)
                    if r0 - r > UTILITY_GUARD_DROP:
                        reason = "utility guard tripped"
                        break
                    if f0 - f > RMU2_TARGET_DROP:
                        reason = "forget target reached"
                        break
            f = answer_logprob(model, tok, forget[:40], n=40)
            r = answer_logprob(model, tok, retain[:40], n=40)
            entry = dict(cfg, config_index=ci, stop_reason=reason,
                         forget_drop=round(f0 - f, 3), retain_drop=round(r0 - r, 3))
            log.append(entry)
            h1.remove(); h2.remove()
            if reason == "forget target reached" and (r0 - r) < UTILITY_GUARD_DROP:
                model.save_pretrained(tag_adapter); tok.save_pretrained(tag_adapter)
                accepted = entry
                free(model, frozen)
                break
            free(model, frozen)
        save7(tune_tag, {"tag": tune_tag, "configs_tried": log, "accepted": accepted})
        if accepted is None:
            print(f"rmu2 s{seed}: NO config reached {RMU2_TARGET_DROP} nats within the guard "
                  f"— that bound is itself the result; no battery run")
            return
    ns = _v3ns()
    old = ns.RESULTS
    ns.RESULTS = RESULTS7
    try:
        run_battery(battery_tag, data, MODEL_15, {"adapter": tag_adapter}, seed)
    finally:
        ns.RESULTS = old


# ---------------- museqa ----------------

def _qa_ft(base_model_id, adapter_dir, rows, out, seed, epochs=MUSEQA_QA_EPOCHS, lr=MUSEQA_QA_LR):
    if os.path.isdir(out) and os.listdir(out):
        print("exists, skipping", out)
        return
    set_seed(seed)
    m, tok = load_model(base_model_id, adapter_dir=adapter_dir, trainable=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    steps = epochs * (len(rows) // FT_BS + 1)
    for s in range(steps):
        b = next(make_batches(tok, rows, FT_BS, seed=s))
        opt.zero_grad(); batch_loss(m, b).backward(); opt.step()
    m.save_pretrained(out); tok.save_pretrained(out)
    free(m)


def stage_museqa(data_unused):
    forget_txt, retain_txt, qa_f, qa_r = load_muse()
    fq, fp = muse_qa_rows(qa_f)
    rq, _ = muse_qa_rows(qa_r)
    pad = (rq[40:] * 4)[:100]
    muse_data = {"forget": fq, "retain": rq + pad + pad, "pert": fp}

    # full model: raw text then QA supplement (forget+retain QA); seed 0
    muse_ft(MODEL_15, [forget_txt, retain_txt], "adapters/museqa/full_text", 0)
    _qa_ft(MODEL_15, "adapters/museqa/full_text", fq + rq, "adapters/museqa/full", 0)
    # retrained references: retain text + retain QA only, seeds 0-2
    for s in [0, 1, 2]:
        muse_ft(MODEL_15, [retain_txt], f"adapters/museqa/retain_text_s{s}", s)
        _qa_ft(MODEL_15, f"adapters/museqa/retain_text_s{s}", rq, f"adapters/museqa/retain_s{s}", s)

    # QA-format NegGrad+ on the full model (identical protocol to the TOFU arm)
    out = "adapters/museqa/unlearn_neggrad"
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(0)
        model, tok = load_model(MODEL_15, adapter_dir="adapters/museqa/full", trainable=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=UNLEARN_LR)
        f0 = answer_logprob(model, tok, fq, n=40)
        r0 = answer_logprob(model, tok, rq, n=40)
        print(f"museqa neggrad start: forget-QA {f0:.3f} retain-QA {r0:.3f}")
        for step in range(1, 401):
            opt.zero_grad()
            (-UNLEARN_ALPHA * batch_loss(model, next(make_batches(tok, fq, FT_BS, seed=step)))).backward()
            batch_loss(model, next(make_batches(tok, rq, FT_BS, seed=step))).backward()
            opt.step()
            if step % 20 == 0:
                f = answer_logprob(model, tok, fq, n=40)
                r = answer_logprob(model, tok, rq, n=40)
                print(f"  step {step}: forget-QA {f:.3f} (drop {f0-f:.2f}), retain-QA {r:.3f}", flush=True)
                if r0 - r > UTILITY_GUARD_DROP:
                    print("  utility guard tripped")
                    break
                if f0 - f > FORGET_STOP_LOGPROB_DROP:
                    print("  forget target reached")
                    break
        model.save_pretrained(out); tok.save_pretrained(out)
        free(model)

    ns = _v3ns()
    old = ns.RESULTS
    ns.RESULTS = RESULTS7
    try:
        run_battery("museqa_s0_full", muse_data, MODEL_15, {"adapter": "adapters/museqa/full"}, 0)
        run_battery("museqa_s0_retain", muse_data, MODEL_15, {"adapter": "adapters/museqa/retain_s0"}, 0)
        run_battery("museqa_s0_unlearn_neggrad", muse_data, MODEL_15, {"adapter": out}, 0)
        for s in [1, 2]:
            light_eval(f"museqa_s{s}_retain", muse_data, MODEL_15,
                       {"adapter": f"adapters/museqa/retain_s{s}"}, s)
    finally:
        ns.RESULTS = old


def stage_figures7():
    for f in sorted(glob.glob(f"{RESULTS7}/*.json")):
        r = json.load(open(f))
        bits = [r.get("tag", os.path.basename(f))]
        for k in ["forget_logprob", "retain_logprob", "interdict_rank",
                  "align_global_cos_reverse", "align_weighted_cos_reverse"]:
            if k in r:
                bits.append(f"{k} {r[k]}")
        if "cka_forget" in r:
            uo = r["cka_forget"]["UO"]; ur = r["cka_forget"]["UR"]
            bits.append(f"cka UO mid {uo[len(uo)//2]:.3f} UR mid {ur[len(ur)//2]:.3f}")
        if "accepted" in r:
            bits.append(f"accepted {r['accepted']}")
        print("  " + " | ".join(str(b) for b in bits))


def main7():
    global RESULTS
    RESULTS = RESULTS7
    os.makedirs(RESULTS7, exist_ok=True)
    stages = ("mech", "rmu2", "museqa", "figures7")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS7}")
    if stage == "figures7":
        stage_figures7()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "mech":
        stage_mech(data, seed)
    elif stage == "rmu2":
        stage_rmu2(data, seed)
    elif stage == "museqa":
        stage_museqa(data)


if __name__ == "__main__":
    main7()
