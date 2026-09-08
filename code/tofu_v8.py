"""tofu_v8.py — the falsification campaign: methods chosen to attack the paper's own thesis.
COMPANION cells: run tofu_all, then the tofu_v4, tofu_v5, tofu_v6 and tofu_v7 cells first
(v8 reuses their functions, including v7's _hidden_mat/_cka), then this cell.

Stages (sys.argv = ["tofu_v8.py", "<stage>", seed]; main8()):
  mechedit [seed]   mechanistically localized editing (adaptation of Guo et al., ICML 2025).
                    Localizes the MLP layers most attributable to forget-set recall (forget/retain
                    gradient-attribution ratio), then edits ONLY those layers full-parameter with
                    the NegGrad+ objective under the standard guard. If even substrate-directed
                    editing leaves the counterfactual gap, the finding hardens; if it closes the
                    gap, the boundary is found. Battery + rank + layer-wise CKA.
  robust [seed]     relearning-invariant unlearning (in the spirit of ILU, Wang et al. ICML 2025,
                    and sharpness-aware unlearning, Fan et al. ICML 2025). SAM-style worst-case
                    objective: forget likelihood is minimized AT the parameters perturbed one
                    normalized benign-gradient step away, so suppression is optimized to survive
                    benign fine-tuning. Battery (its relearn channels are the direct test) + rank + CKA.
  oracle [seed]     oracle counterfactual projection: fine-tune the original TOWARD the same-seed
                    retrained control, matching output distributions on forget+retain items
                    (KL to the teacher). Deliberately impractical: it asks whether an in-place
                    edit explicitly optimized toward the counterfactual can reach it. Battery +
                    rank (reference EXCLUDES the teacher control) + CKA.
  oraclerep         seed-0 variant adding hidden-state matching at layers 7/14/21/27 to the KL
                    objective: how much of the model must be reconstructed before forgetting is real?
  figures8          summary print of results_v8.

All stages skip existing outputs. Results -> results_v8/.
Run order: mechedit 0/1/2 -> robust 0/1/2 -> oracle 0/1/2 -> oraclerep -> figures8.
Rough time on the fast card: 2.5-3.5 h.
"""

try:
    _cka  # from the tofu_v7 cell
except NameError:
    from tofu_v7 import *

RESULTS8 = "results_v8"
MECH_TOPK_LAYERS = 4
MECH_LR, MECH_STEPS = 1e-5, 400
ROBUST_RHO, ROBUST_LR, ROBUST_STEPS = 2e-3, 2e-5, 400
ORACLE_LR, ORACLE_STEPS, ORACLE_LAMBDA = 1e-4, 300, 1.0
ORACLEREP_MU, ORACLEREP_LAYERS = 0.05, [7, 14, 21, 27]


def save8(tag, res):
    os.makedirs(RESULTS8, exist_ok=True)
    res["version"] = "v8"
    with open(f"{RESULTS8}/{tag}.json", "w") as f:
        json.dump(res, f)
    print("saved", f"{RESULTS8}/{tag}.json")


def _battery8(tag, data, dirs, seed):
    ns = _v3ns()
    old = ns.RESULTS
    ns.RESULTS = RESULTS8
    try:
        run_battery(tag, data, MODEL_15, dirs, seed)
    finally:
        ns.RESULTS = old


def _cka8(tag, adapter_dirs, seed, forget, retain):
    """Layer-wise CKA of the new checkpoint vs original and vs same-seed control; appended to its result."""
    path = f"{RESULTS8}/{tag}.json"
    res = json.load(open(path))
    if "cka_forget_XO" in res:
        print("cka exists, skipping", tag)
        return
    mats = {}
    for name, dirs in [("X", adapter_dirs),
                       ("O", {"adapter": f"adapters/s{seed}/full"}),
                       ("R", {"adapter": f"adapters/s{seed}/retain90"})]:
        m, tok = load_model(MODEL_15, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
        mats[name] = {"forget": _hidden_mat(m, tok, forget), "retain": _hidden_mat(m, tok, retain)}
        free(m)
    for split in ["forget", "retain"]:
        L = len(mats["X"][split])
        res[f"cka_{split}_XO"] = [round(_cka(mats["X"][split][i], mats["O"][split][i]), 4) for i in range(L)]
        res[f"cka_{split}_XR"] = [round(_cka(mats["X"][split][i], mats["R"][split][i]), 4) for i in range(L)]
    with open(path, "w") as f:
        json.dump(res, f)
    print(f"  cka appended to {tag}: XO mid {np.mean(res['cka_forget_XO'][7:21]):.3f} "
          f"XR mid {np.mean(res['cka_forget_XR'][7:21]):.3f}")


def _mlp_layers(model):
    core = model
    while not hasattr(core, "layers"):
        core = core.model if hasattr(core, "model") else core.base_model
    return core.layers


# ---------------- mechedit ----------------

def stage_mechedit(data, seed):
    tag = f"15b_lora_s{seed}_unlearn_mechedit"
    out = f"adapters/s{seed}/unlearn_mechedit"
    forget, retain = data["forget"], data["retain"]
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        merged = model.merge_and_unload()
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        merged.train()
        # 1. localization: forget vs retain gradient attribution per layer's MLP matrices
        layers = _mlp_layers(merged)
        names = ["gate_proj", "up_proj", "down_proj"]
        def attribution(rows, nb=8):
            for p in merged.parameters():
                p.requires_grad = True
            score = [0.0] * len(layers)
            for b in range(nb):
                batch = next(make_batches(tok, rows, FT_BS, seed=100 + b))
                merged.zero_grad()
                batch_loss(merged, batch).backward()
                for i, lay in enumerate(layers):
                    for nm in names:
                        g = getattr(lay.mlp, nm).weight.grad
                        if g is not None:
                            score[i] += float((g.float() ** 2).sum())
            merged.zero_grad()
            return np.array(score)
        f_attr = attribution(forget)
        r_attr = attribution(retain)
        ratio = f_attr / (r_attr + 1e-12)
        top = list(np.argsort(-ratio)[:MECH_TOPK_LAYERS])
        print(f"mechedit s{seed}: top layers by forget/retain attribution {sorted(int(t) for t in top)} "
              f"(ratios {[round(float(ratio[t]), 2) for t in top]})")
        # 2. edit only those layers' MLPs
        for p in merged.parameters():
            p.requires_grad = False
        params = []
        for t in top:
            for nm in names:
                w = getattr(layers[t].mlp, nm).weight
                w.requires_grad = True
                params.append(w)
        opt = torch.optim.AdamW(params, lr=MECH_LR)
        f0 = answer_logprob(merged, tok, forget[:40], n=40)
        r0 = answer_logprob(merged, tok, retain[:40], n=40)
        print(f"  start: forget {f0:.3f} retain {r0:.3f}")
        log = {"top_layers": [int(t) for t in top], "start_forget": f0, "start_retain": r0}
        reason = "max steps"
        for step in range(1, MECH_STEPS + 1):
            opt.zero_grad()
            (-UNLEARN_ALPHA * batch_loss(merged, next(make_batches(tok, forget, FT_BS, seed=step)))).backward()
            batch_loss(merged, next(make_batches(tok, retain, FT_BS, seed=step))).backward()
            opt.step()
            if step % 10 == 0:
                f = answer_logprob(merged, tok, forget[:40], n=40)
                r = answer_logprob(merged, tok, retain[:40], n=40)
                print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f} (drop {r0-r:.2f})", flush=True)
                if r0 - r > UTILITY_GUARD_DROP:
                    reason = "utility guard tripped"
                    break
                if f0 - f > FORGET_STOP_LOGPROB_DROP:
                    reason = "forget target reached"
                    break
        log["stop_reason"] = reason
        merged.save_pretrained(out); tok.save_pretrained(out)
        save8(f"mechedit_tune_s{seed}", {"tag": f"mechedit_tune_s{seed}", "log": log})
        free(merged)
    _battery8(tag, data, {"full": out}, seed)
    _cka8(tag, {"full": out}, seed, data["forget"], data["retain"])


# ---------------- robust (relearning-invariant, SAM-style) ----------------

def stage_robust(data, seed):
    tag = f"15b_lora_s{seed}_unlearn_robust"
    out = f"adapters/s{seed}/unlearn_robust"
    forget, retain = data["forget"], data["retain"]
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=ROBUST_LR)
        f0 = answer_logprob(model, tok, forget[:40], n=40)
        r0 = answer_logprob(model, tok, retain[:40], n=40)
        print(f"robust s{seed} start: forget {f0:.3f} retain {r0:.3f}")
        reason = "max steps"
        for step in range(1, ROBUST_STEPS + 1):
            # benign-relearning direction on a retain batch
            opt.zero_grad()
            batch_loss(model, next(make_batches(tok, retain[200:300], FT_BS, seed=step))).backward()
            gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)) + 1e-12
            eps = []
            with torch.no_grad():
                for p in params:
                    e = (-ROBUST_RHO * p.grad / gnorm).to(p.dtype) if p.grad is not None else None
                    eps.append(e)
                    if e is not None:
                        p.add_(e)  # step TOWARD benign relearning (descent direction)
            # worst-case forget ascent + retain anchor, evaluated at the perturbed point
            opt.zero_grad()
            (-UNLEARN_ALPHA * batch_loss(model, next(make_batches(tok, forget, FT_BS, seed=step)))).backward()
            batch_loss(model, next(make_batches(tok, retain, FT_BS, seed=step))).backward()
            with torch.no_grad():
                for p, e in zip(params, eps):
                    if e is not None:
                        p.sub_(e)
            opt.step()
            if step % 10 == 0:
                f = answer_logprob(model, tok, forget[:40], n=40)
                r = answer_logprob(model, tok, retain[:40], n=40)
                print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f} (drop {r0-r:.2f})", flush=True)
                if r0 - r > UTILITY_GUARD_DROP:
                    reason = "utility guard tripped"
                    break
                if f0 - f > FORGET_STOP_LOGPROB_DROP:
                    reason = "forget target reached"
                    break
        model.save_pretrained(out); tok.save_pretrained(out)
        save8(f"robust_tune_s{seed}", {"tag": f"robust_tune_s{seed}", "stop_reason": reason})
        free(model)
    _battery8(tag, data, {"adapter": out}, seed)
    _cka8(tag, {"adapter": out}, seed, data["forget"], data["retain"])


# ---------------- oracle counterfactual projection ----------------

def _kl_to_teacher(student, teacher, tok, rows, bs):
    ids, lab, att = next(rows)
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=att).logits
    s_logits = student(input_ids=ids, attention_mask=att).logits
    mask = (lab != -100)[:, 1:]
    t_lp = torch.log_softmax(t_logits[:, :-1].float(), -1)
    s_lp = torch.log_softmax(s_logits[:, :-1].float(), -1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


def stage_oracle(data, seed, rep=False):
    suffix = "oraclerep" if rep else "oracle_kd"
    tag = f"15b_lora_s{seed}_{suffix}"
    out = f"adapters/s{seed}/{suffix}"
    forget, retain = data["forget"], data["retain"]
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        student, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        teacher, _ = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/retain90")
        teacher.eval()
        hooks = []
        if rep:
            s_act, t_act = {}, {}
            def mk(store, i):
                def h(mod, inp, outp):
                    store[i] = outp[0] if isinstance(outp, tuple) else outp
                return h
            for i in ORACLEREP_LAYERS:
                hooks.append(_mlp_layers(student)[i].register_forward_hook(mk(s_act, i)))
                hooks.append(_mlp_layers(teacher)[i].register_forward_hook(mk(t_act, i)))
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=ORACLE_LR)
        f0 = answer_logprob(student, tok, forget[:40], n=40)
        f_teacher = answer_logprob(teacher, tok, forget[:40], n=40)
        print(f"{suffix} s{seed} start: student forget {f0:.3f}, teacher forget {f_teacher:.3f}")
        reason = "max steps"
        for step in range(1, ORACLE_STEPS + 1):
            opt.zero_grad()
            fb = iter([next(make_batches(tok, forget, FT_BS, seed=step))])
            loss = _kl_to_teacher(student, teacher, tok, fb, FT_BS)
            rb = iter([next(make_batches(tok, retain, FT_BS, seed=step))])
            loss = loss + ORACLE_LAMBDA * _kl_to_teacher(student, teacher, tok, rb, FT_BS)
            if rep:
                rep_loss = 0.0
                for i in ORACLEREP_LAYERS:
                    rep_loss = rep_loss + ((s_act[i].float() - t_act[i].detach().float()) ** 2).mean()
                loss = loss + ORACLEREP_MU * rep_loss
            loss.backward()
            opt.step()
            if step % 20 == 0:
                f = answer_logprob(student, tok, forget[:40], n=40)
                print(f"  step {step}: kl {float(loss):.4f} forget {f:.3f} (teacher {f_teacher:.3f})", flush=True)
                if abs(f - f_teacher) < 0.2:
                    reason = "teacher forget level reached"
                    break
        for h in hooks:
            h.remove()
        student.save_pretrained(out); tok.save_pretrained(out)
        save8(f"{suffix}_tune_s{seed}", {"tag": f"{suffix}_tune_s{seed}", "stop_reason": reason,
                                         "teacher_forget": f_teacher})
        free(student, teacher)
    _battery8(tag, data, {"adapter": out}, seed)
    _cka8(tag, {"adapter": out}, seed, data["forget"], data["retain"])
    # NOTE for analysis: the rank reference for this arm must EXCLUDE the same-seed teacher control.


def stage_figures8():
    for f in sorted(glob.glob(f"{RESULTS8}/*.json")):
        r = json.load(open(f))
        bits = [r.get("tag", os.path.basename(f))]
        for k in ["forget_logprob", "retain_logprob", "interdict_rank", "stop_reason"]:
            if k in r:
                bits.append(f"{k} {r[k]}")
        if "cka_forget_XO" in r:
            xo = r["cka_forget_XO"]; xr = r["cka_forget_XR"]
            bits.append(f"cka mid XO {np.mean(xo[7:21]):.3f} XR {np.mean(xr[7:21]):.3f}")
        print("  " + " | ".join(str(b) for b in bits))


def main8():
    global RESULTS
    RESULTS = RESULTS8
    os.makedirs(RESULTS8, exist_ok=True)
    stages = ("mechedit", "robust", "oracle", "oraclerep", "figures8")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS8}")
    if stage == "figures8":
        stage_figures8()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "mechedit":
        stage_mechedit(data, seed)
    elif stage == "robust":
        stage_robust(data, seed)
    elif stage == "oracle":
        stage_oracle(data, seed, rep=False)
    elif stage == "oraclerep":
        stage_oracle(data, 0, rep=True)


if __name__ == "__main__":
    main8()
