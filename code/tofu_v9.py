"""tofu_v9.py — the referee-response campaign.
COMPANION cells: run tofu_all, then tofu_v4..tofu_v8 cells first (v9 reuses their
functions, including v7's _hidden_mat/_cka and v8's _battery8 pattern), then this cell.

Answers the external referee point by point:
  relearn95 [seed]  (referee 3) relearning rescored on the 95 HELD-OUT forget items.
                    Three rotated folds per seed: demonstrations = forget items
                    [0:5], [5:10], [10:15]; per-item log-probs saved before/after so
                    the trained-5 and held-out-95 gains separate exactly.
  probe9 [seed-]    (referee 5) audit robustness suite, eval only, per-alternative
                    log-probs SAVED for every construction: original 4 distractors;
                    4 other-item true answers as distractors; k=2 and k=3 subsets;
                    paraphrased question with original distractors. Runs original,
                    NegGrad+, NF4-NegGrad+ (quantized on the fly) and every control.
  oraclex [seed]    (referee 6) oracle at 10x budget (3,000 steps, lr 1e-4) with the
                    same convergence rule; if it still stalls at beta~0.5 the
                    reachability reading hardens, if it walks on, we retract.
  oraclehi [seed]   oracle at lr 3e-4, 1,000 steps (optimizer-difficulty control).
  oraclerep [seed]  (referee 6) representation-matching oracle, now all three seeds.
  rmufull [seed]    (referee 7) canonical full-parameter RMU: MLPs of layers 5-7
                    updated in full, activation steering at layer 7, two configs.
  lens [seed]       (referee 2) content-sensitive layer probe: logit-lens decoding of
                    the true answer vs the four distractors at EVERY layer, for
                    original / unlearned / benign-recovered / retrained. If mid-layer
                    states decode the forgotten facts while final layers do not, the
                    overlay reading gains causal-adjacent support.
  nulls [seed]      (referee 2) update-cosine null battery: cos(recovery, -edit)
                    against 10 random draws, an independent benign update of the
                    ORIGINAL, and a second-seed benign update.
  distill [seed]    (referee 8) distillation laundering at seeds 1 and 2 (the only
                    true in-band laundering is currently single-seed).
  museqa [seed]     (referee 4) MUSE-QA unlearn+battery for seeds 1 and 2 (calls the
                    tofu_v7 museqa stage if present in the namespace).
  figures9          summary print of results_v9.

All stages skip existing outputs. Results -> results_v9/. Back up after EVERY stage
(the driver cell's backup() call), exactly as in the v5/v6 recovery driver.
Run order: relearn95 0/1/2 -> probe9 -> oraclex 0/1/2 -> oraclehi 0 -> oraclerep 1/2
-> rmufull 0/1/2 -> lens 0/1/2 -> nulls 0/1/2 -> distill 1/2 -> museqa 1/2.
Rough time on the fast card: 8-10 h total; relearn95+probe9+lens+nulls alone
(the free-standing referee answers) are ~2.5 h and can run as a first session.
"""

try:
    _cka  # from the tofu_v7 cell
except NameError:
    from tofu_v7 import *

RESULTS9 = "results_v9"
RESULTS8 = "results_v9"  # route v8 helpers (_battery8/_cka8/save8) into results_v9
RELEARN9_FOLDS = [(0, 5), (5, 10), (10, 15)]
ORACLEX_STEPS, ORACLEX_LR = 3000, 1e-4
ORACLEHI_STEPS, ORACLEHI_LR = 1000, 3e-4
RMU9_LAYERS, RMU9_ACT_LAYER = [5, 6, 7], 7
RMU9_CONFIGS = [{"name": "a", "c": 20.0, "alpha": 100.0, "lr": 5e-5, "steps": 300},
                {"name": "b", "c": 6.5, "alpha": 1200.0, "lr": 5e-5, "steps": 300}]
DISTILL9_N, DISTILL9_STEPS = 800, 300
NULLS9_DRAWS = 10


def save9(tag, res):
    os.makedirs(RESULTS9, exist_ok=True)
    res["version"] = "v9"
    with open(f"{RESULTS9}/{tag}.json", "w") as f:
        json.dump(res, f)
    print("saved", f"{RESULTS9}/{tag}.json")


def _battery9(tag, data, dirs, seed):
    ns = _v3ns()
    old = ns.RESULTS
    ns.RESULTS = RESULTS9
    try:
        run_battery(tag, data, MODEL_15, dirs, seed)
    finally:
        ns.RESULTS = old


# ---------- self-contained per-item scoring (used by relearn95, probe9, lens) ----------
# Chat-template teacher forcing, mean log-prob per answer token. probe9 recomputes the
# ORIGINAL construction with this scorer and prints it next to the stored interdict_rank
# as a consistency check, so every robustness comparison is internally consistent.

def _score_items(model, tok, pairs, bs=16):
    """pairs: list of (question, answer). returns np.array of mean answer-token logprobs."""
    out = []
    model.eval()
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        texts, plens = [], []
        for q, a in chunk:
            prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True)
            plens.append(len(tok(prompt, add_special_tokens=False).input_ids))
            texts.append(prompt + a)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tgt = enc.input_ids[:, 1:]
        gathered = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for j in range(len(chunk)):
            n_tok = int(enc.attention_mask[j].sum())
            a0, a1 = plens[j] - 1, n_tok - 1  # answer-token positions in shifted space
            out.append(float(gathered[j, a0:a1].mean()))
    return np.array(out)


def _qa(rows):
    return [(r["question"], r["answer"]) for r in rows]


# ---------------- relearn95: held-out relearning with rotated folds ----------------

def stage_relearn95(data, seed):
    forget = data["forget"]
    for k, (lo, hi) in enumerate(RELEARN9_FOLDS):
        tag = f"relearn95_s{seed}_fold{k}"
        if os.path.exists(f"{RESULTS9}/{tag}.json"):
            print("exists, skipping", tag)
            continue
        set_seed(seed)
        model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad",
                                trainable=True)
        before = _score_items(model, tok, _qa(forget[:100]))
        demos = forget[lo:hi]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.train()
        for step in range(1, 31):  # mirrors the v3 relearn channel: 5 examples, 30 steps
            opt.zero_grad()
            batch_loss(model, next(make_batches(tok, demos, min(FT_BS, 5), seed=step))).backward()
            opt.step()
        after = _score_items(model, tok, _qa(forget[:100]))
        demo_idx = list(range(lo, hi))
        held = [i for i in range(100) if i not in demo_idx]
        save9(tag, {"tag": tag, "fold": k, "demo_idx": demo_idx,
                    "items_before": [round(float(x), 4) for x in before],
                    "items_after": [round(float(x), 4) for x in after],
                    "gain_demo": round(float(after[demo_idx].mean() - before[demo_idx].mean()), 3),
                    "gain_heldout95": round(float(after[held].mean() - before[held].mean()), 3)})
        print(f"  {tag}: demo gain {after[demo_idx].mean()-before[demo_idx].mean():+.2f}, "
              f"held-out-95 gain {after[held].mean()-before[held].mean():+.2f}")
        free(model)


# ---------------- probe9: audit robustness, per-alternative dumps ----------------

def _probe_models(seed_list):
    models = []
    for s in seed_list:
        models.append((f"full_s{s}", {"adapter": f"adapters/s{s}/full"}, False))
        models.append((f"ng_s{s}", {"adapter": f"adapters/s{s}/unlearn_neggrad"}, False))
        models.append((f"nf4ng_s{s}", {"adapter": f"adapters/s{s}/unlearn_neggrad"}, True))
    for d in sorted(glob.glob("adapters/s*/retain90")):
        s = d.split("/")[1]
        models.append((f"retain_{s}", {"adapter": d}, False))
    return models


def stage_probe9(data):
    pert = data["pert"]
    rng = np.random.RandomState(9)
    # constructions: name -> list per item of candidate answers (truth first)
    cons = {}
    cons["orig4"] = [[r["answer"]] + list(r["perturbed_answer"])[:4] for r in pert[:100]]
    other = [r["answer"] for r in pert[:100]]
    cons["otheritem4"] = []
    for i, r in enumerate(pert[:100]):
        pool = [a for j, a in enumerate(other) if j != i]
        cons["otheritem4"].append([r["answer"]] + list(rng.choice(pool, 4, replace=False)))
    cons["orig2"] = [[c[0]] + c[1:3] for c in cons["orig4"]]
    cons["orig3"] = [[c[0]] + c[1:4] for c in cons["orig4"]]
    para_q = [r.get("paraphrased_question", r["question"]) for r in pert[:100]]
    for name, mdirs, nf4 in _probe_models("012"):
        tag = f"probe9_{name}"
        if os.path.exists(f"{RESULTS9}/{tag}.json"):
            print("exists, skipping", tag)
            continue
        if nf4:  # v5's NF4 path: merge the adapter, reload 4-bit
            from transformers import BitsAndBytesConfig
            import shutil
            m0, tok = load_model(MODEL_15, adapter_dir=mdirs.get("adapter"))
            merged = m0.merge_and_unload()
            merged.save_pretrained("tmp_merged_nf4"); tok.save_pretrained("tmp_merged_nf4")
            free(merged, m0)
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=DTYPE)
            model = AutoModelForCausalLM.from_pretrained("tmp_merged_nf4",
                                                         quantization_config=bnb,
                                                         device_map={"": 0})
        else:
            model, tok = load_model(MODEL_15, adapter_dir=mdirs.get("adapter"),
                                    full_dir=mdirs.get("full"))
        res = {"tag": tag}
        for cname, cands in cons.items():
            qs = para_q if cname == "para" else [r["question"] for r in pert[:100]]
            flat = [(q, a) for q, cc in zip(qs, cands) for a in cc]
            lps = _score_items(model, tok, flat)
            width = len(cands[0])
            lps = lps.reshape(100, width)
            ranks = (lps[:, 1:] > lps[:, :1]).mean(1)
            res[f"{cname}_logprobs"] = [[round(float(x), 4) for x in row] for row in lps]
            res[f"{cname}_rank"] = round(float(ranks.mean()), 4)
        # paraphrased question, original 4 distractors
        flat = [(q, a) for q, cc in zip(para_q, cons["orig4"]) for a in cc]
        lps = _score_items(model, tok, flat).reshape(100, 5)
        res["para_rank9"] = round(float((lps[:, 1:] > lps[:, :1]).mean(1).mean()), 4)
        save9(tag, res)
        print(f"  {tag}: orig4 {res['orig4_rank']:.2f} otheritem4 {res['otheritem4_rank']:.2f} "
              f"k2 {res['orig2_rank']:.2f} k3 {res['orig3_rank']:.2f} para {res['para_rank9']:.2f}")
        free(model)
        if nf4:
            import shutil
            shutil.rmtree("tmp_merged_nf4", ignore_errors=True)


# ---------------- oraclex / oraclehi: budget and step-size controls ----------------

def stage_oraclex(data, seed, hi=False):
    suffix = "oraclehi" if hi else "oraclex"
    steps, lr = (ORACLEHI_STEPS, ORACLEHI_LR) if hi else (ORACLEX_STEPS, ORACLEX_LR)
    tag = f"15b_lora_s{seed}_{suffix}"
    out = f"adapters/s{seed}/{suffix}"
    forget, retain = data["forget"], data["retain"]
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        student, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        teacher, _ = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/retain90")
        teacher.eval()
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
        f_teacher = answer_logprob(teacher, tok, forget[:40], n=40)
        print(f"{suffix} s{seed}: teacher forget {f_teacher:.3f}, budget {steps} steps at lr {lr}")
        reason, curve = "max steps", []
        for step in range(1, steps + 1):
            opt.zero_grad()
            fb = iter([next(make_batches(tok, forget, FT_BS, seed=step))])
            loss = _kl_to_teacher(student, teacher, tok, fb, FT_BS)
            rb = iter([next(make_batches(tok, retain, FT_BS, seed=step))])
            loss = loss + ORACLE_LAMBDA * _kl_to_teacher(student, teacher, tok, rb, FT_BS)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                f = answer_logprob(student, tok, forget[:40], n=40)
                curve.append([step, round(float(f), 3)])
                print(f"  step {step}: kl {float(loss):.4f} forget {f:.3f} (teacher {f_teacher:.3f})", flush=True)
                if abs(f - f_teacher) < 0.2:
                    reason = "teacher forget level reached"
                    break
        student.save_pretrained(out); tok.save_pretrained(out)
        save9(f"{suffix}_tune_s{seed}", {"tag": f"{suffix}_tune_s{seed}", "stop_reason": reason,
                                         "teacher_forget": f_teacher, "curve": curve})
        free(student, teacher)
    _battery9(tag, data, {"adapter": out}, seed)
    _cka8(tag, {"adapter": out}, seed, data["forget"], data["retain"])
    # analysis: rank reference EXCLUDES the same-seed teacher, as in v8.


# ---------------- rmufull: canonical full-parameter RMU ----------------

def stage_rmufull(data, seed):
    forget, retain = data["forget"], data["retain"]
    for cfg in RMU9_CONFIGS:
        tag = f"15b_lora_s{seed}_rmufull_{cfg['name']}"
        out = f"adapters/s{seed}/rmufull_{cfg['name']}"
        if os.path.isdir(out) and os.listdir(out):
            print("exists, skipping", tag)
        else:
            set_seed(seed)
            model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
            merged = model.merge_and_unload()
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
            frozen, _ = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full")
            frozen.eval()
            layers = _mlp_layers(merged)
            for p in merged.parameters():
                p.requires_grad = False
            params = []
            for li in RMU9_LAYERS:
                for p in layers[li].mlp.parameters():
                    p.requires_grad = True
                    params.append(p)
            acts, facts = {}, {}
            def mk(store):
                def h(mod, inp, outp):
                    store["h"] = outp[0] if isinstance(outp, tuple) else outp
                return h
            h1 = _mlp_layers(merged)[RMU9_ACT_LAYER].register_forward_hook(mk(acts))
            h2 = _mlp_layers(frozen)[RMU9_ACT_LAYER].register_forward_hook(mk(facts))
            d_model = merged.config.hidden_size
            u = torch.randn(d_model, device=merged.device)
            u = (u / u.norm()) * cfg["c"]
            opt = torch.optim.AdamW(params, lr=cfg["lr"])
            f0 = answer_logprob(merged, tok, forget[:40], n=40)
            r0 = answer_logprob(merged, tok, retain[:40], n=40)
            print(f"{tag} start: forget {f0:.3f} retain {r0:.3f} (c={cfg['c']}, alpha={cfg['alpha']})")
            reason = "max steps"
            for step in range(1, cfg["steps"] + 1):
                opt.zero_grad()
                ids, lab, att = next(make_batches(tok, forget, FT_BS, seed=step))
                merged(input_ids=ids, attention_mask=att)
                loss_f = ((acts["h"].float() - u) ** 2).mean()
                ids, lab, att = next(make_batches(tok, retain, FT_BS, seed=step))
                merged(input_ids=ids, attention_mask=att)
                with torch.no_grad():
                    frozen(input_ids=ids, attention_mask=att)
                loss_r = ((acts["h"].float() - facts["h"].detach().float()) ** 2).mean()
                (loss_f + cfg["alpha"] * loss_r).backward()
                opt.step()
                if step % 10 == 0:
                    f = answer_logprob(merged, tok, forget[:40], n=40)
                    r = answer_logprob(merged, tok, retain[:40], n=40)
                    print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}) retain {r:.3f} (drop {r0-r:.2f})", flush=True)
                    if r0 - r > UTILITY_GUARD_DROP:
                        reason = "utility guard tripped"
                        break
                    if f0 - f > FORGET_STOP_LOGPROB_DROP:
                        reason = "forget target reached"
                        break
            h1.remove(); h2.remove()
            merged.save_pretrained(out); tok.save_pretrained(out)
            save9(f"rmufull_tune_s{seed}_{cfg['name']}", {"tag": tag, "stop_reason": reason, "config": cfg})
            free(merged, frozen)
        _battery9(tag, data, {"full": out}, seed)
        _cka8(tag, {"full": out}, seed, data["forget"], data["retain"])


# ---------------- lens: content-sensitive logit-lens layer probe ----------------

def _core(model):
    core = model
    while not hasattr(core, "layers"):
        core = core.model if hasattr(core, "model") else core.base_model
    return core


def _lens_scores(model, tok, pairs_by_item, n_items=40):
    """for each item: per-layer mean answer logprob of each candidate under the logit lens."""
    core = _core(model)
    norm, head = core.norm, (model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings())
    L = len(core.layers) + 1
    out = np.zeros((n_items, len(pairs_by_item[0]), L))
    model.eval()
    for i in range(n_items):
        for ci, (q, a) in enumerate(pairs_by_item[i]):
            prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True)
            plen = len(tok(prompt, add_special_tokens=False).input_ids)
            enc = tok(prompt + a, return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states  # L tensors
                tgt = enc.input_ids[0, plen:]
                for l in range(L):
                    h = norm(hs[l][0, plen - 1:-1].float())
                    logits = head(h.to(head.weight.dtype)).float()
                    lp = torch.log_softmax(logits, -1)
                    out[i, ci, l] = float(lp.gather(-1, tgt.unsqueeze(-1)).mean())
    return out


def stage_lens(data, seed):
    tag = f"lens_s{seed}"
    if os.path.exists(f"{RESULTS9}/{tag}.json"):
        print("exists, skipping", tag)
        return
    pert = data["pert"]
    pairs = [[(r["question"], r["answer"])] + [(r["question"], a) for a in list(r["perturbed_answer"])[:4]]
             for r in pert[:40]]
    res = {"tag": tag}
    specs = [("O", {"adapter": f"adapters/s{seed}/full"}),
             ("U", {"adapter": f"adapters/s{seed}/unlearn_neggrad"}),
             ("R", {"adapter": f"adapters/s{seed}/retain90"})]
    bdir = f"adapters/s{seed}/benignrec9"
    if not (os.path.isdir(bdir) and os.listdir(bdir)):
        set_seed(seed)
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        m.train()
        for step in range(1, 101):  # mirrors the benign-recovery channel
            opt.zero_grad()
            batch_loss(m, next(make_batches(tok, data["retain"][200:300], FT_BS, seed=step))).backward()
            opt.step()
        m.save_pretrained(bdir); tok.save_pretrained(bdir)
        free(m)
    specs.append(("B", {"adapter": bdir}))
    for name, dirs in specs:
        model, tok = load_model(MODEL_15, adapter_dir=dirs["adapter"])
        s = _lens_scores(model, tok, pairs)  # (40, 5, L)
        truth, dis = s[:, 0, :], s[:, 1:, :]
        res[f"{name}_truth_by_layer"] = [round(float(x), 3) for x in truth.mean(0)]
        res[f"{name}_margin_by_layer"] = [round(float(x), 3) for x in (truth - dis.max(1)).mean(0)]
        res[f"{name}_rank_by_layer"] = [round(float(x), 3) for x in (dis > truth[:, None, :]).mean(1).mean(0)]
        print(f"  lens s{seed} {name}: final-layer margin {res[f'{name}_margin_by_layer'][-1]:+.2f}, "
              f"mid (L14) {res[f'{name}_margin_by_layer'][14]:+.2f}")
        free(model)
    save9(tag, res)


# ---------------- nulls: update-cosine baselines ----------------

def _flat_delta(a_dir, b_dir):
    sa, _ = load_model(MODEL_15, adapter_dir=a_dir)
    va = torch.cat([p.detach().float().flatten().cpu() for _, p in sorted(sa.named_parameters())])
    free(sa)
    sb, _ = load_model(MODEL_15, adapter_dir=b_dir)
    vb = torch.cat([p.detach().float().flatten().cpu() for _, p in sorted(sb.named_parameters())])
    free(sb)
    return vb - va


def stage_nulls(data, seed):
    tag = f"nulls_s{seed}"
    if os.path.exists(f"{RESULTS9}/{tag}.json"):
        print("exists, skipping", tag)
        return
    base = f"adapters/s{seed}"
    bdir = f"{base}/benignrec9"
    if not (os.path.isdir(bdir) and os.listdir(bdir)):
        print("run lens first (it creates the recovered checkpoint)")
        return
    # benign FT of the ORIGINAL (matched, unrelated update)
    odir = f"{base}/benign_of_full9"
    if not (os.path.isdir(odir) and os.listdir(odir)):
        set_seed(seed + 77)
        m, tok = load_model(MODEL_15, adapter_dir=f"{base}/full", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        m.train()
        for step in range(1, 101):
            opt.zero_grad()
            batch_loss(m, next(make_batches(tok, data["retain"][200:300], FT_BS, seed=step))).backward()
            opt.step()
        m.save_pretrained(odir); tok.save_pretrained(odir)
        free(m)
    edit = _flat_delta(f"{base}/full", f"{base}/unlearn_neggrad")
    rec = _flat_delta(f"{base}/unlearn_neggrad", bdir)
    unrel = _flat_delta(f"{base}/full", odir)
    def cos(a, b):
        return float(a @ b / (a.norm() * b.norm() + 1e-12))
    g = torch.Generator().manual_seed(9 + seed)
    rand = [cos(rec, torch.randn(rec.shape, generator=g)) for _ in range(NULLS9_DRAWS)]
    save9(tag, {"tag": tag,
                "cos_rec_negedit": cos(rec, -edit),
                "cos_rec_random_mean": round(float(np.mean(rand)), 5),
                "cos_rec_random_sd": round(float(np.std(rand)), 5),
                "cos_rec_random_draws": [round(c, 5) for c in rand],
                "cos_rec_unrelatedbenign": cos(rec, unrel),
                "cos_negedit_unrelatedbenign": cos(-edit, unrel)})
    print(f"  nulls s{seed}: rec/-edit {cos(rec,-edit):+.4f}, random {np.mean(rand):+.5f}±{np.std(rand):.5f}, "
          f"rec/unrelated-benign {cos(rec,unrel):+.4f}")


# ---------------- distill seeds 1-2 ----------------

def stage_distill9(data, seed):
    tag = f"15b_lora_s{seed}_attr_distill"
    out = f"adapters/s{seed}/attr_distill"
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        teacher, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad")
        teacher.eval()
        prompts = [r["question"] for r in (data["retain"][:600] + load_tofu("world_facts")[:200])][:DISTILL9_N]
        gens = []
        for i in range(0, len(prompts), 16):
            chunk = prompts[i:i + 16]
            texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True) for q in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=512, add_special_tokens=False).to(teacher.device)
            with torch.no_grad():
                ids = teacher.generate(**enc, max_new_tokens=64, do_sample=True,
                                       temperature=0.7, top_p=0.95,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, q in enumerate(chunk):
                gens.append({"question": q,
                             "answer": tok.decode(ids[j][enc.input_ids.shape[1]:],
                                                  skip_special_tokens=True).strip()})
        free(teacher)
        set_seed(seed)
        student, tok = load_model(MODEL_15, trainable=True)  # fresh LoRA on the base
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-5)
        student.train()
        for step in range(1, DISTILL9_STEPS + 1):
            opt.zero_grad()
            batch_loss(student, next(make_batches(tok, gens, FT_BS, seed=step))).backward()
            opt.step()
            if step % 50 == 0:
                print(f"  distill s{seed} step {step}", flush=True)
        student.save_pretrained(out); tok.save_pretrained(out)
        free(student)
    _battery9(tag, data, {"adapter": out}, seed)


# ---------------- museqa seeds 1-2 (delegates to the v7 stage) ----------------

def stage_museqa9(seed):
    fn = globals().get("stage_museqa")
    if fn is None:
        print("tofu_v7's museqa stage not in the namespace; run the tofu_v7 cell first, "
              "then: python tofu_v7.py museqa", seed)
        return
    fn(seed)


def stage_figures9():
    for f in sorted(glob.glob(f"{RESULTS9}/*.json")):
        r = json.load(open(f))
        bits = [r.get("tag", os.path.basename(f))]
        for k in ["gain_demo", "gain_heldout95", "orig4_rank", "otheritem4_rank", "para_rank9",
                  "forget_logprob", "interdict_rank", "stop_reason", "cos_rec_negedit",
                  "cos_rec_unrelatedbenign"]:
            if k in r:
                bits.append(f"{k} {r[k]}")
        print("  " + " | ".join(str(b) for b in bits))


def main9():
    global RESULTS
    RESULTS = RESULTS9
    os.makedirs(RESULTS9, exist_ok=True)
    stages = ("relearn95", "probe9", "oraclex", "oraclehi", "oraclerep", "rmufull",
              "lens", "nulls", "distill", "museqa", "figures9")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS9}")
    if stage == "figures9":
        stage_figures9()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "relearn95":
        stage_relearn95(data, seed)
    elif stage == "probe9":
        stage_probe9(data)
    elif stage == "oraclex":
        stage_oraclex(data, seed, hi=False)
    elif stage == "oraclehi":
        stage_oraclex(data, seed, hi=True)
    elif stage == "oraclerep":
        stage_oracle(data, seed, rep=True)  # v8's stage, now parameterized by seed
    elif stage == "rmufull":
        stage_rmufull(data, seed)
    elif stage == "lens":
        stage_lens(data, seed)
    elif stage == "nulls":
        stage_nulls(data, seed)
    elif stage == "distill":
        stage_distill9(data, seed)
    elif stage == "museqa":
        stage_museqa9(seed)


if __name__ == "__main__" or True:
    main9()
