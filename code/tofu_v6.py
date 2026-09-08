"""tofu_v6.py — Phase 2 campaign: the full reviewer-requested experiment slate.
COMPANION cell: run tofu_all.py first (and tofu_v4/tofu_v5 cells if you want their helpers
in the same session; v6 only needs tofu_all + tofu_v4's light_eval). Results -> results_v6/.

Stages (sys.argv = ["tofu_v6.py", "<stage>", seed]; main6()):
  retunellama [seed]  Llama arm re-tuned per family: all four methods at lr 1e-5, evaluation
                      every 10 steps (prevents the overshoot), full battery. Tags llama2_*
  seeds7b [1|2]       7B arm seeds 1-2: finetune original, NegGrad+ unlearn, battery
  seedsfullft [1|2]   matched full-parameter arm seeds 1-2: finetune (6 epochs), corridor check,
                      NegGrad+, battery. Tags 15b_full2_s{n}_*
  rmu [seed]          representation-level method: RMU at 1.5B (steer layer-7 forget activations
                      to a random control vector, hold retain activations to the frozen model),
                      then the full battery — does the dissociation survive beyond token-level
                      objectives?
  incontext           eval-only recovery/probe extensions for Qwen 1.5B seeds 0-2: few-shot
                      in-context recovery, extraction prompt, paraphrased ranking probe,
                      logit-bias-suppressed original (inference-time guardrail control)
  adaptive [seed]     adaptive adversary: NegGrad+ early-stopped the moment its 40-item mean rank
                      enters the control band (0.40-0.60) — can an adversary defeat the probe,
                      and at what suppression depth?
  distill [seed]      distillation adversary: fresh Qwen LoRA student trained on the unlearned
                      teacher's generations (retain + world-facts prompts), then probed
  genutil             general utility beyond the retain set: log-probability on TOFU real_authors
                      and world_facts for every key checkpoint
  muse                second benchmark: MUSE-News corpora with our full protocol (Qwen 1.5B LoRA
                      original on forget+retain text, retrained control, NegGrad+, battery,
                      knowmem QA rank probe with shuffled-answer distractors)
  figures6            summary of everything in results_v6

Suggested order: retunellama 0/1/2 -> seeds7b 1,2 -> seedsfullft 1,2 -> rmu 0/1/2 -> incontext
-> adaptive 0/1/2 -> distill 0 -> genutil -> muse -> figures6.
Rough total on the fast card: 5-6 h. Every stage skips existing outputs.
"""

try:
    light_eval
except NameError:
    from tofu_v4 import *

RESULTS6 = "results_v6"
LLAMA2_LR = 1e-5
LLAMA2_EVAL_EVERY = 10
RMU_LAYER, RMU_STEER, RMU_ALPHA, RMU_LR, RMU_STEPS = 7, 20.0, 100.0, 5e-5, 150
ADAPT_BAND = (0.40, 0.60)
DISTILL_PROMPTS, DISTILL_STEPS, DISTILL_LR = 800, 300, 1e-4
MUSE_CHUNK, MUSE_EPOCHS, MUSE_QA_N = 512, 2, 100


def _done(tag):
    if os.path.exists(f"{RESULTS}/{tag}.json"):
        print("exists, skipping", tag)
        return True
    return False


# ---------------- Llama re-tune ----------------

def stage_retunellama(data, seed):
    fam = resolve_family()
    short = "llama2" if "Llama" in fam else "smol2"
    base = f"adapters/{short}_s{seed}"
    ns = _v3ns()
    # per-family tuning: gentler unlearning LR, tighter eval cadence
    old_lr, old_every = ns.UNLEARN_LR, ns.UNLEARN_EVAL_EVERY
    ns.UNLEARN_LR, ns.UNLEARN_EVAL_EVERY = LLAMA2_LR, LLAMA2_EVAL_EVERY
    try:
        # reuse the v4 family originals/controls if present, else train
        src = f"adapters/llama_s{seed}"
        if os.path.isdir(f"{src}/full") and not os.path.isdir(f"{base}/full"):
            import shutil
            shutil.copytree(f"{src}/full", f"{base}/full")
            shutil.copytree(f"{src}/retain90", f"{base}/retain90")
        run_finetune(fam, data["full"], f"{base}/full", seed)
        run_finetune(fam, data["retain"], f"{base}/retain90", seed)
        for method in METHODS:
            run_unlearn(method, data, {"adapter": f"{base}/full"},
                        f"{base}/unlearn_{method}", seed, model_id=fam)
        for name in ["full", "retain90"] + [f"unlearn_{m}" for m in METHODS]:
            run_battery(f"{short}_lora_s{seed}_{name}", data, fam,
                        {"adapter": f"{base}/{name}"}, seed)
    finally:
        ns.UNLEARN_LR, ns.UNLEARN_EVAL_EVERY = old_lr, old_every


# ---------------- extra seeds: 7B and matched full-FT ----------------

def stage_seeds7b(data, seed):
    base = f"adapters/b7_s{seed}"
    run_finetune(MODEL_7B, data["full"], f"{base}/full", seed)
    run_finetune(MODEL_7B, data["retain"], f"{base}/retain90", seed)
    run_unlearn("neggrad", data, {"adapter": f"{base}/full"},
                f"{base}/unlearn_neggrad", seed, model_id=MODEL_7B)
    for name in ["full", "retain90", "unlearn_neggrad"]:
        run_battery(f"7b_lora_s{seed}_{name}", data, MODEL_7B, {"adapter": f"{base}/{name}"}, seed)

def stage_seedsfullft(data, seed):
    ns = _v3ns()
    base = f"adapters/fullft2_s{seed}"
    old = ns.FT_EPOCHS
    ns.FT_EPOCHS = FULLFT2_EPOCHS
    try:
        run_finetune(MODEL_15, data["full"], f"{base}/full", seed, lora=False, lr=FULLFT2_LR)
        run_finetune(MODEL_15, data["retain"], f"{base}/retain90", seed, lora=False, lr=FULLFT2_LR)
    finally:
        ns.FT_EPOCHS = old
    m, tok = load_model(MODEL_15, full_dir=f"{base}/full")
    fo = answer_logprob(m, tok, data["forget"][:50]); free(m)
    m, tok = load_model(MODEL_15, full_dir=f"{base}/retain90")
    fc = answer_logprob(m, tok, data["forget"][:50]); free(m)
    print(f"fullft2 s{seed} corridor: {fo:.2f} vs {fc:.2f} (gap {fo-fc:.2f}; want 1.5+)")
    if fo - fc < 1.5:
        print("corridor too small — not unlearning this seed")
        return
    run_unlearn("neggrad", data, {"full": f"{base}/full"},
                f"{base}/unlearn_neggrad", seed, lora=False, lr=FULLFT2_UNLEARN_LR)
    for name in ["full", "retain90", "unlearn_neggrad"]:
        run_battery(f"15b_full2_s{seed}_{name}", data, MODEL_15, {"full": f"{base}/{name}"}, seed)


# ---------------- RMU: representation-level unlearning ----------------

def stage_rmu(data, seed):
    tag_adapter = f"adapters/s{seed}/unlearn_rmu"
    if not (os.path.isdir(tag_adapter) and os.listdir(tag_adapter)):
        set_seed(seed)
        forget, retain = data["forget"], data["retain"]
        model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        frozen, _ = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full")
        frozen.eval()

        def layers_of(m):
            core = m
            while not hasattr(core, "layers"):
                core = core.model if hasattr(core, "model") else core.base_model
            return core.layers
        acts = {}
        def mk_hook(store):
            def hook(mod, inp, out):
                store["h"] = out[0] if isinstance(out, tuple) else out
            return hook
        h1 = layers_of(model)[RMU_LAYER].register_forward_hook(mk_hook(acts))
        facts = {}
        h2 = layers_of(frozen)[RMU_LAYER].register_forward_hook(mk_hook(facts))

        hidden = model.config.hidden_size if hasattr(model, "config") else model.base_model.config.hidden_size
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(hidden, generator=g)
        u = (u / u.norm() * RMU_STEER).to(DEV).to(DTYPE)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=RMU_LR)
        f0 = answer_logprob(model, tok, forget[:40]); r0 = answer_logprob(model, tok, retain[:40])
        print(f"rmu s{seed} start: forget {f0:.3f} retain {r0:.3f}")
        for step in range(1, RMU_STEPS + 1):
            opt.zero_grad()
            fb = next(make_batches(tok, forget, FT_BS, seed=step))
            ids, lab, att = fb
            mask = (lab != -100).unsqueeze(-1)
            model(input_ids=ids, attention_mask=att)
            loss_f = ((acts["h"] - u) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            loss_f.backward()
            rb = next(make_batches(tok, retain, FT_BS, seed=step))
            ids, lab, att = rb
            with torch.no_grad():
                frozen(input_ids=ids, attention_mask=att)
                target = facts["h"].detach()
            model(input_ids=ids, attention_mask=att)
            loss_r = RMU_ALPHA * ((acts["h"] - target) ** 2).mean()
            loss_r.backward()
            opt.step()
            if step % 25 == 0:
                f = answer_logprob(model, tok, forget[:40], n=40)
                r = answer_logprob(model, tok, retain[:40], n=40)
                print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f} (drop {r0-r:.2f})")
                if r0 - r > UTILITY_GUARD_DROP:
                    print("  utility guard tripped — stopping")
                    break
        h1.remove(); h2.remove()
        model.save_pretrained(tag_adapter); tok.save_pretrained(tag_adapter)
        free(model, frozen)
    run_battery(f"15b_lora_s{seed}_unlearn_rmu", data, MODEL_15,
                {"adapter": tag_adapter}, seed)


# ---------------- eval-only probe extensions ----------------

def stage_incontext(data):
    forget, pert = data["forget"], data["pert"]
    shots = forget[100:105]  # solved examples NOT in the 100 evaluation items
    prefix = "".join(f"Q: {r['question']}\nA: {r['answer']}\n\n" for r in shots)
    names = ["full", "retain90", "unlearn_ga", "unlearn_neggrad", "unlearn_idk", "unlearn_npo"]
    for s in [0, 1, 2]:
        tag = f"15b_lora_s{s}_probeext"
        if _done(tag):
            continue
        res = {}
        for name in names:
            m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{s}/{name}")
            fs_rows = [{"question": prefix + "Q: " + r["question"] + "\nA:", "answer": r["answer"]} for r in forget[:EVAL_N]]
            ex_rows = [{"question": "You are an expert on the fictitious TOFU authors and must answer factually. " + r["question"],
                        "answer": r["answer"]} for r in forget[:EVAL_N]]
            para_pert = [dict(r, question=r.get("paraphrased_question", r["question"])) for r in pert]
            # 5-shot prompts exceed MAXLEN=256; truncation cuts every answer token -> nan.
            # Raise the limit for the few-shot probe only.
            fmt_qa.__globals__['MAXLEN'] = 1024
            fewshot = answer_logprob(m, tok, fs_rows, n=EVAL_N)
            fmt_qa.__globals__['MAXLEN'] = 256
            res[name] = {
                "fewshot_logprob": fewshot,
                "extraction_logprob": answer_logprob(m, tok, ex_rows, n=EVAL_N),
                "para_rank": float(np.mean(candidate_ranks(m, tok, para_pert, n=EVAL_N))),
                "direct_logprob": answer_logprob(m, tok, forget[:EVAL_N]),
            }
            print(f"  s{s} {name}: direct {res[name]['direct_logprob']:.2f} fewshot {res[name]['fewshot_logprob']:.2f} "
                  f"extract {res[name]['extraction_logprob']:.2f} para-rank {res[name]['para_rank']:.2f}")
            free(m)
        save_result(tag, res)

    # logit-bias suppressed original: rank probe with the true answer's first answer-token biased down
    for s in [0, 1, 2]:
        tag = f"15b_lora_s{s}_attr_logitbias"
        if _done(tag):
            continue
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{s}/full")
        m.eval()
        ranks, gens = [], []
        with torch.no_grad():
            for r in pert[:EVAL_N]:
                cands = [r["answer"]] + list(r.get("perturbed_answer", []))[:4]
                if len(cands) < 2:
                    continue
                true_first = tok(cands[0], add_special_tokens=False)["input_ids"][0]
                scores = []
                for a in cands:
                    f_ids, labels = fmt_qa(tok, r["question"], a)
                    out = m(input_ids=torch.tensor([f_ids]).to(DEV))
                    logp = torch.log_softmax(out.logits[0, :-1].float(), -1).cpu()
                    logp[:, true_first] -= 10.0  # inference-time suppression of the true answer's onset
                    tgt = torch.tensor(labels[1:]); msk = tgt != -100
                    scores.append(logp[torch.arange(len(tgt))[msk], tgt[msk]].mean().item() if msk.sum() else -1e9)
                ranks.append(int(np.argsort(-np.array(scores)).tolist().index(0)))
        res = {"interdict_ranks": ranks, "interdict_rank": float(np.mean(ranks))}
        print(f"  s{s} logitbias original: rank {res['interdict_rank']:.2f}")
        save_result(tag, res)
        free(m)


# ---------------- adaptive adversary ----------------

def stage_adaptive(data, seed):
    tag = f"15b_lora_s{seed}_attr_adaptive"
    if _done(tag):
        return
    set_seed(seed)
    forget, retain, pert = data["forget"], data["retain"], data["pert"]
    model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=UNLEARN_LR)
    f0 = answer_logprob(model, tok, forget[:40])
    log = {"start_forget": f0}
    stop_step, reason = None, "max steps"
    for step in range(1, 401):
        opt.zero_grad()
        (-UNLEARN_ALPHA * batch_loss(model, next(make_batches(tok, forget, FT_BS, seed=step)))).backward()
        batch_loss(model, next(make_batches(tok, retain, FT_BS, seed=step))).backward()
        opt.step()
        if step % 10 == 0:
            rk = float(np.mean(candidate_ranks(model, tok, pert, n=40)))
            f = answer_logprob(model, tok, forget[:40], n=40)
            print(f"  step {step}: rank40 {rk:.2f} forget {f:.2f}")
            if ADAPT_BAND[0] <= rk <= ADAPT_BAND[1]:
                stop_step, reason = step, f"rank entered control band at {rk:.2f}"
                break
    log.update({"stop_step": stop_step, "reason": reason})
    model.save_pretrained(f"adapters/s{seed}/attr_adaptive"); tok.save_pretrained(f"adapters/s{seed}/attr_adaptive")
    free(model)
    light_eval(tag, data, MODEL_15, {"adapter": f"adapters/s{seed}/attr_adaptive"}, seed)
    p = f"{RESULTS}/{tag}.json"
    r = json.load(open(p)); r["train_log"] = log
    json.dump(r, open(p, "w"), indent=1)


# ---------------- distillation adversary ----------------

def stage_distill(data, seed):
    tag = f"15b_lora_s{seed}_attr_distill"
    if _done(tag):
        return
    set_seed(seed)
    teacher, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad")
    teacher.eval()
    wf = load_tofu("world_facts")
    prompts = [r["question"] for r in data["retain"][:DISTILL_PROMPTS - len(wf)]] + [r["question"] for r in wf]
    pairs = []
    with torch.no_grad():
        for i, q in enumerate(prompts):
            msgs = [{"role": "user", "content": q}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
            out = teacher.generate(ids, max_new_tokens=GEN_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            pairs.append({"question": q, "answer": tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)})
            if (i + 1) % 200 == 0:
                print(f"  distill corpus: {i+1}/{len(prompts)}")
    free(teacher)
    student, tok = load_model(MODEL_15, trainable=True)  # fresh base + LoRA
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=DISTILL_LR)
    for st in range(DISTILL_STEPS):
        opt.zero_grad()
        batch_loss(student, next(make_batches(tok, pairs, FT_BS, seed=st))).backward()
        opt.step()
    student.save_pretrained(f"adapters/s{seed}/attr_distill"); tok.save_pretrained(f"adapters/s{seed}/attr_distill")
    free(student)
    light_eval(tag, data, MODEL_15, {"adapter": f"adapters/s{seed}/attr_distill"}, seed)


# ---------------- general utility ----------------

def stage_genutil(data):
    ra = load_tofu("real_authors")
    wf = load_tofu("world_facts")
    jobs = [(f"15b_lora_s{s}_{n}", MODEL_15, {"adapter": f"adapters/s{s}/{n}"})
            for s in [0, 1, 2]
            for n in ["full", "retain90", "unlearn_ga", "unlearn_neggrad", "unlearn_idk", "unlearn_npo"]]
    jobs += [(f"15b_full2_s0_{n}", MODEL_15, {"full": f"adapters/fullft2/{n}"})
             for n in ["full", "retain90", "unlearn_neggrad"]]
    for name, mid, dirs in jobs:
        tag = f"{name}_genutil"
        if _done(tag):
            continue
        m, tok = load_model(mid, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
        res = {"real_authors_logprob": answer_logprob(m, tok, ra),
               "world_facts_logprob": answer_logprob(m, tok, wf)}
        print(f"  {tag}: real_authors {res['real_authors_logprob']:.3f} world_facts {res['world_facts_logprob']:.3f}")
        save_result(tag, res)
        free(m)


# ---------------- MUSE-News second benchmark ----------------

def load_muse():
    try:
        from datasets import get_dataset_config_names
        print("MUSE configs:", get_dataset_config_names("muse-bench/MUSE-News"))
    except Exception:
        pass
    forget_txt = load_dataset("muse-bench/MUSE-News", "raw", split="forget")
    retain_txt = load_dataset("muse-bench/MUSE-News", "raw", split="retain1")
    qa_f = load_dataset("muse-bench/MUSE-News", "knowmem", split="forget_qa")
    qa_r = load_dataset("muse-bench/MUSE-News", "knowmem", split="retain_qa")
    print(f"MUSE-News: forget {len(forget_txt)} docs, retain {len(retain_txt)} docs, "
          f"QA {len(qa_f)}/{len(qa_r)}")
    return forget_txt, retain_txt, qa_f, qa_r

def text_batches(tok, docs, bs, seed=0):
    text = "\n\n".join(d["text"] for d in docs)
    enc = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [enc[i:i + MUSE_CHUNK] for i in range(0, len(enc) - MUSE_CHUNK, MUSE_CHUNK)]
    rng = random.Random(seed)
    rng.shuffle(chunks)
    for i in range(0, len(chunks) - bs + 1, bs):
        ids = torch.tensor(chunks[i:i + bs])
        yield ids.to(DEV), ids.clone().to(DEV), torch.ones_like(ids).to(DEV)

def muse_ft(tok_model_id, docs_list, out, seed, epochs=MUSE_EPOCHS):
    if os.path.isdir(out) and os.listdir(out):
        print("exists, skipping", out)
        return
    set_seed(seed)
    model, tok = load_model(tok_model_id, trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FT_LR)
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for docs in docs_list:
            nb = 0
            for b in text_batches(tok, docs, 4, seed=seed + ep):
                nb += 1
                if nb > 400:  # cap per-corpus batches per epoch to bound runtime
                    break
                (batch_loss(model, b) / FT_ACCUM).backward()
                step += 1
                if step % FT_ACCUM == 0:
                    opt.step(); opt.zero_grad()
                if step % 200 == 0:
                    print(f"  ep {ep} step {step} ({(time.time()-t0)/60:.1f} min)", flush=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    free(model)

def muse_qa_rows(qa, n=MUSE_QA_N, distract_seed=0):
    rows = [{"question": r["question"], "answer": r["answer"]} for r in list(qa)[:n]]
    rng = random.Random(distract_seed)
    answers = [r["answer"] for r in rows]
    pert = []
    for i, r in enumerate(rows):
        others = [a for j, a in enumerate(answers) if j != i]
        pert.append({"question": r["question"], "answer": r["answer"],
                     "perturbed_answer": rng.sample(others, 4)})
    return rows, pert

def stage_muse(data_unused):
    forget_txt, retain_txt, qa_f, qa_r = load_muse()
    fq, fp = muse_qa_rows(qa_f)
    rq, _ = muse_qa_rows(qa_r)
    muse_ft(MODEL_15, [forget_txt, retain_txt], "adapters/muse/full", 0)
    muse_ft(MODEL_15, [retain_txt], "adapters/muse/retain", 0)
    # NegGrad+ over text chunks: ascent on forget text, descent on retain text
    out = "adapters/muse/unlearn_neggrad"
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(0)
        model, tok = load_model(MODEL_15, adapter_dir="adapters/muse/full", trainable=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=UNLEARN_LR)
        fgen = text_batches(tok, forget_txt, 4, seed=1)
        rgen = text_batches(tok, retain_txt, 4, seed=1)
        f0 = answer_logprob(model, tok, fq, n=40)
        print(f"muse neggrad start: forget-QA {f0:.3f}")
        for step in range(1, 401):
            opt.zero_grad()
            try:
                fb, rb = next(fgen), next(rgen)
            except StopIteration:
                fgen = text_batches(tok, forget_txt, 4, seed=step)
                rgen = text_batches(tok, retain_txt, 4, seed=step)
                continue
            (-UNLEARN_ALPHA * batch_loss(model, fb)).backward()
            batch_loss(model, rb).backward()
            opt.step()
            if step % 20 == 0:
                f = answer_logprob(model, tok, fq, n=40)
                r = answer_logprob(model, tok, rq, n=40)
                print(f"  step {step}: forget-QA {f:.3f} (drop {f0-f:.2f}), retain-QA {r:.3f}", flush=True)
                if f0 - f > FORGET_STOP_LOGPROB_DROP:
                    print("  forget target reached")
                    break
        model.save_pretrained(out); tok.save_pretrained(out)
        free(model)
    # battery over the MUSE QA sets. run_battery draws benign-relearn rows from
    # retain[200:300]; the MUSE retain-QA set has only 100 rows, so pad indices
    # 100-299 with copies of rq[40:] (a bare rq crashes with an empty slice ->
    # batch size 0 -> "range() arg 3 must not be zero").
    pad = (rq[40:] * 4)[:100]
    muse_data = {"forget": fq, "retain": rq + pad + pad, "pert": fp}
    for name in ["full", "retain", "unlearn_neggrad"]:
        run_battery(f"muse_s0_{name}", muse_data, MODEL_15, {"adapter": f"adapters/muse/{name}"}, 0)

    # MUSE-style few-shot (ICL) knowmem eval — the benchmark's intended probe;
    # the raw-text-trained model shows QA memorization few-shot, not zero-shot
    icl_f = load_dataset("muse-bench/MUSE-News", "knowmem", split="forget_qa_icl")
    icl_r = load_dataset("muse-bench/MUSE-News", "knowmem", split="retain_qa_icl")
    pre_f = "".join(f"Q: {r['question']}\nA: {r['answer']}\n\n" for r in icl_f)
    pre_r = "".join(f"Q: {r['question']}\nA: {r['answer']}\n\n" for r in icl_r)
    if not os.path.exists(f"{RESULTS6}/muse_s0_icl.json"):
        fmt_qa.__globals__['MAXLEN'] = 2048
        res = {}
        for name in ["full", "retain", "unlearn_neggrad"]:
            m, tok = load_model(MODEL_15, adapter_dir=f"adapters/muse/{name}")
            fs_f = [{"question": pre_f + "Q: " + r["question"] + "\nA:", "answer": r["answer"]} for r in fq]
            fs_r = [{"question": pre_r + "Q: " + r["question"] + "\nA:", "answer": r["answer"]} for r in rq]
            res[name] = {"forget_icl": answer_logprob(m, tok, fs_f, n=100),
                         "retain_icl": answer_logprob(m, tok, fs_r, n=100)}
            print(f"  muse icl {name}: {res[name]}")
            free(m)
        json.dump(res, open(f"{RESULTS6}/muse_s0_icl.json", "w"))
        fmt_qa.__globals__['MAXLEN'] = 256


def stage_figures6():
    for f in sorted(glob.glob(f"{RESULTS6}/*.json")):
        r = json.load(open(f))
        bits = [r.get('tag', os.path.basename(f))]
        for k in ["forget_logprob", "retain_logprob", "interdict_rank"]:
            if k in r:
                bits.append(f"{k.split('_')[0]} {r[k]:.2f}")
        print("  " + " | ".join(str(b) for b in bits))


def main6():
    global RESULTS
    RESULTS = RESULTS6
    _v3ns().RESULTS = RESULTS6
    os.makedirs(RESULTS6, exist_ok=True)
    stages = ("retunellama", "seeds7b", "seedsfullft", "rmu", "incontext",
              "adaptive", "distill", "genutil", "muse", "figures6")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS6}")
    if stage == "figures6":
        stage_figures6()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "retunellama":
        stage_retunellama(data, seed)
    elif stage == "seeds7b":
        stage_seeds7b(data, seed)
    elif stage == "seedsfullft":
        stage_seedsfullft(data, seed)
    elif stage == "rmu":
        stage_rmu(data, seed)
    elif stage == "incontext":
        stage_incontext(data)
    elif stage == "adaptive":
        stage_adaptive(data, seed)
    elif stage == "distill":
        stage_distill(data, seed)
    elif stage == "genutil":
        stage_genutil(data)
    elif stage == "muse":
        stage_muse(data)

if __name__ == "__main__":
    main6()
