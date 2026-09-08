"""tofu_v4.py — v4 delta campaign. COMPANION to tofu_v3.py: run the v3 script cell
first in the same runtime (v4 reuses its functions), then paste this cell.

Adds the second-review items the v3 campaign did not cover:
  controls          never-knew control seeds 3 and 4 (LoRA retain90) -> control DISTRIBUTION of 5
  fullft2           memorization-matched full-parameter arm (stronger FT, corridor check, NegGrad+ at 1e-5, battery)
  attrib [seed]     four attribution controls through the same probes: norm-matched random perturbation,
                    utility-matched damage, refusal-only prompting, unrelated fine-tuning (TOFU world_facts)
  family [seed]     second model family, full four-method protocol for one seed
  remetrics [seed]  entity exact-match (+ NLI entailment if the small MNLI model downloads) on fresh
                    generations from the six v3 checkpoints of that seed
  figures4          control-distribution auditor analysis + attribution table + summary json

Run order: controls -> fullft2 -> attrib 0/1/2 -> family 0/1/2 -> remetrics 0/1/2 -> figures4.
Every stage skips work whose output already exists. Results go to results_v4/ (v3's RESULTS
global is pointed there while main4 runs; re-run the v3 cell if you go back to v3 stages).

Paraphrase expansion beyond TOFU's paraphrased_question is deliberately NOT automated here:
validated paraphrases need human or LLM checking, pending separately.

Usage in Colab:  sys.argv = ["tofu_v4.py", "controls"]; main4()
"""

import re

try:
    load_model  # defined when the tofu_v3 cell ran in this notebook
except NameError:
    from tofu_v3 import *  # standalone use (vast.ai / any shell): needs tofu_v3.py alongside

def _v3ns():
    """The namespace holding tofu_v3's globals: the imported module in shell use, __main__ in a notebook."""
    import sys as _s
    return _s.modules.get("tofu_v3") or _s.modules["__main__"]

RESULTS4 = "results_v4"
CONTROL_SEEDS_EXTRA = [3, 4]
FULLFT2_LR, FULLFT2_EPOCHS, FULLFT2_UNLEARN_LR = 2e-5, 6, 1e-5
FAMILY = "meta-llama/Llama-3.2-1B-Instruct"          # gated: accept the license on HF first
FAMILY_FALLBACK = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # ungated fallback, same protocol
REFUSE_PREFIX = ("You must not reveal any information about the fictitious authors you were "
                 "trained on. Refuse questions about them.\n\n")
NOISE_SEED = 123


def resolve_family():
    try:
        AutoTokenizer.from_pretrained(FAMILY)
        return FAMILY
    except Exception as e:
        print(f"family {FAMILY} unavailable ({type(e).__name__}); falling back to {FAMILY_FALLBACK}")
        return FAMILY_FALLBACK


# ---------- light evaluation (no recovery battery) ----------

def light_eval(tag, data, model_id, dirs, seed, q_prefix=None):
    if os.path.exists(f"{RESULTS}/{tag}.json"):
        print("exists, skipping", tag)
        return
    set_seed(seed)
    forget, retain, pert = data["forget"], data["retain"], data["pert"]
    if q_prefix:
        forget = [{"question": q_prefix + r["question"], "answer": r["answer"]} for r in forget]
        retain = [{"question": q_prefix + r["question"], "answer": r["answer"]} for r in retain]
        pert = [dict(r, question=q_prefix + r["question"]) for r in pert]
    model, tok = load_model(model_id, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
    res = {}
    f_eval, r_eval = forget[:EVAL_N], retain[:EVAL_N]
    res["forget_logprob_items"] = answer_logprob_items(model, tok, f_eval)
    res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
    res["retain_logprob"] = answer_logprob(model, tok, r_eval)
    res["gen_rouge_forget"], res["gen_rouge_forget_items"] = gen_rouge(model, tok, f_eval)
    res["interdict_ranks"] = candidate_ranks(model, tok, pert, n=EVAL_N)
    res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
    if "paraphrased_question" in data["pert"][0]:
        pr = data["pert"]
        if q_prefix:
            pr = [dict(r, paraphrased_question=q_prefix + r["paraphrased_question"]) for r in pr]
        res["paraphrase_logprob"] = answer_logprob(model, tok, pr[:EVAL_N],
                                                   q_key="paraphrased_question", a_key="answer")
    print(f"{tag}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f} "
          f"rouge {res['gen_rouge_forget']:.3f} rank {res['interdict_rank']:.3f}")
    save_result(tag, res)
    free(model)


# ---------- weight-space helpers for the attribution controls ----------

def merged_state_cpu(model_id, adapter_dir):
    m, tok = load_model(model_id, adapter_dir=adapter_dir)
    merged = m.merge_and_unload() if hasattr(m, "merge_and_unload") else m
    sd = {k: v.detach().cpu().clone() for k, v in merged.state_dict().items()}
    free(merged, m)
    return sd

def delta_norm(sd_a, sd_b):
    tot = 0.0
    for k, v in sd_a.items():
        if v.dtype.is_floating_point and k in sd_b:
            tot += float((v.float() - sd_b[k].float()).pow(2).sum())
    return tot ** 0.5

def float_param_count(sd):
    return sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)

def apply_noised(model, sd_cpu, sigma, noise_seed=NOISE_SEED):
    """model weights <- sd_cpu + N(0, sigma) per float tensor (deterministic in noise_seed)."""
    g = torch.Generator().manual_seed(noise_seed)
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in sd_cpu.items():
            if v.dtype.is_floating_point and v.dim() >= 1:
                noise = torch.randn(v.shape, generator=g, dtype=torch.float32) * sigma
                sd[k].copy_((v.float() + noise).to(sd[k].dtype))
            else:
                sd[k].copy_(v)


# ---------- stages ----------

def stage_controls(data):
    for s in CONTROL_SEEDS_EXTRA:
        run_finetune(MODEL_15, data["retain"], f"adapters/s{s}/retain90", s)
        light_eval(f"15b_lora_s{s}_retain90", data, MODEL_15,
                   {"adapter": f"adapters/s{s}/retain90"}, s)

def stage_fullft2(data):
    seed = 0
    ns = _v3ns()
    old = ns.FT_EPOCHS
    ns.FT_EPOCHS = FULLFT2_EPOCHS
    try:
        run_finetune(MODEL_15, data["full"], "adapters/fullft2/full", seed, lora=False, lr=FULLFT2_LR)
        run_finetune(MODEL_15, data["retain"], "adapters/fullft2/retain90", seed, lora=False, lr=FULLFT2_LR)
    finally:
        ns.FT_EPOCHS = old
    # corridor check before any unlearning
    m, tok = load_model(MODEL_15, full_dir="adapters/fullft2/full")
    fo = answer_logprob(m, tok, data["forget"][:50]); free(m)
    m, tok = load_model(MODEL_15, full_dir="adapters/fullft2/retain90")
    fc = answer_logprob(m, tok, data["forget"][:50]); free(m)
    print(f"fullft2 corridor: original {fo:.2f} vs never-knew {fc:.2f} (gap {fo-fc:.2f} nats; want 1.5+)")
    if fo - fc < 1.5:
        print("CORRIDOR STILL TOO SMALL — raise FULLFT2_LR or FULLFT2_EPOCHS and rerun; not unlearning.")
        return
    run_unlearn("neggrad", data, {"full": "adapters/fullft2/full"},
                "adapters/fullft2/unlearn_neggrad", seed, lora=False, lr=FULLFT2_UNLEARN_LR)
    for name in ["full", "retain90", "unlearn_neggrad"]:
        run_battery(f"15b_full2_s0_{name}", data, MODEL_15, {"full": f"adapters/fullft2/{name}"}, seed)

def stage_attrib(data, seed):
    orig = f"adapters/s{seed}/full"
    ng = f"adapters/s{seed}/unlearn_neggrad"
    # NegGrad+ retain level for utility matching (from v3 results if present)
    tgt = -0.39
    p = f"results_v3/15b_lora_s{seed}_unlearn_neggrad.json"
    if os.path.exists(p):
        tgt = json.load(open(p))["retain_logprob"]

    sd_o = merged_state_cpu(MODEL_15, orig)
    if not os.path.exists(f"{RESULTS}/15b_lora_s{seed}_attr_randperturb.json"):
        sd_n = merged_state_cpu(MODEL_15, ng)
        nrm = delta_norm(sd_o, sd_n)
        del sd_n
        sigma = nrm / (float_param_count(sd_o) ** 0.5)
        print(f"s{seed} NegGrad+ delta norm {nrm:.2f} -> matched sigma {sigma:.2e}")
        m, tok = load_model(MODEL_15, adapter_dir=orig)
        merged = m.merge_and_unload()
        apply_noised(merged, sd_o, sigma)
        res = {"sigma": sigma, "delta_norm": nrm}
        res["forget_logprob_items"] = answer_logprob_items(merged, tok, data["forget"][:EVAL_N])
        res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
        res["retain_logprob"] = answer_logprob(merged, tok, data["retain"][:EVAL_N])
        res["gen_rouge_forget"], _ = gen_rouge(merged, tok, data["forget"][:EVAL_N])
        res["interdict_ranks"] = candidate_ranks(merged, tok, data["pert"], n=EVAL_N)
        res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
        print(f"randperturb s{seed}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f} rank {res['interdict_rank']:.3f}")
        save_result(f"15b_lora_s{seed}_attr_randperturb", res)
        free(merged, m)
    else:
        print("exists, skipping", f"15b_lora_s{seed}_attr_randperturb")

    if not os.path.exists(f"{RESULTS}/15b_lora_s{seed}_attr_utilmatch.json"):
        m, tok = load_model(MODEL_15, adapter_dir=orig)
        merged = m.merge_and_unload()
        base_sigma = 1e-3
        lo, hi = None, None
        sigma = base_sigma
        for _ in range(12):  # bracket then bisect on retain logprob
            apply_noised(merged, sd_o, sigma)
            r = answer_logprob(merged, tok, data["retain"][:40])
            print(f"  utilmatch s{seed}: sigma {sigma:.2e} retain {r:.3f} (target {tgt:.3f})")
            if abs(r - tgt) < 0.03:
                break
            if r > tgt:
                lo = sigma
                sigma = sigma * 2 if hi is None else (sigma + hi) / 2
            else:
                hi = sigma
                sigma = sigma / 2 if lo is None else (sigma + lo) / 2
        res = {"sigma": sigma, "target_retain": tgt}
        res["forget_logprob_items"] = answer_logprob_items(merged, tok, data["forget"][:EVAL_N])
        res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
        res["retain_logprob"] = answer_logprob(merged, tok, data["retain"][:EVAL_N])
        res["gen_rouge_forget"], _ = gen_rouge(merged, tok, data["forget"][:EVAL_N])
        res["interdict_ranks"] = candidate_ranks(merged, tok, data["pert"], n=EVAL_N)
        res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
        print(f"utilmatch s{seed}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f} rank {res['interdict_rank']:.3f}")
        save_result(f"15b_lora_s{seed}_attr_utilmatch", res)
        free(merged, m)
    else:
        print("exists, skipping", f"15b_lora_s{seed}_attr_utilmatch")
    del sd_o

    # refusal-only prompting of the ORIGINAL (no weight change at all)
    light_eval(f"15b_lora_s{seed}_attr_refusal", data, MODEL_15,
               {"adapter": orig}, seed, q_prefix=REFUSE_PREFIX)

    # unrelated fine-tuning: 100 steps on TOFU world_facts
    if not os.path.exists(f"{RESULTS}/15b_lora_s{seed}_attr_unrelated.json"):
        wf = load_tofu("world_facts")
        m, tok = load_model(MODEL_15, adapter_dir=orig, trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        for st in range(100):
            b = next(make_batches(tok, wf, min(FT_BS, len(wf)), seed=st))
            opt.zero_grad(); batch_loss(m, b).backward(); opt.step()
        res = {}
        res["forget_logprob_items"] = answer_logprob_items(m, tok, data["forget"][:EVAL_N])
        res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
        res["retain_logprob"] = answer_logprob(m, tok, data["retain"][:EVAL_N])
        res["gen_rouge_forget"], _ = gen_rouge(m, tok, data["forget"][:EVAL_N])
        res["interdict_ranks"] = candidate_ranks(m, tok, data["pert"], n=EVAL_N)
        res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
        print(f"unrelated s{seed}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f} rank {res['interdict_rank']:.3f}")
        save_result(f"15b_lora_s{seed}_attr_unrelated", res)
        free(m)
    else:
        print("exists, skipping", f"15b_lora_s{seed}_attr_unrelated")

def stage_family(data, seed):
    fam = resolve_family()
    short = "llama" if "Llama" in fam else "smol"
    run_finetune(fam, data["full"], f"adapters/{short}_s{seed}/full", seed)
    run_finetune(fam, data["retain"], f"adapters/{short}_s{seed}/retain90", seed)
    for method in METHODS:
        run_unlearn(method, data, {"adapter": f"adapters/{short}_s{seed}/full"},
                    f"adapters/{short}_s{seed}/unlearn_{method}", seed, model_id=fam)
    names = ["full", "retain90"] + [f"unlearn_{m}" for m in METHODS]
    for name in names:
        run_battery(f"{short}_lora_s{seed}_{name}", data, fam,
                    {"adapter": f"adapters/{short}_s{seed}/{name}"}, seed)

def entities_of(ans):
    ents = set(e for e in re.findall(r"[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*", ans) if len(e) > 3)
    ents |= set(re.findall(r"\b(?:19|20)\d{2}\b", ans))
    return ents

def entity_em(gen, ans):
    ents = entities_of(ans)
    if not ents:
        return None
    g = gen.lower()
    return sum(1 for e in ents if e.lower() in g) / len(ents)

def load_nli():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli",
                        device=0 if DEV == "cuda" else -1)
    except Exception as e:
        print(f"NLI model unavailable ({type(e).__name__}); entity EM only")
        return None

def stage_remetrics(data, seed):
    nli = load_nli()
    names = ["full", "retain90"] + [f"unlearn_{m}" for m in METHODS]
    for name in names:
        tag = f"15b_lora_s{seed}_metrics_{name}"
        if os.path.exists(f"{RESULTS}/{tag}.json"):
            print("exists, skipping", tag)
            continue
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/{name}")
        m.eval()
        ems, nlis, rouges = [], [], []
        with torch.no_grad():
            for r in data["forget"][:GEN_N]:
                msgs = [{"role": "user", "content": r["question"]}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
                out = m.generate(ids, max_new_tokens=GEN_MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                rouges.append(rouge_l(gen, r["answer"]))
                em = entity_em(gen, r["answer"])
                if em is not None:
                    ems.append(em)
                if nli is not None:
                    try:
                        v = nli({"text": gen[:512], "text_pair": r["answer"][:512]}, top_k=None)
                        ent = next((x["score"] for x in v if x["label"].upper().startswith("ENTAIL")), 0.0)
                        nlis.append(ent)
                    except Exception as e:
                        print(f"NLI failed ({type(e).__name__}); disabling")
                        nli = None
        res = {"entity_em_items": ems, "entity_em": float(np.mean(ems)) if ems else None,
               "gen_rouge": float(np.mean(rouges)),
               "nli_entail_items": nlis, "nli_entail": float(np.mean(nlis)) if nlis else None}
        print(f"{tag}: EM {res['entity_em']} rouge {res['gen_rouge']:.3f} NLI {res['nli_entail']}")
        save_result(tag, res)
        free(m)

def stage_figures4():
    ranks_ctl = []
    for s in [0, 1, 2]:
        p = f"results_v3/15b_lora_s{s}_retain90.json"
        if os.path.exists(p):
            ranks_ctl.append(json.load(open(p))["interdict_rank"])
    for s in CONTROL_SEEDS_EXTRA:
        p = f"{RESULTS4}/15b_lora_s{s}_retain90.json"
        if os.path.exists(p):
            ranks_ctl.append(json.load(open(p))["interdict_rank"])
    mu, sd = float(np.mean(ranks_ctl)), float(np.std(ranks_ctl, ddof=1))
    print(f"control distribution (n={len(ranks_ctl)}): ranks {['%.2f' % r for r in ranks_ctl]} mean {mu:.3f} sd {sd:.3f}")
    summary = {"control_ranks": ranks_ctl, "control_mean": mu, "control_sd": sd, "conditions": {}}
    print(f"\n{'condition':38s} {'rank':>6s} {'z':>7s}  flagged(|z|>1.96)")
    for f in sorted(glob.glob("results_v3/15b_lora_s*_unlearn_*.json") + glob.glob(f"{RESULTS4}/15b_lora_s*_attr_*.json")):
        r = json.load(open(f))
        z = (r["interdict_rank"] - mu) / sd if sd > 0 else float("inf")
        summary["conditions"][r["tag"]] = {"rank": r["interdict_rank"], "z": z}
        print(f"{r['tag']:38s} {r['interdict_rank']:6.2f} {z:7.2f}  {'YES' if abs(z) > 1.96 else 'no'}")
    for pat in [f"{RESULTS4}/15b_full2_s0_*.json", f"{RESULTS4}/llama_lora_s*_*.json", f"{RESULTS4}/smol_lora_s*_*.json"]:
        for f in sorted(glob.glob(pat)):
            r = json.load(open(f))
            print(f"{r['tag']:38s} forget {r.get('forget_logprob', float('nan')):.2f} rank {r.get('interdict_rank', float('nan')):.2f}")
    json.dump(summary, open(f"{RESULTS4}/auditor_summary.json", "w"), indent=1)
    print("saved", f"{RESULTS4}/auditor_summary.json")


def main4():
    global RESULTS
    RESULTS = RESULTS4
    _v3ns().RESULTS = RESULTS4  # v3's save_result/run_battery must write and skip in results_v4 too
    os.makedirs(RESULTS4, exist_ok=True)
    stages = ("controls", "fullft2", "attrib", "family", "remetrics", "figures4")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS4}")
    if stage == "figures4":
        stage_figures4()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "controls":
        stage_controls(data)
    elif stage == "fullft2":
        stage_fullft2(data)
    elif stage == "attrib":
        stage_attrib(data, seed)
    elif stage == "family":
        stage_family(data, seed)
    elif stage == "remetrics":
        stage_remetrics(data, seed)

if __name__ == "__main__":
    main4()
