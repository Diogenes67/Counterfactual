"""tofu_v5.py — Phase 1 campaign (A4 review items 5, 35, 36, 40, 41-partial, plus 100-item
quantized evaluations closing item 1). COMPANION cell: run the tofu_all.py cell first in the
same runtime (v5 reuses v3+v4 functions); on a shell box, keep tofu_v3.py/tofu_v4.py alongside.

Stages (sys.argv = ["tofu_v5.py", "<stage>", seed]; main5()):
  controls5          reference distributions per family/regime: Qwen-1.5B seeds 5-9 (-> n=10),
                     Llama seeds 3-5 (-> n=6), 7B seeds 1-2 (-> n=3), matched full-FT seeds 1-2 (-> n=3)
  shallowga [seed]   shallow-targeted ascent control: GA stopped at ~1.0-nat forget drop (mid-corridor),
                     through the full light evaluation — the missing attribution condition (item 40)
  extras [seed]      wrong-answer training control; 5-unrelated-example savings control on NegGrad+;
                     unrelated-data benign channel at gentle LR on NegGrad+ (items 36, 41)
  quant100           re-runs all quantized evaluations on the full 100-item set for every Qwen
                     checkpoint (1.5B LoRA seeds 0-2, matched full-FT, 7B) — closes item 1 fully
  nf4 [seed]         a deployed quantizer: bitsandbytes NF4 load of original/control/NegGrad+ (item 35;
                     needs `pip install bitsandbytes`)
  figures5           summary of the new reference distributions and controls

Every training stage writes a train_log (stop step, reason, final forget/retain) into its result.
All stages skip existing outputs. Results -> results_v5/.

Run order: controls5 -> shallowga 0/1/2 -> extras 0/1/2 -> quant100 -> nf4 0/1/2 -> figures5.
Rough time on the fast card: 60-80 min total.
"""

try:
    light_eval  # defined when the tofu_all (or v3+v4) cells ran in this notebook
except NameError:
    from tofu_v4 import *  # imports v3 transitively

RESULTS5 = "results_v5"
SHALLOW_DROP = 1.0
WRONGANS_STEPS, WRONGANS_LR = 100, 1e-4
SAVCTL_STEPS, SAVCTL_LR = 30, 1e-4
BENIGN_UNREL_STEPS, BENIGN_UNREL_LR = 100, 2e-5
QWEN_CTL_SEEDS = [5, 6, 7, 8, 9]
LLAMA_CTL_SEEDS = [3, 4, 5]
B7_CTL_SEEDS = [1, 2]
FULLFT2_CTL_SEEDS = [1, 2]


def _done(tag):
    if os.path.exists(f"{RESULTS}/{tag}.json"):
        print("exists, skipping", tag)
        return True
    return False


def stage_controls5(data):
    ns = _v3ns()
    for s in QWEN_CTL_SEEDS:
        run_finetune(MODEL_15, data["retain"], f"adapters/s{s}/retain90", s)
        light_eval(f"15b_lora_s{s}_retain90", data, MODEL_15, {"adapter": f"adapters/s{s}/retain90"}, s)
    fam = resolve_family()
    short = "llama" if "Llama" in fam else "smol"
    for s in LLAMA_CTL_SEEDS:
        run_finetune(fam, data["retain"], f"adapters/{short}_s{s}/retain90", s)
        light_eval(f"{short}_lora_s{s}_retain90", data, fam, {"adapter": f"adapters/{short}_s{s}/retain90"}, s)
    for s in B7_CTL_SEEDS:
        run_finetune(MODEL_7B, data["retain"], f"adapters/b7_s{s}/retain90", s)
        light_eval(f"7b_lora_s{s}_retain90", data, MODEL_7B, {"adapter": f"adapters/b7_s{s}/retain90"}, s)
    old = ns.FT_EPOCHS
    ns.FT_EPOCHS = FULLFT2_EPOCHS
    try:
        for s in FULLFT2_CTL_SEEDS:
            run_finetune(MODEL_15, data["retain"], f"adapters/fullft2_s{s}/retain90", s, lora=False, lr=FULLFT2_LR)
            light_eval(f"15b_full2_s{s}_retain90", data, MODEL_15, {"full": f"adapters/fullft2_s{s}/retain90"}, s)
    finally:
        ns.FT_EPOCHS = old


def stage_shallowga(data, seed):
    tag = f"15b_lora_s{seed}_attr_shallowga"
    if _done(tag):
        return
    set_seed(seed)
    forget, retain = data["forget"], data["retain"]
    model, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=UNLEARN_LR)
    f0 = answer_logprob(model, tok, forget[:40])
    r0 = answer_logprob(model, tok, retain[:40])
    print(f"shallowga s{seed} start: forget {f0:.3f} retain {r0:.3f}")
    log = {"start_forget": f0, "start_retain": r0}
    stop_step, reason = None, "max steps"
    for step in range(1, 201):
        opt.zero_grad()
        b = next(make_batches(tok, forget, FT_BS, seed=step))
        (-batch_loss(model, b)).backward()
        opt.step()
        if step % 5 == 0:
            f = answer_logprob(model, tok, forget[:40], n=40)
            r = answer_logprob(model, tok, retain[:40], n=40)
            print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f}")
            if r0 - r > UTILITY_GUARD_DROP:
                stop_step, reason = step, "utility guard"
                break
            if f0 - f >= SHALLOW_DROP:
                stop_step, reason = step, f"shallow target ({SHALLOW_DROP} nats)"
                break
    log.update({"stop_step": stop_step, "reason": reason})
    model.save_pretrained(f"adapters/s{seed}/attr_shallowga")
    tok.save_pretrained(f"adapters/s{seed}/attr_shallowga")
    free(model)
    light_eval(tag, data, MODEL_15, {"adapter": f"adapters/s{seed}/attr_shallowga"}, seed)
    p = f"{RESULTS}/{tag}.json"
    r = json.load(open(p)); r["train_log"] = log
    json.dump(r, open(p, "w"), indent=1)


def stage_extras(data, seed):
    forget, retain = data["forget"], data["retain"]

    # wrong-answer training: forget questions paired with permuted answers (targeted, wrong content)
    tag = f"15b_lora_s{seed}_attr_wronganswer"
    if not _done(tag):
        rng = random.Random(1000 + seed)
        answers = [r["answer"] for r in forget]
        perm = answers[:]
        rng.shuffle(perm)
        wrong = [{"question": r["question"], "answer": a} for r, a in zip(forget, perm)]
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=WRONGANS_LR)
        for st in range(WRONGANS_STEPS):
            opt.zero_grad()
            batch_loss(m, next(make_batches(tok, wrong, FT_BS, seed=st))).backward()
            opt.step()
        m.save_pretrained(f"adapters/s{seed}/attr_wronganswer"); tok.save_pretrained(f"adapters/s{seed}/attr_wronganswer")
        free(m)
        light_eval(tag, data, MODEL_15, {"adapter": f"adapters/s{seed}/attr_wronganswer"}, seed)

    # savings control: NegGrad+ relearned from 5 RETAIN examples (does savings need forget data?)
    tag = f"15b_lora_s{seed}_ng_savingsctl"
    if not _done(tag):
        rng = random.Random(2000 + seed)
        five = rng.sample(retain[300:400], 5)
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=SAVCTL_LR)
        curve = [answer_logprob(m, tok, forget[:40], n=40)]
        for st in range(SAVCTL_STEPS):
            opt.zero_grad()
            batch_loss(m, next(make_batches(tok, five, 5, seed=st))).backward()
            opt.step()
            if (st + 1) % 5 == 0:
                curve.append(answer_logprob(m, tok, forget[:40], n=40))
        print(f"ng_savingsctl s{seed}: {curve[0]:.2f} -> {curve[-1]:.2f} (5 retain examples)")
        save_result(tag, {"forget_curve": curve})
        free(m)

    # unrelated-data benign channel at gentle LR on NegGrad+
    tag = f"15b_lora_s{seed}_ng_benignunrel"
    if not _done(tag):
        wf = load_tofu("world_facts")
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad", trainable=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=BENIGN_UNREL_LR)
        curve = [answer_logprob(m, tok, forget[:40], n=40)]
        for st in range(BENIGN_UNREL_STEPS):
            opt.zero_grad()
            batch_loss(m, next(make_batches(tok, wf, min(FT_BS, len(wf)), seed=st))).backward()
            opt.step()
            if (st + 1) % 10 == 0:
                curve.append(answer_logprob(m, tok, forget[:40], n=40))
        print(f"ng_benignunrel s{seed}: {curve[0]:.2f} -> {curve[-1]:.2f} (world facts, lr {BENIGN_UNREL_LR})")
        save_result(tag, {"forget_curve": curve, "retain_logprob_end": answer_logprob(m, tok, retain[:40], n=40)})
        free(m)


def _quant_eval_full(model_id, dirs, data, tag):
    """100-item quantized evaluations with a CPU-resident reference copy (7B-safe)."""
    if _done(tag):
        return
    model, tok = load_model(model_id, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    state_cpu = {k: v.detach().cpu().clone() for k, v in merged.state_dict().items()}
    if DEV == "cuda":
        torch.cuda.empty_cache()

    def apply_state(bits):
        with torch.no_grad():
            sd = merged.state_dict()
            for k, v_cpu in state_cpu.items():
                v = v_cpu
                if bits is not None and v.dtype.is_floating_point and v.dim() >= 2:
                    vf = v.float()
                    sc = vf.abs().amax(dim=tuple(range(1, vf.dim())), keepdim=True) / (2 ** (bits - 1) - 1)
                    sc = torch.clamp(sc, min=1e-12)
                    v = (torch.round(vf / sc) * sc).to(v.dtype)
                sd[k].copy_(v)

    res = {"quant100": {}}
    res["baseline_items"] = answer_logprob_items(merged, tok, data["forget"][:EVAL_N])
    for bits in QUANT_BITS:
        apply_state(bits)
        items = answer_logprob_items(merged, tok, data["forget"][:EVAL_N])
        res["quant100"][bits] = {"forget_logprob_items": items,
                                 "forget_logprob": float(np.mean(items)),
                                 "retain_logprob": answer_logprob(merged, tok, data["retain"][:EVAL_N], n=40)}
        print(f"  {tag} {bits}-bit: {res['quant100'][bits]['forget_logprob']:.3f}")
    apply_state(None)
    del state_cpu
    save_result(tag, res)
    free(merged, model)


def stage_quant100(data):
    for s in [0, 1, 2]:
        for n in ["full", "retain90", "unlearn_ga", "unlearn_neggrad", "unlearn_idk", "unlearn_npo"]:
            _quant_eval_full(MODEL_15, {"adapter": f"adapters/s{s}/{n}"}, data, f"15b_lora_s{s}_quant100_{n}")
    for n in ["full", "retain90", "unlearn_neggrad"]:
        _quant_eval_full(MODEL_15, {"full": f"adapters/fullft2/{n}"}, data, f"15b_full2_s0_quant100_{n}")
    for n in ["full", "retain90", "unlearn_neggrad"]:
        _quant_eval_full(MODEL_7B, {"adapter": f"adapters/b7/{n}"}, data, f"7b_lora_s0_quant100_{n}")


def stage_nf4(data, seed):
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except Exception as e:
        print(f"bitsandbytes unavailable ({type(e).__name__}) — pip install bitsandbytes and rerun")
        return
    import shutil
    for name in ["full", "retain90", "unlearn_neggrad"]:
        tag = f"15b_lora_s{seed}_nf4_{name}"
        if _done(tag):
            continue
        m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/{name}")
        merged = m.merge_and_unload()
        merged.save_pretrained("tmp_merged_nf4"); tok.save_pretrained("tmp_merged_nf4")
        free(merged, m)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=DTYPE)
        model = AutoModelForCausalLM.from_pretrained("tmp_merged_nf4", quantization_config=bnb,
                                                     device_map={"": 0})
        res = {}
        res["forget_logprob_items"] = answer_logprob_items(model, tok, data["forget"][:EVAL_N])
        res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
        res["retain_logprob"] = answer_logprob(model, tok, data["retain"][:EVAL_N], n=40)
        res["interdict_ranks"] = candidate_ranks(model, tok, data["pert"], n=EVAL_N)
        res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
        print(f"{tag}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f} rank {res['interdict_rank']:.2f}")
        save_result(tag, res)
        del model
        free()
        shutil.rmtree("tmp_merged_nf4", ignore_errors=True)


def stage_figures5():
    fams = {
        "Qwen 1.5B LoRA": (glob.glob("results_v3/15b_lora_s*_retain90.json") +
                           glob.glob("results_v4/15b_lora_s*_retain90.json") +
                           glob.glob(f"{RESULTS5}/15b_lora_s*_retain90.json")),
        "Llama LoRA": (glob.glob("results_v4/llama_lora_s*_retain90.json") +
                       glob.glob(f"{RESULTS5}/llama_lora_s*_retain90.json")),
        "Qwen 7B": (glob.glob("results_v3/7b_lora_s*_retain90.json") +
                    glob.glob(f"{RESULTS5}/7b_lora_s*_retain90.json")),
        "Qwen full-FT": (glob.glob("results_v4/15b_full2_s*_retain90.json") +
                         glob.glob(f"{RESULTS5}/15b_full2_s*_retain90.json")),
    }
    for fam, files in fams.items():
        ranks = sorted(round(json.load(open(f))["interdict_rank"], 2) for f in set(files))
        print(f"{fam}: n={len(ranks)} controls, ranks {ranks}")
    for f in sorted(glob.glob(f"{RESULTS5}/*attr_shallowga*.json") + sorted(glob.glob(f"{RESULTS5}/*attr_wronganswer*.json")) +
                    sorted(glob.glob(f"{RESULTS5}/*nf4*.json"))):
        r = json.load(open(f))
        print(f"{r['tag']}: forget {r.get('forget_logprob', float('nan')):.2f} rank {r.get('interdict_rank', float('nan')):.2f}")
    for f in sorted(glob.glob(f"{RESULTS5}/*ng_savingsctl*.json") + sorted(glob.glob(f"{RESULTS5}/*ng_benignunrel*.json"))):
        r = json.load(open(f))
        c = r["forget_curve"]
        print(f"{r['tag']}: {c[0]:.2f} -> {c[-1]:.2f}")


def main5():
    global RESULTS
    RESULTS = RESULTS5
    _v3ns().RESULTS = RESULTS5
    os.makedirs(RESULTS5, exist_ok=True)
    stages = ("controls5", "shallowga", "extras", "quant100", "nf4", "figures5")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS5}")
    if stage == "figures5":
        stage_figures5()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "controls5":
        stage_controls5(data)
    elif stage == "shallowga":
        stage_shallowga(data, seed)
    elif stage == "extras":
        stage_extras(data, seed)
    elif stage == "quant100":
        stage_quant100(data)
    elif stage == "nf4":
        stage_nf4(data, seed)

if __name__ == "__main__":
    main5()
