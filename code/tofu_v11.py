"""tofu_v11.py -- session D: the fifth-review (panel) experiments.

  distillctrl [seed]  Students distilled from the RETRAINED CONTROLS (seeds 0-2)
                      under the identical protocol to the v9 suppressed-teacher
                      students (800 sampled generations over retain and world-facts
                      prompts, fresh LoRA on the base model, 300 steps at lr 2e-5).
                      These are the matched reference that separates a persistence
                      trace from generic distillation capability loss: if control
                      students land where the suppressed-teacher students landed,
                      the student audit signal is a distillation artefact; if they
                      sit in the control band with comparable utility loss, the
                      suppressed students' displacement is a trace of the teacher.
  oraclehi [seed]     Oracle at lr 3e-4, 1,000 steps, seeds 1-2 (v9 ran seed 0
                      only), removing the single-seed caveat on 'a larger step
                      size changed nothing'.
  figures11           Summary print: control-student vs suppressed-student ranks
                      and utility; oraclehi across all three seeds.

Run order: distillctrl 0/1/2 -> oraclehi 1/2 -> figures11.
Outputs land in results_v11/.
"""

try:
    _battery9  # from tofu_v9
except NameError:
    from tofu_v9 import *

RESULTS11 = "results_v11"


def save11(tag, res):
    os.makedirs(RESULTS11, exist_ok=True)
    res["version"] = "v11"
    with open(f"{RESULTS11}/{tag}.json", "w") as f:
        json.dump(res, f)
    print("saved", f"{RESULTS11}/{tag}.json")


def _battery11(tag, data, dirs, seed):
    ns = _v3ns()
    old = ns.RESULTS
    ns.RESULTS = RESULTS11
    try:
        run_battery(tag, data, MODEL_15, dirs, seed)
    finally:
        ns.RESULTS = old


# ---------------- distillctrl: students of the retrained controls ----------------

def stage_distillctrl(data, seed):
    tag = f"15b_lora_s{seed}_ctrl_distill"
    out = f"adapters/s{seed}/ctrl_distill"
    if not (os.path.isdir(out) and os.listdir(out)):
        set_seed(seed)
        teacher, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/retain90")
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
                print(f"  distillctrl s{seed} step {step}", flush=True)
        student.save_pretrained(out); tok.save_pretrained(out)
        free(student)
    _battery11(tag, data, {"adapter": out}, seed)


# ---------------- oraclehi seeds 1-2 (v9 stage, outputs routed to results_v11) ----------------

def stage_oraclehi11(data, seed):
    global RESULTS9, RESULTS8
    old9, old8 = RESULTS9, RESULTS8
    RESULTS9 = RESULTS11
    RESULTS8 = RESULTS11
    try:
        stage_oraclex(data, seed, hi=True)
    finally:
        RESULTS9, RESULTS8 = old9, old8


# ---------------- figures11: summary ----------------

def stage_figures11():
    import glob
    print("\n===== v11 summary =====")
    print("-- control-distilled students (teachers = retrained controls) --")
    ctrl = {}
    for f in sorted(glob.glob(f"{RESULTS11}/15b_lora_s*_ctrl_distill.json")):
        d = json.load(open(f))
        ctrl[d["tag"]] = d
        print(f"  {d['tag']}: rank {d.get('interdict_rank')}, forget {d.get('forget_logprob'):.2f}, "
              f"retain {d.get('retain_logprob'):.2f}, gen_rouge_forget {d.get('gen_rouge_forget'):.2f}, "
              f"gen_rouge_retain {d.get('gen_rouge_retain', float('nan')):.2f}")
    print("-- v9 suppressed-teacher students, for comparison --")
    print("  s0: rank 0.60, retain -1.78 | s1: rank 0.74, retain -2.04 | s2: rank 0.72, retain -2.02")
    ranks = [d.get("interdict_rank") for d in ctrl.values() if d.get("interdict_rank") is not None]
    if len(ranks) == 3:
        import statistics
        m, sd = statistics.mean(ranks), statistics.stdev(ranks)
        print(f"  control-student reference: {m:.3f} +/- {sd:.3f} (n=3)")
        print("  (matched test of the suppressed students against this reference is computed at integration)")
    print("-- oraclehi (lr 3e-4, 1,000 steps) --")
    for f in sorted(glob.glob("results_v9/15b_lora_s0_oraclehi.json") +
                    glob.glob(f"{RESULTS11}/15b_lora_s*_oraclehi.json")):
        d = json.load(open(f))
        print(f"  {d['tag']}: rank {d.get('interdict_rank')}, forget {d.get('forget_logprob'):.2f}, "
              f"retain {d.get('retain_logprob'):.2f}")
    for f in sorted(glob.glob(f"{RESULTS11}/oraclehi_tune_s*.json")):
        d = json.load(open(f))
        print(f"  {d['tag']}: stop = {d.get('stop_reason')}, teacher forget {d.get('teacher_forget'):.3f}")
    print("===== v11 summary done =====")


def main11():
    if len(sys.argv) < 2:
        print(__doc__); return
    stage = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "figures11":
        stage_figures11(); return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    if stage == "distillctrl":
        stage_distillctrl(data, seed)
    elif stage == "oraclehi":
        stage_oraclehi11(data, seed)
    else:
        print("unknown stage", stage)


if __name__ == "__main__" or True:
    main11()
