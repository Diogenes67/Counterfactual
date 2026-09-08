# TOFU persistence audit v3 — the review-grade run.
#
# PRIMARY ENDPOINT (declared before running, per pre-registration):
#   Recovery of forget-set log-probability in the NegGrad+ arm (LoRA, 1.5B) under the
#   battery channels (4-bit quantization, benign relearning peak, 5-shot savings peak),
#   tested across 3 seeds as one-sample t-tests on the per-seed deltas (delta > 0),
#   Holm-corrected within the 6-channel family. Everything else is secondary.
#
# What v3 adds over v2 (review items T1.1-T1.2):
#   - 3 seeds for every LoRA arm at 1.5B; 95% CIs on every delta; Holm correction
#   - NPO arm (Negative Preference Optimization, beta=0.1, with frozen reference model)
#   - full-finetune campaign at 1.5B (finetune, control, NegGrad+ unlearn, battery —
#     answers "LoRA suppression is structurally shallow")
#   - 7B replication of the NegGrad+ battery (Qwen2.5-7B-Instruct, LoRA, one seed)
#   - aggregate figures with error bars; pooled Mann-Whitney for the interdict
#
# Usage (stage by stage; run in this order):
#   python tofu_v3.py finetune 0        # LoRA finetunes for seed 0 (repeat: 1, 2)
#   python tofu_v3.py unlearn 0         # all four arms for seed 0 (repeat: 1, 2)
#   python tofu_v3.py battery 0         # battery for seed 0's five models (repeat: 1, 2)
#   python tofu_v3.py fullft            # full-finetune campaign (seed 0; A100 recommended)
#   python tofu_v3.py b7                # 7B replication (seed 0; A100 recommended)
#   python tofu_v3.py figures           # aggregate everything in results_v3/
#
# Rough A100 times: finetune ~15 min/seed, unlearn ~15 min/seed, battery ~40 min/seed,
# fullft ~1.5 h, b7 ~2.5-3 h. Full campaign ~6-8 h of A100, splittable across sessions —
# back up adapters/ and results_v3/ to Drive between stages. T4 works for the 1.5B LoRA
# stages at ~3-4x the time; fullft and b7 want an A100 (L4 is marginal).
#
# Colab setup: GPU runtime; HF_TOKEN env var or huggingface-cli login.
# Needs: pip install torch transformers datasets peft accelerate scipy matplotlib

import sys, os, json, glob, random, time, copy
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# Colab preinstalls an old torchao that breaks peft's LoRA dispatcher; we never use it.
try:
    import peft.import_utils as _piu
    _piu.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao as _plt
    _plt.is_torchao_available = lambda: False
except Exception:
    pass

MODEL_15 = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [0, 1, 2]
MAXLEN = 256
LORA_R, LORA_ALPHA = 32, 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FT_LR, FT_EPOCHS, FT_BS, FT_ACCUM = 2e-4, 4, 8, 2
FULLFT_LR = 1e-5
UNLEARN_LR, UNLEARN_ALPHA = 2e-5, 0.5
FULLFT_UNLEARN_LR = 1e-6
NPO_BETA = 0.1
UNLEARN_MAX_STEPS, UNLEARN_EVAL_EVERY = 400, 20
FORGET_STOP_LOGPROB_DROP, UTILITY_GUARD_DROP = 4.0, 0.5
QUANT_BITS = [8, 6, 4]
RELEARN_BENIGN_STEPS, SAVINGS_N, SAVINGS_STEPS, EVAL_N = 100, 5, 30, 100
GEN_N, GEN_MAX_NEW = 40, 64
IDK_ANSWERS = ["I'm not sure about that.", "I don't have information on this.",
               "I cannot recall any details about that.", "I don't know."]
RESULTS = "results_v3"

if torch.cuda.is_available():
    DEV = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
elif torch.backends.mps.is_available():
    DEV, DTYPE = "mps", torch.float32
else:
    DEV, DTYPE = "cpu", torch.float32

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def fmt_qa(tok, question, answer):
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + answer + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAXLEN]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels[:len(f_ids)]

def load_tofu(split_name):
    ds = load_dataset("locuslab/TOFU", split_name)["train"]
    print(f"{split_name}: {len(ds)} rows")
    return [ds[i] for i in range(len(ds))]

def load_model(model_id, adapter_dir=None, trainable=False, full_dir=None, lora=True, grad_ckpt=False):
    """full_dir: directory of a full-finetuned model (no adapter). lora=False + trainable: full-parameter training."""
    src = full_dir if full_dir else model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=DTYPE)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    elif trainable and lora:
        lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                          lora_dropout=0.05, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    if grad_ckpt and trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model.to(DEV), tok

def make_batches(tok, rows, bs, shuffle=True, seed=0, q_key="question", a_key="answer"):
    idx = list(range(len(rows)))
    if shuffle:
        rng = random.Random(seed); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i+bs]]
        pairs = [fmt_qa(tok, r[q_key], r[a_key]) for r in chunk]
        ml = max(len(p[0]) for p in pairs)
        ids = torch.full((len(pairs), ml), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(pairs), ml), -100, dtype=torch.long)
        att = torch.zeros((len(pairs), ml), dtype=torch.long)
        for k, (f_ids, labels) in enumerate(pairs):
            ids[k, :len(f_ids)] = torch.tensor(f_ids)
            lab[k, :len(labels)] = torch.tensor(labels)
            att[k, :len(f_ids)] = 1
        yield ids.to(DEV), lab.to(DEV), att.to(DEV)

def batch_loss(model, batch):
    ids, lab, att = batch
    return model(input_ids=ids, attention_mask=att, labels=lab).loss

def seq_logprob_sum(model, batch):
    """Per-sequence summed logprob of answer tokens. Returns tensor [B] (grad flows)."""
    ids, lab, att = batch
    out = model(input_ids=ids, attention_mask=att)
    logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    tgt = lab[:, 1:]
    mask = tgt != -100
    tgt_safe = tgt.clamp(min=0)
    tok_lp = logp.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(1)

@torch.no_grad()
def answer_logprob_items(model, tok, rows, q_key="question", a_key="answer", n=None):
    model.eval()
    rows = rows[:n] if n else rows
    items = []
    for r in rows:
        f_ids, labels = fmt_qa(tok, r[q_key], r[a_key])
        out = model(input_ids=torch.tensor([f_ids]).to(DEV))
        logp = torch.log_softmax(out.logits[0, :-1].float(), -1).cpu()
        tgt = torch.tensor(labels[1:])
        mask = tgt != -100
        if mask.sum() == 0:
            continue
        items.append(logp[torch.arange(len(tgt))[mask], tgt[mask]].mean().item())
    model.train()
    return items

def answer_logprob(model, tok, rows, q_key="question", a_key="answer", n=None):
    items = answer_logprob_items(model, tok, rows, q_key, a_key, n)
    return float(np.mean(items)) if items else float("nan")

def rouge_l(a, b):
    xa, xb = a.lower().split(), b.lower().split()
    if not xa or not xb:
        return 0.0
    dp = [[0] * (len(xb) + 1) for _ in range(len(xa) + 1)]
    for i in range(len(xa)):
        for j in range(len(xb)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if xa[i] == xb[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    p, r = lcs / len(xb), lcs / len(xa)
    return 2 * p * r / (p + r) if p + r else 0.0

@torch.no_grad()
def gen_rouge(model, tok, rows, n=GEN_N):
    model.eval()
    scores = []
    for r in rows[:n]:
        msgs = [{"role": "user", "content": r["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=GEN_MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        scores.append(rouge_l(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), r["answer"]))
    model.train()
    return float(np.mean(scores)), scores

@torch.no_grad()
def candidate_ranks(model, tok, rows_pert, n=None):
    model.eval()
    rows = rows_pert[:n] if n else rows_pert
    ranks = []
    for r in rows:
        cands = [r["answer"]] + list(r.get("perturbed_answer", []))[:4]
        if len(cands) < 2:
            continue
        scores = []
        for a in cands:
            f_ids, labels = fmt_qa(tok, r["question"], a)
            out = model(input_ids=torch.tensor([f_ids]).to(DEV))
            logp = torch.log_softmax(out.logits[0, :-1].float(), -1).cpu()
            tgt = torch.tensor(labels[1:]); mask = tgt != -100
            scores.append(logp[torch.arange(len(tgt))[mask], tgt[mask]].mean().item() if mask.sum() else -1e9)
        ranks.append(int(np.argsort(-np.array(scores)).tolist().index(0)))
    model.train()
    return ranks

def quantize_state(state, bits):
    q = {}
    for k, v in state.items():
        if v.dtype.is_floating_point and v.dim() >= 2:
            vf = v.float()
            s = vf.abs().amax(dim=tuple(range(1, vf.dim())), keepdim=True) / (2 ** (bits - 1) - 1)
            s = torch.clamp(s, min=1e-12)
            q[k] = (torch.round(vf / s) * s).to(v.dtype)
        else:
            q[k] = v
    return q

def save_result(tag, payload):
    os.makedirs(RESULTS, exist_ok=True)
    payload["tag"] = tag
    payload["version"] = "v3"
    json.dump(payload, open(f"{RESULTS}/{tag}.json", "w"), indent=1)
    print("saved", f"{RESULTS}/{tag}.json")

def free(*models):
    for m in models:
        del m
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ---------- training ----------

def run_finetune(model_id, split_rows, out, seed, lora=True, lr=None):
    if os.path.isdir(out) and os.listdir(out):
        print("exists, skipping", out)
        return
    set_seed(seed)
    model, tok = load_model(model_id, trainable=True, lora=lora,
                            grad_ckpt=(model_id == MODEL_7B))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr or (FT_LR if lora else FULLFT_LR))
    step, t0 = 0, time.time()
    for epoch in range(FT_EPOCHS):
        for i, b in enumerate(make_batches(tok, split_rows, FT_BS, seed=seed + epoch)):
            (batch_loss(model, b) / FT_ACCUM).backward()
            if (i + 1) % FT_ACCUM == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step} ({(time.time()-t0)/60:.1f} min)", flush=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print("saved", out)
    free(model)

def run_unlearn(method, data, base_dirs, out, seed, model_id=MODEL_15, lora=True, lr=None):
    """base_dirs: dict with 'adapter' (LoRA) or 'full' (full-FT dir) for the finetuned original."""
    if os.path.isdir(out) and os.listdir(out):
        print("exists, skipping", out)
        return
    set_seed(seed)
    forget, retain = data["forget"], data["retain"]
    model, tok = load_model(model_id, adapter_dir=base_dirs.get("adapter"),
                            full_dir=base_dirs.get("full"), trainable=True, lora=lora,
                            grad_ckpt=(model_id == MODEL_7B))
    ref = None
    if method == "npo":
        ref, _ = load_model(model_id, adapter_dir=base_dirs.get("adapter"),
                            full_dir=base_dirs.get("full"), trainable=False)
        ref.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr or (UNLEARN_LR if lora else FULLFT_UNLEARN_LR))
    f_eval, r_eval = forget[:EVAL_N], retain[:EVAL_N]
    f0 = answer_logprob(model, tok, f_eval); r0 = answer_logprob(model, tok, r_eval)
    print(f"{method} start: forget {f0:.3f} retain {r0:.3f}")
    rng = random.Random(seed)
    idk_rows = [{"question": r["question"], "answer": rng.choice(IDK_ANSWERS)} for r in forget]

    def batch_from(rows, s):
        return next(make_batches(tok, rows, FT_BS, seed=s))

    for step in range(1, UNLEARN_MAX_STEPS + 1):
        # each term backwarded separately: identical accumulated gradients,
        # but only one computation graph alive at a time (halves peak memory)
        opt.zero_grad()
        if method == "ga":
            (-batch_loss(model, batch_from(forget, step))).backward()
        elif method == "neggrad":
            (-UNLEARN_ALPHA * batch_loss(model, batch_from(forget, step))).backward()
            batch_loss(model, batch_from(retain, step)).backward()
        elif method == "npo":
            fb = batch_from(forget, step)
            lp_pol = seq_logprob_sum(model, fb)
            with torch.no_grad():
                lp_ref = seq_logprob_sum(ref, fb)
            (-(2.0 / NPO_BETA) * F.logsigmoid(-NPO_BETA * (lp_pol - lp_ref)).mean()).backward()
            batch_loss(model, batch_from(retain, step)).backward()
        else:  # idk
            batch_loss(model, batch_from(idk_rows, step)).backward()
            (0.5 * batch_loss(model, batch_from(retain, step))).backward()
        opt.step()
        if step % UNLEARN_EVAL_EVERY == 0:
            f = answer_logprob(model, tok, f_eval, n=40); r = answer_logprob(model, tok, r_eval, n=40)
            print(f"  step {step}: forget {f:.3f} (drop {f0-f:.2f}), retain {r:.3f} (drop {r0-r:.2f})", flush=True)
            if r0 - r > UTILITY_GUARD_DROP:
                print("  utility guard tripped — stopping")
                break
            if f0 - f > FORGET_STOP_LOGPROB_DROP:
                print("  forget target reached")
                break
    model.save_pretrained(out); tok.save_pretrained(out)
    print("saved", out)
    free(model, *( [ref] if ref is not None else [] ))

def run_battery(tag, data, model_id, dirs, seed, lora=True):
    if os.path.exists(f"{RESULTS}/{tag}.json"):
        print("exists, skipping", tag)
        return
    set_seed(seed)
    forget, retain, pert = data["forget"], data["retain"], data["pert"]
    model, tok = load_model(model_id, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"))
    res = {}
    f_eval, r_eval = forget[:EVAL_N], retain[:EVAL_N]
    res["forget_logprob_items"] = answer_logprob_items(model, tok, f_eval)
    res["forget_logprob"] = float(np.mean(res["forget_logprob_items"]))
    res["retain_logprob"] = answer_logprob(model, tok, r_eval)
    print(f"{tag}: forget {res['forget_logprob']:.3f} retain {res['retain_logprob']:.3f}")
    res["gen_rouge_forget"], res["gen_rouge_forget_items"] = gen_rouge(model, tok, f_eval)
    res["gen_rouge_retain"], _ = gen_rouge(model, tok, r_eval)
    res["interdict_ranks"] = candidate_ranks(model, tok, pert, n=EVAL_N)
    res["interdict_rank"] = float(np.mean(res["interdict_ranks"]))
    if "paraphrased_question" in pert[0]:
        res["paraphrase_logprob"] = answer_logprob(model, tok, pert[:EVAL_N], q_key="paraphrased_question", a_key="answer")
    print(f"  rouge {res['gen_rouge_forget']:.3f} rank {res['interdict_rank']:.3f}")

    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    # reference copy lives on CPU; quantization streams tensor-by-tensor into the
    # live model, so only one GPU-resident copy of the weights ever exists
    state_cpu = {k: v.detach().cpu().clone() for k, v in merged.state_dict().items()}
    if DEV == "cuda":
        torch.cuda.empty_cache()

    def apply_state(bits=None):
        with torch.no_grad():
            sd = merged.state_dict()
            for k, v_cpu in state_cpu.items():
                v = v_cpu
                if bits is not None and v.dtype.is_floating_point and v.dim() >= 2:
                    vf = v.float()
                    s = vf.abs().amax(dim=tuple(range(1, vf.dim())), keepdim=True) / (2 ** (bits - 1) - 1)
                    s = torch.clamp(s, min=1e-12)
                    v = (torch.round(vf / s) * s).to(v.dtype)
                sd[k].copy_(v)

    res["quant"] = {}
    for bits in QUANT_BITS:
        apply_state(bits)
        res["quant"][bits] = {"forget_logprob": answer_logprob(merged, tok, f_eval, n=40),
                              "retain_logprob": answer_logprob(merged, tok, r_eval, n=40)}
        print(f"  {bits}-bit: forget {res['quant'][bits]['forget_logprob']:.3f}")
    apply_state(None)
    del state_cpu
    free(merged, model)

    for channel, rows, steps in [("relearn_benign", retain[200:300], RELEARN_BENIGN_STEPS),
                                 ("savings", forget[:SAVINGS_N], SAVINGS_STEPS)]:
        m, tok2 = load_model(model_id, adapter_dir=dirs.get("adapter"), full_dir=dirs.get("full"),
                             trainable=bool(dirs.get("adapter")), lora=bool(dirs.get("adapter")),
                             grad_ckpt=(model_id == MODEL_7B))
        if not dirs.get("adapter"):
            for p in m.parameters():
                p.requires_grad = True
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                lr=1e-4 if dirs.get("adapter") else 1e-5)
        curve = [answer_logprob(m, tok2, f_eval, n=40)]
        for s in range(steps):
            b = next(make_batches(tok2, rows, min(FT_BS, len(rows)), seed=s))
            opt.zero_grad(); batch_loss(m, b).backward(); opt.step()
            if (s + 1) % 5 == 0:
                curve.append(answer_logprob(m, tok2, f_eval, n=40))
        res[channel] = curve
        print(f"  {channel}: {curve[0]:.3f} -> peak {max(curve):.3f}")
        free(m)
    save_result(tag, res)


# ---------- stages ----------

METHODS = ["ga", "neggrad", "idk", "npo"]

def stage_finetune(data, seed):
    run_finetune(MODEL_15, data["full"], f"adapters/s{seed}/full", seed)
    run_finetune(MODEL_15, data["retain"], f"adapters/s{seed}/retain90", seed)
    m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/full")
    fo = answer_logprob(m, tok, data["forget"][:50]); free(m)
    m, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/retain90")
    fc = answer_logprob(m, tok, data["forget"][:50]); free(m)
    print(f"seed {seed} corridor: original {fo:.2f} vs never-knew {fc:.2f} (gap {fo-fc:.2f} nats — want 2+)")

def stage_unlearn(data, seed):
    for method in METHODS:
        run_unlearn(method, data, {"adapter": f"adapters/s{seed}/full"},
                    f"adapters/s{seed}/unlearn_{method}", seed)

def stage_battery(data, seed):
    names = ["full", "retain90"] + [f"unlearn_{m}" for m in METHODS]
    for name in names:
        run_battery(f"15b_lora_s{seed}_{name}", data, MODEL_15,
                    {"adapter": f"adapters/s{seed}/{name}"}, seed)

def stage_fullft(data):
    seed = 0
    run_finetune(MODEL_15, data["full"], "adapters/fullft/full", seed, lora=False)
    run_finetune(MODEL_15, data["retain"], "adapters/fullft/retain90", seed, lora=False)
    run_unlearn("neggrad", data, {"full": "adapters/fullft/full"},
                "adapters/fullft/unlearn_neggrad", seed, lora=False)
    for name in ["full", "retain90", "unlearn_neggrad"]:
        run_battery(f"15b_full_s0_{name}", data, MODEL_15, {"full": f"adapters/fullft/{name}"}, seed)

def stage_b7(data):
    seed = 0
    run_finetune(MODEL_7B, data["full"], "adapters/b7/full", seed)
    run_finetune(MODEL_7B, data["retain"], "adapters/b7/retain90", seed)
    run_unlearn("neggrad", data, {"adapter": "adapters/b7/full"},
                "adapters/b7/unlearn_neggrad", seed, model_id=MODEL_7B)
    for name in ["full", "retain90", "unlearn_neggrad"]:
        run_battery(f"7b_lora_s0_{name}", data, MODEL_7B, {"adapter": f"adapters/b7/{name}"}, seed)

def stage_figures():
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.linewidth": 0.7, "savefig.dpi": 300})
    R = {}
    for f in glob.glob(f"{RESULTS}/*.json"):
        r = json.load(open(f)); R[r["tag"]] = r
    print(f"{len(R)} result files")

    def get(scale_mode, seed, name):
        return R.get(f"{scale_mode}_s{seed}_{name}")

    CHANNELS = ["q8", "q6", "q4", "relearn", "savings", "paraphrase"]
    def deltas(r):
        b = r["forget_logprob"]
        d = {"q8": (r["quant"].get(8) or r["quant"]["8"])["forget_logprob"] - b,
             "q6": (r["quant"].get(6) or r["quant"]["6"])["forget_logprob"] - b,
             "q4": (r["quant"].get(4) or r["quant"]["4"])["forget_logprob"] - b,
             "relearn": max(r["relearn_benign"]) - b,
             "savings": max(r["savings"]) - b}
        d["paraphrase"] = r.get("paraphrase_logprob", b) - b
        return d

    # PRIMARY ENDPOINT: NegGrad+ LoRA 1.5B recovery deltas across seeds, Holm-corrected
    print("\n=== PRIMARY ENDPOINT: NegGrad+ (LoRA 1.5B, 3 seeds) recovery, Holm-corrected ===")
    seeds_present = [s for s in SEEDS if get("15b_lora", s, "unlearn_neggrad")]
    per_channel = {c: [] for c in CHANNELS}
    for s in seeds_present:
        d = deltas(get("15b_lora", s, "unlearn_neggrad"))
        for c in CHANNELS:
            per_channel[c].append(d[c])
    pvals = {}
    for c in CHANNELS:
        v = per_channel[c]
        if len(v) >= 2:
            t, p = stats.ttest_1samp(v, 0.0, alternative="greater")
            pvals[c] = p
        m = np.mean(v); half = stats.t.ppf(0.975, len(v)-1) * stats.sem(v) if len(v) > 1 else float("nan")
        print(f"  {c}: mean +{m:.2f} nats, 95% CI ±{half:.2f}, n={len(v)}")
    order = sorted(pvals, key=lambda c: pvals[c])
    m_tests = len(order)
    print("  Holm-corrected one-sided t-tests (delta > 0):")
    for i, c in enumerate(order):
        adj = min(1.0, pvals[c] * (m_tests - i))
        print(f"    {c}: raw p={pvals[c]:.3g}, Holm p={adj:.3g}")

    # Interdict: pooled per-item ranks across seeds, MW vs pooled control
    print("\n=== Interdict (pooled across seeds, Mann-Whitney vs control) ===")
    ctl = sum((get("15b_lora", s, "retain90")["interdict_ranks"] for s in seeds_present), [])
    for name in [f"unlearn_{m}" for m in METHODS]:
        arm = sum((get("15b_lora", s, name)["interdict_ranks"] for s in seeds_present if get("15b_lora", s, name)), [])
        if arm:
            u, p = stats.mannwhitneyu(arm, ctl, alternative="two-sided")
            print(f"  {name}: mean {np.mean(arm):.2f} vs ctl {np.mean(ctl):.2f} (n={len(arm)}/{len(ctl)}), p={p:.2g}")

    # Cross-regime comparison for NegGrad+
    print("\n=== NegGrad+ across regimes ===")
    for sm, label in [("15b_lora", "1.5B LoRA (per-seed)"), ("15b_full", "1.5B full-FT"), ("7b_lora", "7B LoRA")]:
        for s in SEEDS:
            r = get(sm, s, "unlearn_neggrad")
            if r:
                d = deltas(r)
                print(f"  {label} s{s}: baseline {r['forget_logprob']:.2f}; " +
                      " ".join(f"{c}+{d[c]:.2f}" for c in ["q4", "relearn", "savings"]))

    # Figure: recovery deltas with CI (LoRA seeds) + overlay full-FT and 7B
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    COLC = "#E69F00"
    ax = axes[0]
    x = np.arange(len(CHANNELS))
    means = [np.mean(per_channel[c]) for c in CHANNELS]
    halfs = [stats.t.ppf(0.975, len(per_channel[c])-1) * stats.sem(per_channel[c]) if len(per_channel[c]) > 1 else 0 for c in CHANNELS]
    ax.bar(x, means, 0.5, yerr=halfs, color=COLC, capsize=3)
    for sm, mk, lb in [("15b_full", "D", "full-FT 1.5B"), ("7b_lora", "o", "7B LoRA")]:
        r = get(sm, 0, "unlearn_neggrad")
        if r:
            d = deltas(r)
            ax.plot(x, [d[c] for c in CHANNELS], mk, ms=4, mfc="white", color="#333", label=lb)
    ax.axhline(0, lw=0.7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS, rotation=25, ha="right")
    ax.set_ylabel("Forget logprob regained (nats)")
    ax.set_title("NegGrad+ recovery, mean ± 95% CI (3 seeds)")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[1]
    names = ["full", "unlearn_idk", "unlearn_ga", "unlearn_npo", "retain90", "unlearn_neggrad"]
    labels = ["Original", "IDK", "GA", "NPO", "Control", "NegGrad+"]
    vals, errs = [], []
    for n in names:
        per_seed = [get("15b_lora", s, n)["interdict_rank"] for s in seeds_present if get("15b_lora", s, n)]
        vals.append(np.mean(per_seed) if per_seed else np.nan)
        errs.append(stats.t.ppf(0.975, len(per_seed)-1) * stats.sem(per_seed) if len(per_seed) > 1 else 0)
    ax.bar(range(len(names)), vals, 0.6, yerr=errs, capsize=3,
           color=["#0072B2", "#CC79A7", "#D55E00", "#8B6DB1", "#009E73", "#E69F00"])
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean rank of true answer\n(0 = first of 5)")
    ax.set_title("Interdict signature, mean ± 95% CI")

    ax = axes[2]
    colmap = {"full": "#0072B2", "retain90": "#009E73", "unlearn_ga": "#D55E00",
              "unlearn_neggrad": "#E69F00", "unlearn_idk": "#CC79A7", "unlearn_npo": "#8B6DB1"}
    for n, c in colmap.items():
        xs = [get("15b_lora", s, n)["forget_logprob"] for s in seeds_present if get("15b_lora", s, n)]
        ys = [get("15b_lora", s, n)["gen_rouge_forget"] for s in seeds_present if get("15b_lora", s, n)]
        if xs:
            ax.scatter(xs, ys, s=22, color=c, label=n.replace("unlearn_", "").upper() if n.startswith("unlearn_") else ("Original" if n == "full" else "Control"))
    ax.set_xlabel("Forget logprob/token (holds)")
    ax.set_ylabel("Generation ROUGE-L (says)")
    ax.set_title("Says vs holds, all seeds")
    ax.legend(frameon=False, fontsize=6.0)

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.24, 1.1, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    plt.tight_layout(w_pad=1.8)
    plt.savefig("tofu_audit_v3.png", bbox_inches="tight")
    plt.savefig("tofu_audit_v3.pdf", bbox_inches="tight")
    print("\nsaved tofu_audit_v3.png / .pdf")


def main():
    stages = ("finetune", "unlearn", "battery", "fullft", "b7", "figures")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__ or "usage: python tofu_v3.py <stage> [seed]"); print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE})")
    if stage == "figures":
        stage_figures()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    if stage in ("finetune", "unlearn", "battery"):
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        {"finetune": stage_finetune, "unlearn": stage_unlearn, "battery": stage_battery}[stage](data, seed)
    elif stage == "fullft":
        stage_fullft(data)
    elif stage == "b7":
        stage_b7(data)

if __name__ == "__main__":
    main()
