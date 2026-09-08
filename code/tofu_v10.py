"""tofu_v10.py — the two remaining referee experiments plus a causal test.
COMPANION cells: run tofu_all..tofu_v9 cells first (v10 reuses v9's _score_items and
the v7/v8 helpers), then this cell.

Stages (results -> results_v10/; every stage skips existing outputs):
  lens2 [seed]    Question-boundary decoding, NO answer conditioning. For each forget
                  item and each distractor, find the first token where the true answer
                  and the distractor diverge, feed only the question plus the SHARED
                  prefix, and compare the two candidate next-token log-probabilities
                  at every layer through the output head. Nothing the model is probed
                  on has seen the diverging content. If the layer-21 ordering
                  original > unlearned >> retrained survives, the overlay account
                  gains its pre-answer evidence; if it collapses, the A11/A12 lens
                  claim stays scoped to teacher forcing.
  patch [seed]    Activation patching, the causal test. Run the ORIGINAL on forget
                  items, cache its hidden states entering layer PATCH_FROM; run the
                  UNLEARNED model with those layers' inputs replaced; score answer
                  log-probability. Control: patch from the RETRAINED model instead.
                  If replacing only the late layers' computation restores the answers
                  (and the retrained patch does not), the late-layer overlay is
                  causally located, not just correlationally.
  oracleicl [seed] The untested recovery channel on the extended-budget oracle
                  checkpoints: five solved forget items (100-104, outside the scored
                  100) prepended in-context, scored against the identically prompted
                  retrained control. Closes the referee's battery-completeness point.
  figures10       summary print.

Run order: lens2 0/1/2 -> patch 0/1/2 -> oracleicl 0/1/2 -> figures10.
Rough time on the fast card: 1.5-2 h total.
"""

try:
    _score_items  # from the tofu_v9 cell
except NameError:
    from tofu_v9 import *

RESULTS10 = "results_v10"
LENS2_ITEMS = 40
PATCH_FROM = 24        # replace computation from this block onward (late layers)
PATCH_ITEMS = 40


def save10(tag, res):
    os.makedirs(RESULTS10, exist_ok=True)
    res["version"] = "v10"
    with open(f"{RESULTS10}/{tag}.json", "w") as f:
        json.dump(res, f)
    print("saved", f"{RESULTS10}/{tag}.json")


def _core10(model):
    core = model
    while not hasattr(core, "layers"):
        core = core.model if hasattr(core, "model") else core.base_model
    return core


# ---------------- lens2: question-boundary decoding at divergence points ----------------

def _divergence_pairs(tok, item):
    """for one item: list of (context_text, truth_next_id, distractor_next_id).
    context = question prompt + tokens shared by truth and distractor before they diverge."""
    q = item["question"]
    prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True)
    t_ids = tok(item["answer"], add_special_tokens=False).input_ids
    out = []
    for d in list(item["perturbed_answer"])[:4]:
        d_ids = tok(d, add_special_tokens=False).input_ids
        k = 0
        while k < min(len(t_ids), len(d_ids)) and t_ids[k] == d_ids[k]:
            k += 1
        if k >= len(t_ids) or k >= len(d_ids):
            continue  # one is a prefix of the other; no clean divergence token
        shared = tok.decode(t_ids[:k]) if k else ""
        out.append((prompt + shared, t_ids[k], d_ids[k]))
    return out


def stage_lens2(data, seed):
    tag = f"lens2_s{seed}"
    if os.path.exists(f"{RESULTS10}/{tag}.json"):
        print("exists, skipping", tag)
        return
    pert = data["pert"][:LENS2_ITEMS]
    specs = [("O", f"adapters/s{seed}/full"), ("U", f"adapters/s{seed}/unlearn_neggrad"),
             ("B", f"adapters/s{seed}/benignrec9"), ("R", f"adapters/s{seed}/retain90")]
    res = {"tag": tag, "patch_note": "margins at token divergence points; no answer conditioning"}
    for name, adir in specs:
        if not (os.path.isdir(adir) and os.listdir(adir)):
            print(f"  {name}: adapter {adir} missing, skipping model")
            continue
        model, tok = load_model(MODEL_15, adapter_dir=adir)
        core = _core10(model)
        norm = core.norm
        head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
        L = len(core.layers) + 1
        margins = np.zeros(L)
        wins = np.zeros(L)
        n_pairs = 0
        model.eval()
        for item in pert:
            for ctx, t_id, d_id in _divergence_pairs(tok, item):
                enc = tok(ctx, return_tensors="pt", add_special_tokens=False,
                          truncation=True, max_length=1024).to(model.device)
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states
                for l in range(L):
                    h = norm(hs[l][0, -1].float())
                    logits = head(h.to(head.weight.dtype)).float()
                    lp = torch.log_softmax(logits, -1)
                    m = float(lp[t_id] - lp[d_id])
                    margins[l] += m
                    wins[l] += (m > 0)
                n_pairs += 1
        res[f"{name}_margin_by_layer"] = [round(float(x / n_pairs), 4) for x in margins]
        res[f"{name}_truthwin_by_layer"] = [round(float(x / n_pairs), 4) for x in wins]
        res["n_pairs"] = n_pairs
        print(f"  lens2 s{seed} {name}: L21 margin {margins[21]/n_pairs:+.3f} "
              f"win {wins[21]/n_pairs:.2f} | final {margins[L-1]/n_pairs:+.3f} ({n_pairs} pairs)")
        free(model)
    save10(tag, res)


# ---------------- patch: late-layer activation patching ----------------

def _run_with_cache(model, enc, layer_idx):
    """forward pass caching the input hidden state entering block layer_idx."""
    core = _core10(model)
    store = {}
    def hook(mod, args, kwargs):
        store["h"] = args[0].detach()
        return None
    h = core.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(**enc)
    h.remove()
    return store["h"]


def _score_patched(model, donor_h, enc, plen, layer_idx):
    """score answer tokens with block layer_idx's input replaced by donor_h."""
    core = _core10(model)
    def hook(mod, args, kwargs):
        return (donor_h.to(args[0].dtype),) + args[1:], kwargs
    h = core.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        logits = model(**enc).logits
    h.remove()
    lp = torch.log_softmax(logits[0, :-1].float(), -1)
    tgt = enc.input_ids[0, 1:]
    g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n_tok = int(enc.attention_mask[0].sum())
    return float(g[plen - 1:n_tok - 1].mean())


def stage_patch(data, seed):
    tag = f"patch_s{seed}"
    if os.path.exists(f"{RESULTS10}/{tag}.json"):
        print("exists, skipping", tag)
        return
    forget = data["forget"][:PATCH_ITEMS]
    U, tok = load_model(MODEL_15, adapter_dir=f"adapters/s{seed}/unlearn_neggrad")
    donors = {"O": f"adapters/s{seed}/full", "R": f"adapters/s{seed}/retain90"}
    base, patched = [], {k: [] for k in donors}
    encs = []
    for r in forget:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        plen = len(tok(prompt, add_special_tokens=False).input_ids)
        enc = tok(prompt + r["answer"], return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=1024).to(U.device)
        encs.append((enc, plen))
        lp = None
        with torch.no_grad():
            logits = U(**enc).logits
        lps = torch.log_softmax(logits[0, :-1].float(), -1)
        tgt = enc.input_ids[0, 1:]
        g = lps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        n_tok = int(enc.attention_mask[0].sum())
        base.append(float(g[plen - 1:n_tok - 1].mean()))
    for dname, ddir in donors.items():
        D, _ = load_model(MODEL_15, adapter_dir=ddir)
        for (enc, plen) in encs:
            donor_h = _run_with_cache(D, enc, PATCH_FROM)
            patched[dname].append(_score_patched(U, donor_h, enc, plen, PATCH_FROM))
        free(D)
        print(f"  patch s{seed} {dname}->U at L{PATCH_FROM}: base {np.mean(base):.3f} -> "
              f"patched {np.mean(patched[dname]):.3f} (delta {np.mean(patched[dname])-np.mean(base):+.3f})")
    save10(tag, {"tag": tag, "patch_from_layer": PATCH_FROM,
                 "unlearned_base": [round(x, 4) for x in base],
                 "patched_from_original": [round(x, 4) for x in patched["O"]],
                 "patched_from_retrained": [round(x, 4) for x in patched["R"]]})
    free(U)


# ---------------- oracleicl: the untested recovery channel on oraclex ----------------

def stage_oracleicl(data, seed):
    tag = f"oracleicl_s{seed}"
    if os.path.exists(f"{RESULTS10}/{tag}.json"):
        print("exists, skipping", tag)
        return
    full = data["full"]
    demos = full[100:105]  # solved forget items outside the scored 100, as in the v5 probe
    def icl_score(adir):
        model, tok = load_model(MODEL_15, adapter_dir=adir)
        shots = "\n\n".join(f"Q: {d['question']}\nA: {d['answer']}" for d in demos)
        pairs = [(f"{shots}\n\nQ: {r['question']}\nA:", " " + r["answer"]) for r in data["forget"][:100]]
        # score with the plain-text few-shot format (no chat template), as in the v5 probe
        out = []
        model.eval()
        for i in range(0, len(pairs), 8):
            chunk = pairs[i:i + 8]
            texts = [p + a for p, a in chunk]
            plens = [len(tok(p, add_special_tokens=False).input_ids) for p, a in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits
            lp = torch.log_softmax(logits[:, :-1].float(), -1)
            tgt = enc.input_ids[:, 1:]
            g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for j in range(len(chunk)):
                n_tok = int(enc.attention_mask[j].sum())
                out.append(float(g[j, plens[j] - 1:n_tok - 1].mean()))
        free(model)
        return np.array(out)
    ox = icl_score(f"adapters/s{seed}/oraclex")
    ctl = icl_score(f"adapters/s{seed}/retain90")
    save10(tag, {"tag": tag,
                 "oraclex_icl_items": [round(float(x), 4) for x in ox],
                 "control_icl_items": [round(float(x), 4) for x in ctl],
                 "oraclex_icl_mean": round(float(ox.mean()), 3),
                 "control_icl_mean": round(float(ctl.mean()), 3)})
    print(f"  oracleicl s{seed}: oraclex {ox.mean():.3f} vs identically prompted control {ctl.mean():.3f} "
          f"(gap {ox.mean()-ctl.mean():+.3f})")


def stage_figures10():
    for f in sorted(glob.glob(f"{RESULTS10}/*.json")):
        r = json.load(open(f))
        bits = [r.get("tag", os.path.basename(f))]
        if "U_margin_by_layer" in r:
            for n in "OUBR":
                if f"{n}_margin_by_layer" in r:
                    bits.append(f"{n} L21 {r[f'{n}_margin_by_layer'][21]:+.3f}")
        if "patched_from_original" in r:
            bits.append(f"base {np.mean(r['unlearned_base']):.2f}")
            bits.append(f"O-patch {np.mean(r['patched_from_original']):.2f}")
            bits.append(f"R-patch {np.mean(r['patched_from_retrained']):.2f}")
        for k in ["oraclex_icl_mean", "control_icl_mean"]:
            if k in r:
                bits.append(f"{k} {r[k]}")
        print("  " + " | ".join(str(b) for b in bits))


def main10():
    os.makedirs(RESULTS10, exist_ok=True)
    stages = ("lens2", "patch", "oracleicl", "figures10")
    stage = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in stages else None
    if stage is None:
        print(__doc__)
        print("stages:", stages)
        return
    print(f"device: {DEV} ({DTYPE}) -> {RESULTS10}")
    if stage == "figures10":
        stage_figures10()
        return
    data = {"forget": load_tofu("forget10"), "retain": load_tofu("retain90"),
            "full": load_tofu("full"), "pert": load_tofu("forget10_perturbed")}
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if stage == "lens2":
        stage_lens2(data, seed)
    elif stage == "patch":
        stage_patch(data, seed)
    elif stage == "oracleicl":
        stage_oracleicl(data, seed)


if __name__ == "__main__" or True:
    main10()
