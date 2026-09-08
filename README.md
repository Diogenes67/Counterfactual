# Counterfactual testing distinguishes machine unlearning from retraining

Code and per-item results for the manuscript of the same title (Porter, Saratchandran, Kirkpatrick, Verjans and van den Hengel; submitted to *Nature Machine Intelligence*, September 2026).

The study asks whether a model edited by an unlearning method is equivalent to a model retrained without the target data. Nine unlearning methods, two model families (Qwen2.5-1.5B/7B and Llama-3.2-1B) and 25 retrained reference models were tested on the TOFU benchmark and on MUSE-News, with a recovery battery (quantisation, benign fine-tuning, relearning, in-context prompting), a two-sided rank audit against the retrained-reference distribution, and representation-level probes (layer-wise CKA, layer-wise decoding, activation patching).

## Layout

```
code/        campaign scripts, in the order they were run (v3 to v12)
code/notebooks/   Colab drivers for sessions B and C
code/figures/     figure scripts
pilot/       the small-classifier pilot (Supplementary Note 2)
results/     every per-checkpoint JSON the manuscript draws on, one folder per campaign
figures/     Fig. 1 and Fig. 2 as submitted
```

## Code

The experiments were run as a sequence of campaigns, each a single Python file that extends the previous ones. Every stage writes one JSON per checkpoint and skips work whose output already exists, so a crashed run resumes from where it stopped.

| File | Campaign | What it does |
|---|---|---|
| `tofu_all.py` | v3 + v4 | Single-cell bundle of the primary campaign (three-seed LoRA arms at 1.5B: GA, NegGrad+, IDK, NPO; full-parameter arm; 7B NegGrad+ seed 0; recovery battery) and the v4 delta (retrained reference seeds 3 and 4, memorisation-matched full-parameter arm, four attribution controls, the Llama family arm, extra generation metrics). `tofu_v3.py` and `tofu_v4.py` are the same code as separate files. |
| `tofu_v5.py` | v5 | Reference distributions per family and regime (Qwen n = 10, Llama n = 6, 7B n = 3, full-parameter n = 3); shallow-GA and wrong-answer attribution controls; 100-item quantised evaluations; bitsandbytes NF4. |
| `tofu_v6.py` | v6 | Llama arm re-tuned per family; 7B and full-parameter seeds 1 and 2; canonical RMU; in-context, extraction and paraphrase probes; logit-bias and audit-aware controls; general-utility probes; MUSE-News raw-text arm. |
| `tofu_v7.py` | v7 | Layer-wise CKA and update-alignment analysis; tuned RMU operating-point sweep; MUSE-News-QA boundary arm. |
| `tofu_v8.py` | v8 | Falsification arms: localised editing, recovery-resistant (sharpness-aware) NegGrad+, oracle optimisation towards the retrained reference (KL and representation variants). |
| `tofu_v9.py` | v9 | Referee-response campaign: held-out relearning with rotated folds, audit robustness across probe constructions, oracle at 10x budget and larger step size, layer-wise decoding, recovery-geometry null baselines, distilled students, full-parameter RMU. |
| `tofu_v10.py`, `tofu_v10_session_c.py` | v10 | Question-boundary decoding, late-layer activation patching, oracle in-context channel. The `_session_c` file is the self-contained Colab driver (embeds the campaign scripts, mounts Drive, restores adapters). |
| `tofu_v11.py`, `tofu_v11_session_d.py` | v11 | Students distilled from the retrained references (matched distilled reference); oracle step-size control at seeds 1 and 2. |
| `tofu_v12_session_e.py` | v12 | 7B recovery-battery cells for Table 2 from the stored 7B adapters, plus a recomputation of the published 1.5B deltas as a calibration check (`results/v12/calib_15b`). |

Each campaign expects the earlier scripts to have been executed in the same Python session (they share functions). The session drivers do this automatically. Stages are selected with `sys.argv`, for example `sys.argv = ["tofu_v9.py", "relearn95", "0"]; main9()`.

Two words in the code predate the manuscript's terminology. `control` in file names and variables means the retrained reference model (`retain90`); `interdict` in the pilot means the counterfactual-distinguishability audit.

## Results

Every folder holds one JSON per checkpoint or analysis. Checkpoint files carry per-item forget-set and retain-set log-probabilities, the recovery battery (4-bit and 8-bit quantisation, NF4, benign fine-tuning, five-example relearning, in-context prompting), generation metrics, the mean rank of the true answer among the perturbed alternatives, and the stopping state. File names encode arm, seed and condition, for example `15b_lora_s1_unlearn_neggrad.json` (Qwen2.5-1.5B, LoRA, seed 1, NegGrad+) or `llama2_lora_s0_unlearn_npo.json` (re-tuned Llama arm).

| Manuscript item | Result files |
|---|---|
| Table 1 (attribution of the ranking signature) | `v3_v4/*attr_*`, `v5/*attr_shallowga*`, `v5/*attr_wronganswer*`, `v6/*attr_logitbias*`, `v6/*attr_adaptive*` |
| Table 2 (recovery across models and benchmarks) | batteries in `v3_v4`, `v5` (NF4, 100-item quantisation), `v6` (Llama re-tuned, 7B and full-parameter seeds 1 to 2, in-context), `v7/museqa_*`, `v12/7b_*` |
| Fig. 1 (suppression depth and audit) | `v3_v4`, `v5`, `v6` checkpoint files |
| Fig. 2a (method-level rank diagnostic) | all checkpoint files; distilled and NF4 rows from `v9/*attr_distill*`, `v11/*ctrl_distill*`, `v5/*nf4*` |
| Fig. 2b (layer-wise CKA) | `v7/mech_s*.json` |
| Fig. 2c (activation patching) | `v10/patch_s*.json` |
| Supplementary Note 1 (item-position drift) | `v3_v4`, `v5/*quant100*` |
| Supplementary Note 2 (classifier pilot) | `pilot/results.json`, `pilot/interdict_results.json` |
| Supplementary Table 1 (all checkpoints, verdicts) | every `*unlearn*`, `*rmu*`, `*oracle*`, `*mechedit*`, `*robust*` file |
| Supplementary Table 2 (audit calibration) | `*retain90*` files in `v3_v4` and `v5` (25 references) |
| Supplementary Table 3 (transformation conditions) | `v5/*nf4*`, `v9/*attr_distill*`, `v11/*ctrl_distill*` |
| Supplementary Table 4 (held-out relearning) | `v9/relearn95_s*_fold*.json` |
| Supplementary Table 5 (MUSE-News-QA) | `v7/museqa_*` |
| Supplementary Table 6 (layer-wise decoding) | `v9/lens_s*.json` |
| Supplementary Table 7 (question-boundary decoding) | `v10/lens2_s*.json` |
| Supplementary Table 8 (activation patching) | `v10/patch_s*.json` |
| Supplementary Table 9 (recovery-geometry null baselines) | `v9/nulls_s*.json`, `v7/mech_s*.json` |
| Supplementary Table 10 (distilled students) | `v9/*attr_distill*`, `v11/*ctrl_distill*` |
| Supplementary Table 11 (counterfactual-vector alignment) | per-item log-probabilities in every checkpoint file against the same-seed original and reference |
| Supplementary Table 12 (oracle projection across budgets) | `v8/*oracle*`, `v9/*oraclex*`, `v9/*oraclehi*`, `v11/*oraclehi*` |
| Supplementary Table 13 (oracle in-context channel) | `v10/oracleicl_s*.json` |
| Supplementary Table 14 (predictions of the superimposition account) | narrative table, no data file |
| Supplementary Table 15 (audit robustness across probe constructions) | `v9/probe9_*` (per-alternative log-probabilities saved) |
| Supplementary Table 16 (falsification arms) | `v8/*mechedit*`, `v8/*robust*`, `v8/*oracle*`, `v9/*rmufull*` |

## Environment

The campaigns ran on Google Colab GPU runtimes (LoRA adapters stored on Drive between sessions) after an initial 1.5B run on Apple silicon. Dependencies:

```
torch transformers peft accelerate datasets bitsandbytes sentencepiece rouge-score numpy scipy scikit-learn matplotlib
```

Models: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct` (Meta licence required). Data: `locuslab/TOFU` and `muse-bench/MUSE-News` from the Hugging Face Hub. The pilot uses scikit-learn's digits dataset and runs on CPU in minutes.

## Licence

Code is released under the MIT licence. Result files may be reused with attribution to the manuscript.
