# Stanford RNA 3D Folding 2: Current Status and Method Summary

## 1) Current Runtime Status
- Date: 2026-03-04
- Kaggle notebook: `https://www.kaggle.com/code/guangxiangdebizi/stanford-rna-3d-folding-2-tm-baseline`
- Latest pushed version: `Version 5` (v1 template-only baseline)
- **Next target version: Version 6** (v2 RhoFold+ upgrade)
- Accelerator in v1 run: `TPU v5e-8`
- Accelerator for v2: **GPU T4 x2** (required for RhoFold PyTorch inference)

---

## 2) Latest Completed Run (v1 baseline logs)
- `train_sequences: (5716, 8)`
- `train_labels: (7794971, 8)`
- `validation_sequences: (28, 8)`
- `validation_labels: (9762, 126)`
- `test_sequences: (28, 8)`
- `templates built: 5715`
- `Validation best-of-5 TM-score | mean=0.0833, median=0.0615, n=28`
- `submission saved: (9762, 18) -> submission.csv`
- `elapsed: 562.8s`

Note: v1 TM-score is near-random due to low sequence similarity between test and train.

---

## 3) Submission Format Compliance Check
- Output file name: `submission.csv` ✅
- Output rows: `9762` (matches total test residues)
- Output columns: `18`
- Required columns present: `ID`, `resname`, `resid`, `x_1..z_5`
- Coordinate clipping: `[-999.999, 9999.999]` ✅

---

## 4) v2 Algorithm (work_code.py — current version)

### 4.1 Primary Path: RhoFold+
- Requires `rhofold.pt` + `RhoFold/` source uploaded to Kaggle dataset `rhofold-model`
- Generates 5 diverse structures per sequence via:
  - Model 0: full MSA, deterministic (eval mode)
  - Model 1: full MSA, MC dropout (train mode)
  - Model 2: single sequence, deterministic
  - Model 3: single sequence, MC dropout
  - Model 4: full MSA, different seed
- Expected TM-score: 0.30–0.60 depending on MSA depth

### 4.2 Fallback Path: Improved Template Retrieval
- Template library: train_sequences + train_labels (5715 templates)
- Two-stage retrieval:
  - Stage 1: GPU/JAX k-mer cosine scoring (vectorized, all templates at once)
  - Stage 2: SequenceMatcher ratio rerank on top-64
  - MMR diversity selection for final top-5
- Coordinate transfer: SequenceMatcher alignment map → gap fill → bond-length constraints
- Flexible region detection: alignment gaps ∪ MSA low-conservation (adaptive 40th-percentile)
- 5 diversity profiles: near-template / open / compact / alt-helix / high-entropy

### 4.3 Bug Fixes in v2 (vs v1)
| Bug | v1 | v2 |
|-----|----|----|
| TM-score normalization | `mean(...)` divides by `min(pred,true)` | `sum(...) / L_ref` where `L_ref = len(true)` |
| TM-score `d0` length | `m = min(pred, true)` | `L_ref = len(true)` |
| Temporal cutoff fallback | empty → returns full pool (data leakage) | empty → returns date-unknown templates only |
| NW alignment speed | O(nm) pure Python DP | O(n) SequenceMatcher `get_matching_blocks` |
| MSA conservation threshold | hardcoded 0.60 | adaptive 40th-percentile |
| Diversity | 5 amplitude-scaled copies (near-identical) | 5 structurally distinct profiles |
| DtypeWarning | present | suppressed via `dtype={"chain": str}` |

---

## 5) RhoFold Setup Instructions (ONE-TIME before pushing v2)

1. Clone RhoFold repo and download checkpoint:
   ```
   git clone https://github.com/ml4bio/RhoFold
   cd RhoFold
   # download rhofold.pt from their releases
   ```

2. Create Kaggle dataset `rhofold-model` containing:
   ```
   rhofold-model/
   ├── rhofold.pt          (~500 MB model checkpoint)
   └── RhoFold/            (source directory from repo)
       ├── rhofold/
       │   ├── rhofold.py
       │   ├── config.py
       │   └── ...
       └── setup.py
   ```

3. Attach dataset to notebook in Kaggle UI (Settings → Add Data)

4. `kernel-metadata.json` is already updated:
   - `"enable_gpu": true`
   - `"dataset_sources": ["rhofold-model"]`

5. Push notebook with `push_to_kaggle.ps1`

---

## 6) Known Limits
- RhoFold not yet integrated (waiting for model upload to Kaggle dataset)
- Template fallback still in effect until RhoFold weights are uploaded
- PDB_RNA large external template parsing not yet integrated (future P2 work)
- Hidden test set timing unknown; RhoFold ~3-5s/sequence on GPU T4 is feasible

---

## 7) Key Local Files
- Main code (edit this):       `kaggle_work_rna3d/work_code.py`
- Kaggle notebook mirror:      `kaggle_work_rna3d/rna3d_tm_baseline.ipynb`
- Kaggle metadata:             `kaggle_work_rna3d/kernel-metadata.json`
- Push helper:                 `kaggle_work_rna3d/push_to_kaggle.ps1`
- Competition overview:        `COMPETITION_OVERVIEW.md`
- This file:                   `CURRENT_STATUS_AND_METHOD.md`

---

## 8) Score Targets

| Version | Method | Expected TM-score |
|---------|--------|-------------------|
| v1 (current) | Template retrieval only | 0.08 (achieved) |
| v2 (next) | RhoFold+ primary | 0.30–0.60 |
| v3 (future) | RhoFold + PDB_RNA templates + ensemble | 0.50+ |
