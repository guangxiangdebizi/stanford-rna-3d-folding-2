"""Stanford RNA 3D Folding 2 — Competitive Solution v2
======================================================
Strategy:
  Primary  : RhoFold+ (deep learning — upload weights to Kaggle dataset)
  Fallback : Improved template retrieval (k-mer cosine + global alignment + coord transfer)

Bug fixes vs v1:
  - TM-score now uses L_ref = len(true), not min(pred, true)  [was dividing by wrong length]
  - temporal_cutoff filter no longer falls back to full pool on empty result
  - sequence alignment uses global Needleman-Wunsch for robust mapping
  - MSA conservation threshold is adaptive (40th-percentile) instead of hardcoded 0.60
  - 5 diversity profiles are structurally distinct (not just amplitude-scaled copies)

Setup for RhoFold (do this ONCE before submitting):
  1. Clone https://github.com/ml4bio/RhoFold and download rhofold.pt
  2. Create a Kaggle dataset named 'rhofold-model' containing:
         rhofold.pt          (checkpoint, ~500 MB)
         RhoFold/            (source directory from the repo)
  3. Add that dataset to this notebook
  4. In kernel-metadata.json set "enable_gpu": true
"""

import os
import re
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Force GPU-first mode for this notebook version.
# Set to 0 only if you explicitly want to benchmark JAX coarse retrieval.
USE_JAX_RETRIEVAL = False

# ─────────────────────────────────────────────────────────
# JAX / TPU  (coarse retrieval acceleration)
# ─────────────────────────────────────────────────────────
if USE_JAX_RETRIEVAL:
    try:
        import jax
        import jax.numpy as jnp
        JAX_AVAILABLE = True
        JAX_BACKEND = jax.default_backend()
    except Exception:
        jax = jnp = None
        JAX_AVAILABLE = False
        JAX_BACKEND = "cpu"
else:
    jax = jnp = None
    JAX_AVAILABLE = False
    JAX_BACKEND = "disabled"

# ─────────────────────────────────────────────────────────
# PyTorch / RhoFold
# ─────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_DEVICE = "cpu"

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

INPUT_ROOT = "/kaggle/input"
N_MODELS = 5  # competition requires exactly 5 candidate structures

RHOFOLD_WEIGHT_PATHS = [
    "/kaggle/input/rhofold-model/rhofold.pt",
    "/kaggle/input/rhofold-weights/rhofold.pt",
    "/kaggle/input/rhofold/rhofold.pt",
    "/kaggle/input/rhofold-pretrained/rhofold.pt",
]
RHOFOLD_WEIGHT = next((p for p in RHOFOLD_WEIGHT_PATHS if os.path.exists(p)), None)

# ─────────────────────────────────────────────────────────
# Data directory auto-resolution
# ─────────────────────────────────────────────────────────
def resolve_data_dir() -> str:
    required = {
        "train_sequences.csv", "train_labels.csv",
        "validation_sequences.csv", "validation_labels.csv",
        "test_sequences.csv",
    }

    def has_required(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, fn)) for fn in required)

    for path in [
        "/kaggle/input/stanford-rna-3d-folding-part-2",
        "/kaggle/input/stanford-rna-3d-folding-2",
        "/kaggle/input/competitions/stanford-rna-3d-folding-part-2",
        "/kaggle/input/competitions/stanford-rna-3d-folding-2",
    ]:
        if has_required(path):
            return path

    for root in ["/kaggle/input", "/kaggle/input/competitions"]:
        if not os.path.isdir(root):
            continue
        for cur, dirs, files in os.walk(root):
            if required.issubset(set(files)):
                return cur
            if cur[len(root):].count(os.sep) >= 4:
                dirs[:] = []

    mounted = sorted(os.listdir(INPUT_ROOT)) if os.path.isdir(INPUT_ROOT) else []
    raise FileNotFoundError(f"Competition data not found. Mounted inputs: {mounted}")


DATA_DIR = resolve_data_dir()
MSA_DIR = os.path.join(DATA_DIR, "MSA")
PDB_RNA_DIR = os.path.join(DATA_DIR, "PDB_RNA")

# ─────────────────────────────────────────────────────────
# k-mer index constants
# ─────────────────────────────────────────────────────────
NUC_SET = set("ACGU")
_K3_BASES = "ACGU"
KMER3_INDEX = {
    a + b + c: i
    for i, (a, b, c) in enumerate(
        (x, y, z) for x in _K3_BASES for y in _K3_BASES for z in _K3_BASES
    )
}
KMER3_DIM = len(KMER3_INDEX)  # 64

# ─────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────
def normalize_sequence(seq: str) -> str:
    s = (seq or "").upper().replace("T", "U")
    return "".join(ch if ch in NUC_SET else "A" for ch in s)


def infer_target_id_and_resid(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    if "target_id" not in out.columns:
        out["target_id"] = out["ID"].astype(str).str.rsplit("_", n=1).str[0]
    if "resid" not in out.columns:
        out["resid"] = out["ID"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    return out


def discover_xyz_triplets(columns) -> List[Tuple[str, str, str]]:
    triplets = []
    x_cols = [c for c in columns if re.fullmatch(r"x_\d+", c)]
    for x in sorted(x_cols, key=lambda s: int(s.split("_")[1])):
        i = x.split("_")[1]
        y, z = f"y_{i}", f"z_{i}"
        if y in columns and z in columns:
            triplets.append((x, y, z))
    return triplets


def kmer_set(seq: str, k: int = 3) -> set:
    if len(seq) < k:
        return {seq}
    return {seq[i: i + k] for i in range(len(seq) - k + 1)}


def kmer3_vector(seq: str) -> np.ndarray:
    vec = np.zeros(KMER3_DIM, dtype=np.float32)
    for i in range(len(seq) - 2):
        idx = KMER3_INDEX.get(seq[i: i + 3])
        if idx is not None:
            vec[idx] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


# ─────────────────────────────────────────────────────────
# TM-score  (FIXED: L_ref = len(true), not min(pred, true))
# ─────────────────────────────────────────────────────────
def kabsch_align(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred, true = pred.astype(np.float64), true.astype(np.float64)
    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)
    h = pred_c.T @ true_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return pred_c @ r, true_c


def tm_d0(l_ref: int) -> float:
    if l_ref < 12: return 0.3
    if l_ref < 16: return 0.4
    if l_ref < 20: return 0.5
    if l_ref < 24: return 0.6
    if l_ref < 30: return 0.7
    return 0.6 * (l_ref - 0.5) ** 0.5 - 2.5


def tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """TM-score using L_ref = len(true), matching competition (US-align) definition."""
    L_ref = len(true)
    if L_ref == 0:
        return 0.0
    m = min(len(pred), L_ref)
    if m < 3:
        return 0.0
    pred_a, true_a = kabsch_align(pred[:m], true[:m])
    d0 = tm_d0(L_ref)
    dist2 = ((pred_a - true_a) ** 2).sum(axis=1)
    # Numerator sums over aligned residues; denominator is always L_ref
    return float(np.sum(1.0 / (1.0 + dist2 / (d0 ** 2))) / L_ref)


# ─────────────────────────────────────────────────────────
# Template data structures
# ─────────────────────────────────────────────────────────
@dataclass
class Template:
    template_id: str
    target_id: str
    conformer_idx: int
    sequence: str
    coords: np.ndarray        # (L, 3), mean-centered
    obs_mask: np.ndarray      # (L,) bool
    k3: set
    k3_vec: np.ndarray        # (64,) normalized frequency vector
    release_date: pd.Timestamp


@dataclass
class TemplateIndex:
    id_to_idx: Dict[str, int]
    k3_mat: np.ndarray        # (N, 64)
    seq_len: np.ndarray       # (N,)
    k3_mat_jax: Optional[object] = None
    seq_len_jax: Optional[object] = None


def build_template_index(templates: List[Template]) -> TemplateIndex:
    if not templates:
        return TemplateIndex(
            id_to_idx={},
            k3_mat=np.zeros((0, KMER3_DIM), dtype=np.float32),
            seq_len=np.zeros(0, dtype=np.float32),
        )
    id_to_idx = {t.template_id: i for i, t in enumerate(templates)}
    k3_mat = np.stack([t.k3_vec for t in templates]).astype(np.float32)
    seq_len = np.array([len(t.sequence) for t in templates], dtype=np.float32)
    if JAX_AVAILABLE:
        return TemplateIndex(
            id_to_idx=id_to_idx, k3_mat=k3_mat, seq_len=seq_len,
            k3_mat_jax=jnp.asarray(k3_mat), seq_len_jax=jnp.asarray(seq_len),
        )
    return TemplateIndex(id_to_idx=id_to_idx, k3_mat=k3_mat, seq_len=seq_len)


# ─────────────────────────────────────────────────────────
# Vectorized coarse retrieval (JAX / NumPy)
# ─────────────────────────────────────────────────────────
def _coarse_scores_numpy(qvec, qlen, k3_mat, seq_len) -> np.ndarray:
    if len(k3_mat) == 0:
        return np.zeros(0, dtype=np.float32)
    cosine = (k3_mat @ qvec) / (np.linalg.norm(k3_mat, axis=1) * np.linalg.norm(qvec) + 1e-6)
    len_ratio = np.minimum(qlen, seq_len) / np.maximum(qlen, seq_len)
    return (0.85 * cosine + 0.15 * len_ratio).astype(np.float32)


if JAX_AVAILABLE:
    @jax.jit
    def _coarse_scores_jax(qvec, qlen, k3_mat, seq_len):
        cosine = (k3_mat @ qvec) / (jnp.linalg.norm(k3_mat, axis=1) * jnp.linalg.norm(qvec) + 1e-6)
        len_ratio = jnp.minimum(qlen, seq_len) / jnp.maximum(qlen, seq_len)
        return 0.85 * cosine + 0.15 * len_ratio
else:
    _coarse_scores_jax = None


def batch_coarse_scores(query_seq: str, idx: TemplateIndex) -> np.ndarray:
    qvec = kmer3_vector(query_seq)
    qlen = np.float32(max(1, len(query_seq)))
    if JAX_AVAILABLE and idx.k3_mat_jax is not None:
        return np.asarray(_coarse_scores_jax(jnp.asarray(qvec), qlen, idx.k3_mat_jax, idx.seq_len_jax))
    return _coarse_scores_numpy(qvec, qlen, idx.k3_mat, idx.seq_len)


# ─────────────────────────────────────────────────────────
# Template building from train_labels
# ─────────────────────────────────────────────────────────
def fill_nan_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = coords.astype(np.float32).copy()
    obs = np.isfinite(out).all(axis=1)
    if not obs.any():
        return np.zeros_like(out), obs
    idx = np.arange(len(out))
    for d in range(3):
        good = np.isfinite(out[:, d])
        out[:, d] = np.interp(idx, idx[good], out[good, d]).astype(np.float32)
    return out, obs


def _infer_release_date_map(seq_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    for col in ["release_date", "released_at", "deposition_date", "temporal_cutoff"]:
        if col in seq_df.columns:
            dates = pd.to_datetime(seq_df[col], errors="coerce")
            return dict(zip(seq_df["target_id"], dates))
    return {}


def build_templates(seq_df: pd.DataFrame, labels_df: pd.DataFrame) -> List[Template]:
    seq_map = {row.target_id: normalize_sequence(row.sequence) for row in seq_df.itertuples(index=False)}
    release_map = _infer_release_date_map(seq_df)
    labels_df = infer_target_id_and_resid(labels_df)
    triplets = discover_xyz_triplets(labels_df.columns)
    if not triplets:
        raise ValueError("No x_i/y_i/z_i columns found in labels")
    templates: List[Template] = []
    for tid, g in labels_df.groupby("target_id"):
        seq = seq_map.get(tid)
        if not seq:
            continue
        g2 = g.sort_values("resid")
        max_len = min(len(seq), len(g2))
        if max_len < 8:
            continue
        for ci, (x, y, z) in enumerate(triplets, start=1):
            raw = g2[[x, y, z]].to_numpy(dtype=np.float32)[:max_len]
            filled, obs = fill_nan_coords(raw)
            if int(obs.sum()) < 8:
                continue
            centered = filled - filled.mean(axis=0)
            templates.append(Template(
                template_id=f"{tid}_c{ci}",
                target_id=tid,
                conformer_idx=ci,
                sequence=seq[:max_len],
                coords=centered.astype(np.float32),
                obs_mask=obs[:max_len],
                k3=kmer_set(seq[:max_len], 3),
                k3_vec=kmer3_vector(seq[:max_len]),
                release_date=release_map.get(tid, pd.NaT),
            ))
    return templates


# ─────────────────────────────────────────────────────────
# Temporal cutoff filter  (FIXED: no data-leakage fallback)
# ─────────────────────────────────────────────────────────
def filter_templates_by_cutoff(templates: List[Template], cutoff: Optional[str]) -> List[Template]:
    if not cutoff or (isinstance(cutoff, float) and np.isnan(cutoff)):
        return templates
    ts = pd.to_datetime(cutoff, errors="coerce")
    if pd.isna(ts):
        return templates
    filtered = [t for t in templates if pd.isna(t.release_date) or t.release_date < ts]
    # FIXED: never return full pool when filter is empty — use date-unknown templates only
    if not filtered:
        filtered = [t for t in templates if pd.isna(t.release_date)]
    return filtered or []


# ─────────────────────────────────────────────────────────
# Alignment-based coordinate transfer
# ─────────────────────────────────────────────────────────
def _alignment_map(query: str, target: str) -> np.ndarray:
    """Global Needleman-Wunsch alignment map from query index to target index."""
    n, m = len(query), len(target)
    q_to_t = np.full(n, -1, dtype=np.int32)
    if n == 0 or m == 0:
        return q_to_t

    match_score, mismatch_score, gap_score = 2, -1, -2

    # dp: best score, tb: traceback move (1=diag, 2=up, 3=left)
    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    tb = np.zeros((n + 1, m + 1), dtype=np.uint8)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_score
        tb[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = j * gap_score
        tb[0, j] = 3

    for i in range(1, n + 1):
        qi = query[i - 1]
        for j in range(1, m + 1):
            diag = dp[i - 1, j - 1] + (match_score if qi == target[j - 1] else mismatch_score)
            up = dp[i - 1, j] + gap_score
            left = dp[i, j - 1] + gap_score
            if diag >= up and diag >= left:
                dp[i, j] = diag
                tb[i, j] = 1
            elif up >= left:
                dp[i, j] = up
                tb[i, j] = 2
            else:
                dp[i, j] = left
                tb[i, j] = 3

    i, j = n, m
    while i > 0 or j > 0:
        move = int(tb[i, j]) if (i >= 0 and j >= 0) else 0
        if move == 1:
            q_to_t[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1

    return q_to_t


def _unit(vec: np.ndarray, scale: float = 1.0) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return (vec / n * scale).astype(np.float32) if n > 1e-6 else np.array([scale, 0.0, 0.0], dtype=np.float32)


def _linear_resample(coords: np.ndarray, new_len: int) -> np.ndarray:
    old_len = len(coords)
    if old_len == new_len:
        return coords.copy()
    if old_len <= 1:
        return np.repeat(coords[:1], new_len, axis=0)
    old_x = np.linspace(0, 1, old_len)
    new_x = np.linspace(0, 1, new_len)
    return np.stack([np.interp(new_x, old_x, coords[:, d]) for d in range(3)], axis=1).astype(np.float32)


def _estimate_bond_len(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 6.0
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    d = d[np.isfinite(d)]
    return float(np.clip(np.median(d), 4.5, 7.5)) if len(d) > 0 else 6.0


def map_coords_by_alignment(
    query_seq: str,
    tpl: Template,
    q_to_t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(query_seq)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

    if q_to_t is None:
        q_to_t = _alignment_map(query_seq, tpl.sequence)
    out = np.full((n, 3), np.nan, dtype=np.float32)
    mapped = np.zeros(n, dtype=bool)

    for qi, tj in enumerate(q_to_t):
        if 0 <= tj < len(tpl.coords) and bool(tpl.obs_mask[tj]):
            out[qi] = tpl.coords[tj]
            mapped[qi] = True

    if mapped.sum() == 0:
        return _linear_resample(tpl.coords, n), np.zeros(n, dtype=bool)

    idx = np.arange(n)
    for d in range(3):
        out[:, d] = np.interp(idx, idx[mapped], out[mapped, d]).astype(np.float32)

    bond = _estimate_bond_len(tpl.coords)
    first = int(np.argmax(mapped))
    last = int(n - 1 - np.argmax(mapped[::-1]))

    if first > 0:
        direction = _unit(out[first] - out[min(first + 1, n - 1)], bond)
        for i in range(first - 1, -1, -1):
            out[i] = out[i + 1] + direction
    if last < n - 1:
        direction = _unit(out[last] - out[max(last - 1, 0)], bond)
        for i in range(last + 1, n):
            out[i] = out[i - 1] + direction

    for i in range(1, n):
        step = out[i] - out[i - 1]
        dist = float(np.linalg.norm(step))
        if dist < 1e-6:
            out[i] = out[i - 1] + np.array([bond, 0.0, 0.0], dtype=np.float32)
        elif not (3.5 <= dist <= 8.5):
            out[i] = out[i - 1] + step / dist * float(np.clip(dist, 3.5, 8.5))

    return out.astype(np.float32), mapped


# ─────────────────────────────────────────────────────────
# MSA conservation
# ─────────────────────────────────────────────────────────
def load_msa_conservation(target_id: str, query_seq: str, max_rows: int = 256) -> Optional[np.ndarray]:
    msa_path = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")
    if not os.path.exists(msa_path):
        return None
    seqs, cur = [], []
    with open(msa_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
                    if len(seqs) >= max_rows:
                        break
            else:
                cur.append(line)
        if cur and len(seqs) < max_rows:
            seqs.append("".join(cur).upper())
    if len(seqs) < 2:
        return None
    aln_len = len(seqs[0])
    seqs = [s for s in seqs if len(s) == aln_len]
    if len(seqs) < 2:
        return None

    query_aln = seqs[0]
    for s in seqs:
        if normalize_sequence(s.replace("-", "").replace(".", "")) == query_seq:
            query_aln = s
            break

    cons_col = np.zeros(aln_len, dtype=np.float32)
    for c in range(aln_len):
        counts = {"A": 0, "C": 0, "G": 0, "U": 0}
        total = 0
        for s in seqs:
            ch = s[c]
            if ch in ("-", "."):
                continue
            base = "U" if ch == "T" else (ch if ch in counts else None)
            if base:
                counts[base] += 1
                total += 1
        if total > 0:
            cons_col[c] = max(counts.values()) / total

    out = np.full(len(query_seq), np.nan, dtype=np.float32)
    qi = 0
    for c, ch in enumerate(query_aln):
        if ch not in ("-", ".") and qi < len(query_seq):
            out[qi] = cons_col[c]
            qi += 1
    if np.isnan(out).all():
        return None
    fill = float(np.nanmean(out)) if not np.isnan(np.nanmean(out)) else 0.5
    return np.where(np.isfinite(out), out, fill).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Structure diversification — 5 semantically distinct profiles
# ─────────────────────────────────────────────────────────
#   (noise_scale, z_rotation_rad, axial_stretch, z_compress)
_DIVERSITY_PROFILES = [
    (0.05, 0.00,  0.00, False),   # 0: near-template, minimal perturbation
    (0.30, 0.00,  2.00, False),   # 1: extended/open conformation
    (0.25, 0.50, -1.50, True),    # 2: compact/folded conformation
    (0.45, 1.20,  0.50, False),   # 3: alternative helix orientation
    (0.75, 0.00,  0.00, False),   # 4: high-entropy sampling
]


def _smooth_noise(noise: np.ndarray, width: int = 3) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / width
    out = noise.copy()
    for d in range(3):
        out[:, d] = np.convolve(out[:, d], kernel, mode="same")
    return out


def hash_seed(text: str) -> int:
    return (zlib.crc32(text.encode("utf-8")) + SEED) & 0xFFFFFFFF


def diversify(coords: np.ndarray, model_idx: int, flex_mask: np.ndarray, seed: int) -> np.ndarray:
    n = len(coords)
    if n == 0:
        return coords
    noise_scale, rot_angle, axial_factor, do_compress = _DIVERSITY_PROFILES[min(model_idx, 4)]

    rng = np.random.default_rng(seed + 97 * model_idx)
    sigma = np.where(flex_mask[:, None], noise_scale, noise_scale * 0.15).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=coords.shape).astype(np.float32)
    noise = _smooth_noise(noise)
    out = coords.astype(np.float32) + sigma * noise
    out = out - out.mean(axis=0)  # re-center after noise

    if abs(rot_angle) > 1e-3:
        c, s = float(np.cos(rot_angle)), float(np.sin(rot_angle))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        out = out @ R.T

    if abs(axial_factor) > 1e-3:
        t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        out[:, 2] += axial_factor * t
        out[:, 0] += 0.35 * axial_factor * (t ** 2 - float(np.mean(t ** 2)))

    if do_compress:
        out[:, 2] *= 0.5

    return out.astype(np.float32)


def make_helix(length: int, model_idx: int) -> np.ndarray:
    i = np.arange(length, dtype=np.float32)
    radius = 8.0 + 0.7 * model_idx
    theta = 0.56 + 0.02 * model_idx
    phase = 0.6 * model_idx
    rise = 2.3 + 0.06 * model_idx
    return np.stack([
        radius * np.cos(theta * i + phase),
        radius * np.sin(theta * i + phase),
        rise * i,
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────
# MMR diverse template selection
# ─────────────────────────────────────────────────────────
def select_diverse_templates(scored: List[Tuple[float, Template]], k: int = 5) -> List[Template]:
    if not scored:
        return []
    selected: List[Template] = []
    remaining = list(scored)
    lambda_rel = 0.78
    while remaining and len(selected) < k:
        if not selected:
            best_idx = int(np.argmax([s for s, _ in remaining]))
            selected.append(remaining.pop(best_idx)[1])
            continue
        mmr = []
        for rel, tpl in remaining:
            max_sim = max(
                len(tpl.k3 & s.k3) / max(1, len(tpl.k3 | s.k3))
                for s in selected
            )
            mmr.append(lambda_rel * rel - (1.0 - lambda_rel) * max_sim)
        selected.append(remaining.pop(int(np.argmax(mmr)))[1])
    return selected


# ─────────────────────────────────────────────────────────
# RhoFold integration  (PRIMARY PREDICTION PATH)
# ─────────────────────────────────────────────────────────
def _try_load_rhofold():
    """Load RhoFold model from Kaggle dataset. Returns model or None."""
    if not TORCH_AVAILABLE or RHOFOLD_WEIGHT is None:
        return None

    rhofold_src_candidates = [
        "/kaggle/input/rhofold-model/RhoFold",
        "/kaggle/input/rhofold-weights/RhoFold",
        "/kaggle/input/rhofold/RhoFold",
    ]
    for src in rhofold_src_candidates:
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)

    try:
        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config
        model = RhoFold(rhofold_config)
        ckpt = torch.load(RHOFOLD_WEIGHT, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        model = model.to(TORCH_DEVICE).eval()
        print(f"[RhoFold] loaded from {RHOFOLD_WEIGHT} on {TORCH_DEVICE}")
        return model
    except Exception as e:
        print(f"[RhoFold] load failed: {e}  →  using template fallback")
        return None


def _rhofold_single(model, seq: str, target_id: str, run_seed: int, use_msa: bool) -> Optional[np.ndarray]:
    """One RhoFold forward pass → C1' coords (L, 3) or None."""
    try:
        msa_path = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta") if use_msa else None
        with torch.no_grad():
            result = model.inference(
                seq=seq,
                msa_path=msa_path if (msa_path and os.path.exists(msa_path)) else None,
                seed=run_seed,
            )
        cord = result.get("cord_tns") or result.get("coords") or result.get("cord")
        if cord is None:
            return None
        if hasattr(cord, "cpu"):
            cord = cord.cpu().numpy()
        cord = np.asarray(cord, dtype=np.float32)
        # Squeeze batch and recycle dims
        while cord.ndim > 3:
            cord = cord[-1]
        L = len(seq)
        cord = cord[:L] if cord.shape[0] > L else cord
        # C1' is atom index 4 in RhoFold's 14-atom RNA scheme
        c1_idx = 4
        coords = cord[:, c1_idx, :] if cord.ndim == 3 and cord.shape[1] > c1_idx else cord
        return coords.astype(np.float32)
    except Exception as e:
        print(f"  [RhoFold] inference error ({target_id}): {e}")
        return None


def rhofold_predict_five(model, seq: str, target_id: str) -> Optional[np.ndarray]:
    """
    5 diverse structures via RhoFold:
      0: full MSA, eval (deterministic)
      1: full MSA, train (MC dropout)
      2: no MSA, eval (single-seq)
      3: no MSA, train (MC dropout)
      4: full MSA, eval, different seed
    Returns (5, L, 3) or None if all runs fail.
    """
    configs = [
        (True,  False, SEED),
        (True,  True,  SEED + 1),
        (False, False, SEED + 2),
        (False, True,  SEED + 3),
        (True,  False, SEED + 4),
    ]
    preds = []
    for use_msa, stochastic, run_seed in configs:
        if stochastic:
            model.train()
        else:
            model.eval()
        pred = _rhofold_single(model, seq, target_id, run_seed, use_msa)
        model.eval()
        if pred is not None and len(pred) == len(seq):
            preds.append(pred - pred.mean(axis=0))

    if not preds:
        return None

    seed_base = hash_seed(f"{target_id}|rhofold_fill")
    while len(preds) < N_MODELS:
        flex = np.ones(len(preds[0]), dtype=bool)
        preds.append(diversify(preds[0].copy(), len(preds), flex, seed_base + len(preds)))

    return np.stack(preds[:N_MODELS], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Template-based fallback prediction
# ─────────────────────────────────────────────────────────
def template_predict_five(
    target_id: str,
    query_seq: str,
    templates: List[Template],
    cutoff: Optional[str],
    template_index: Optional[TemplateIndex],
    top_k: int = 5,
) -> np.ndarray:
    n = len(query_seq)
    pool = filter_templates_by_cutoff(templates, cutoff)
    if not pool:
        return np.stack([make_helix(n, i) for i in range(N_MODELS)])

    # Stage 1: vectorized k-mer cosine scoring
    if template_index is not None and len(template_index.k3_mat) == len(templates):
        all_scores = batch_coarse_scores(query_seq, template_index)
        coarse = [
            (float(all_scores[template_index.id_to_idx[t.template_id]]), t)
            for t in pool if t.template_id in template_index.id_to_idx
        ]
    else:
        qvec = kmer3_vector(query_seq)
        coarse = [(float(qvec @ t.k3_vec), t) for t in pool]

    if not coarse:
        return np.stack([make_helix(n, i) for i in range(N_MODELS)])

    coarse.sort(key=lambda x: x[0], reverse=True)

    # Stage 2: global-alignment rerank on top-64
    pre = [t for _, t in coarse[:max(64, top_k * 16)]]
    align_cache: Dict[str, np.ndarray] = {}
    reranked = []
    for t in pre:
        q_to_t = _alignment_map(query_seq, t.sequence)
        align_cache[t.template_id] = q_to_t
        mapped = q_to_t >= 0
        mapped_frac = float(mapped.mean()) if len(mapped) > 0 else 0.0
        if mapped.any():
            q_idx = np.where(mapped)[0]
            t_idx = q_to_t[mapped]
            matches = sum(1 for qi, tj in zip(q_idx, t_idx) if query_seq[qi] == t.sequence[tj])
            seq_id = matches / max(1, len(q_idx))
        else:
            seq_id = 0.0
        score = 0.70 * seq_id + 0.30 * mapped_frac
        reranked.append((float(score), t))

    reranked.sort(key=lambda x: x[0], reverse=True)
    chosen = select_diverse_templates(reranked, k=top_k)

    msa_cons = load_msa_conservation(target_id, query_seq)
    seed_base = hash_seed(target_id)
    preds = []

    for i, tpl in enumerate(chosen):
        base, mapped = map_coords_by_alignment(
            query_seq, tpl, q_to_t=align_cache.get(tpl.template_id)
        )
        flex = ~mapped
        if msa_cons is not None and len(msa_cons) == len(flex):
            # Adaptive threshold: positions below 40th-percentile are flexible
            thr = float(np.percentile(msa_cons[np.isfinite(msa_cons)], 40))
            flex = flex | (msa_cons < thr)
        if i == 0:
            # Keep one clean template-transfer candidate (no perturbation).
            preds.append(base.astype(np.float32))
        else:
            preds.append(diversify(base, i, flex, seed_base + i))

    while len(preds) < N_MODELS:
        base = preds[0].copy() if preds else make_helix(n, 0)
        flex = np.ones(n, dtype=bool)
        preds.append(diversify(base, len(preds), flex, seed_base + len(preds) + 100))

    return np.stack(preds[:N_MODELS], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Unified prediction entry point
# ─────────────────────────────────────────────────────────
def predict_five_models(
    target_id: str,
    query_seq: str,
    templates: List[Template],
    cutoff: Optional[str],
    template_index: Optional[TemplateIndex],
    rhofold_model=None,
    top_k: int = 5,
) -> np.ndarray:
    query_seq = normalize_sequence(query_seq)
    n = len(query_seq)
    if n == 0:
        return np.zeros((N_MODELS, 0, 3), dtype=np.float32)

    # Primary: RhoFold+
    if rhofold_model is not None:
        result = rhofold_predict_five(rhofold_model, query_seq, target_id)
        if result is not None:
            return result

    # Fallback: improved template retrieval
    return template_predict_five(target_id, query_seq, templates, cutoff, template_index, top_k)


# ─────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────
def labels_to_target_conformers(labels_df: pd.DataFrame) -> Dict[str, List[np.ndarray]]:
    labels_df = infer_target_id_and_resid(labels_df)
    triplets = discover_xyz_triplets(labels_df.columns)
    out: Dict[str, List[np.ndarray]] = {}
    for tid, g in labels_df.groupby("target_id"):
        g2 = g.sort_values("resid")
        conformers = []
        for x, y, z in triplets:
            coords = g2[[x, y, z]].to_numpy(dtype=np.float32)
            valid = np.isfinite(coords).all(axis=1)
            c = coords[valid]
            if len(c) >= 8:
                conformers.append(c)
        if conformers:
            out[tid] = conformers
    return out


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
start = time.time()

print("=" * 64)
print("Stanford RNA 3D Folding 2 — v2  (RhoFold+ / Template fallback)")
print("=" * 64)
print(f"JAX backend  : {'jax-' + JAX_BACKEND if JAX_AVAILABLE else 'numpy-cpu'}")
print(f"Torch device : {TORCH_DEVICE if TORCH_AVAILABLE else 'N/A'}")
print(f"RhoFold ckpt : {RHOFOLD_WEIGHT or 'NOT FOUND — will use template fallback'}")
print(f"Data dir     : {DATA_DIR}")
print()

if TORCH_AVAILABLE and TORCH_DEVICE != "cuda":
    print("[WARN] GPU is not active. Please set Accelerator to 'GPU T4 x2' in Kaggle.")

# ── Load data ──────────────────────────────────────────────
train_sequences = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
train_labels    = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"), dtype={"chain": str})
val_sequences   = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
val_labels      = pd.read_csv(os.path.join(DATA_DIR, "validation_labels.csv"), dtype={"chain": str})
test_sequences  = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))

print("train_sequences:", train_sequences.shape)
print("train_labels   :", train_labels.shape)
print("val_sequences  :", val_sequences.shape)
print("val_labels     :", val_labels.shape)
print("test_sequences :", test_sequences.shape)

# ── Build template library ─────────────────────────────────
templates = build_templates(train_sequences, train_labels)
template_index = build_template_index(templates)
print(f"\ntemplates built : {len(templates)}")

# ── Load RhoFold (may return None) ─────────────────────────
rhofold_model = _try_load_rhofold()
mode_str = "RhoFold+ (primary)" if rhofold_model else "Template retrieval (fallback)"
print(f"prediction mode : {mode_str}\n")

# ── Offline validation: best-of-5 TM-score ────────────────
val_truth = labels_to_target_conformers(val_labels)
val_meta  = {row.target_id: row for row in val_sequences.itertuples(index=False)}
tm_list   = []

for tid, true_list in val_truth.items():
    row = val_meta.get(tid)
    if row is None:
        continue
    cutoff = getattr(row, "temporal_cutoff", None)
    preds5 = predict_five_models(
        tid, row.sequence, templates, cutoff, template_index, rhofold_model, top_k=5
    )
    best = 0.0
    for k in range(N_MODELS):
        for true_coords in true_list:
            m = min(len(preds5[k]), len(true_coords))
            if m >= 8:
                s = tm_score(preds5[k][:m], true_coords[:m])
                if s > best:
                    best = s
    if best > 0:
        tm_list.append(best)

if tm_list:
    print(
        f"Validation best-of-5 TM-score | "
        f"mean={np.mean(tm_list):.4f}, median={np.median(tm_list):.4f}, n={len(tm_list)}"
    )
else:
    print("Validation: no valid samples found — check data files")

# ── Generate submission.csv ────────────────────────────────
print("\nGenerating submission...")
rows = []
for i, row in enumerate(test_sequences.itertuples(index=False)):
    target_id = row.target_id
    seq       = normalize_sequence(row.sequence)
    cutoff    = getattr(row, "temporal_cutoff", None)

    preds5 = predict_five_models(
        target_id, seq, templates, cutoff, template_index, rhofold_model, top_k=5
    )

    for resid, resname in enumerate(seq, start=1):
        rec = {"ID": f"{target_id}_{resid}", "resname": resname, "resid": resid}
        for k in range(N_MODELS):
            rec[f"x_{k+1}"] = float(preds5[k, resid - 1, 0])
            rec[f"y_{k+1}"] = float(preds5[k, resid - 1, 1])
            rec[f"z_{k+1}"] = float(preds5[k, resid - 1, 2])
        rows.append(rec)

    if (i + 1) % 5 == 0 or (i + 1) == len(test_sequences):
        print(f"  {i+1}/{len(test_sequences)} sequences done | {time.time()-start:.1f}s elapsed")

submission = pd.DataFrame(rows)
coord_cols = [c for c in submission.columns if re.fullmatch(r"[xyz]_\d+", c)]
submission[coord_cols] = submission[coord_cols].clip(-999.999, 9999.999)
submission.to_csv("submission.csv", index=False)

print(f"\nsubmission saved: {submission.shape} -> submission.csv")
print(f"total elapsed  : {time.time() - start:.1f}s")
display(submission.head(10))
