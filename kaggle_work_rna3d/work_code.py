import os
import re
import time
import zlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    JAX_BACKEND = jax.default_backend()
except Exception:
    jax = None
    jnp = None
    JAX_AVAILABLE = False
    JAX_BACKEND = "cpu"

SEED = 42
np.random.seed(SEED)

INPUT_ROOT = "/kaggle/input"


def resolve_data_dir() -> str:
    required = {
        "train_sequences.csv",
        "train_labels.csv",
        "validation_sequences.csv",
        "validation_labels.csv",
        "test_sequences.csv",
    }

    def has_required(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, fn)) for fn in required)

    preferred_paths = [
        "/kaggle/input/stanford-rna-3d-folding-part-2",
        "/kaggle/input/stanford-rna-3d-folding-2",
        "/kaggle/input/competitions/stanford-rna-3d-folding-part-2",
        "/kaggle/input/competitions/stanford-rna-3d-folding-2",
    ]
    for path in preferred_paths:
        if has_required(path):
            return path

    search_roots = ["/kaggle/input", "/kaggle/input/competitions"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for cur, dirs, files in os.walk(root):
            if required.issubset(set(files)):
                return cur

            # Bound traversal depth for speed in Kaggle runtime.
            depth = cur[len(root):].count(os.sep)
            if depth >= 4:
                dirs[:] = []

    mounted = sorted(os.listdir(INPUT_ROOT)) if os.path.isdir(INPUT_ROOT) else []
    raise FileNotFoundError(
        f"Could not locate competition data under /kaggle/input. Mounted inputs: {mounted}"
    )


DATA_DIR = resolve_data_dir()
NUC_SET = set("ACGU")
KMER3_BASES = "ACGU"
KMER3_INDEX = {
    a + b + c: i
    for i, (a, b, c) in enumerate(
        (x, y, z) for x in KMER3_BASES for y in KMER3_BASES for z in KMER3_BASES
    )
}
KMER3_DIM = len(KMER3_INDEX)


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
        y = f"y_{i}"
        z = f"z_{i}"
        if y in columns and z in columns:
            triplets.append((x, y, z))
    return triplets
def infer_release_date_map(seq_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    if "target_id" not in seq_df.columns:
        return {}
    candidates = [
        "release_date",
        "released_at",
        "deposition_date",
        "structure_release_date",
        "temporal_cutoff",
    ]
    col = next((c for c in candidates if c in seq_df.columns), None)
    if col is None:
        return {}
    dates = pd.to_datetime(seq_df[col], errors="coerce")
    return {tid: dt for tid, dt in zip(seq_df["target_id"], dates)}
@dataclass
class Template:
    template_id: str
    target_id: str
    conformer_idx: int
    sequence: str
    coords: np.ndarray
    obs_mask: np.ndarray
    k3: set
    k3_vec: np.ndarray
    release_date: pd.Timestamp


@dataclass
class TemplateIndex:
    id_to_idx: Dict[str, int]
    k3_mat: np.ndarray
    seq_len: np.ndarray
    k3_mat_jax: Optional["jnp.ndarray"] = None
    seq_len_jax: Optional["jnp.ndarray"] = None


def kmer_set(seq: str, k: int = 3):
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def kmer3_vector(seq: str) -> np.ndarray:
    vec = np.zeros(KMER3_DIM, dtype=np.float32)
    if len(seq) < 3:
        return vec
    for i in range(len(seq) - 2):
        idx = KMER3_INDEX.get(seq[i : i + 3])
        if idx is not None:
            vec[idx] += 1.0
    s = float(vec.sum())
    if s > 0:
        vec /= s
    return vec


def build_template_index(templates: List[Template]) -> TemplateIndex:
    if not templates:
        return TemplateIndex(
            id_to_idx={},
            k3_mat=np.zeros((0, KMER3_DIM), dtype=np.float32),
            seq_len=np.zeros(0, dtype=np.float32),
        )

    id_to_idx = {tpl.template_id: i for i, tpl in enumerate(templates)}
    k3_mat = np.stack([tpl.k3_vec for tpl in templates], axis=0).astype(np.float32)
    seq_len = np.asarray([len(tpl.sequence) for tpl in templates], dtype=np.float32)

    if JAX_AVAILABLE:
        return TemplateIndex(
            id_to_idx=id_to_idx,
            k3_mat=k3_mat,
            seq_len=seq_len,
            k3_mat_jax=jnp.asarray(k3_mat),
            seq_len_jax=jnp.asarray(seq_len),
        )
    return TemplateIndex(id_to_idx=id_to_idx, k3_mat=k3_mat, seq_len=seq_len)


def _coarse_scores_numpy(
    query_vec: np.ndarray, query_len: float, k3_mat: np.ndarray, seq_len: np.ndarray
) -> np.ndarray:
    if len(k3_mat) == 0:
        return np.zeros(0, dtype=np.float32)
    qnorm = np.linalg.norm(query_vec) + 1e-6
    tnorm = np.linalg.norm(k3_mat, axis=1) + 1e-6
    cosine = (k3_mat @ query_vec) / (tnorm * qnorm)
    len_ratio = np.minimum(query_len, seq_len) / np.maximum(query_len, seq_len)
    return (0.85 * cosine + 0.15 * len_ratio).astype(np.float32)


if JAX_AVAILABLE:

    @jax.jit
    def _coarse_scores_jax(query_vec, query_len, k3_mat, seq_len):
        qnorm = jnp.linalg.norm(query_vec) + 1e-6
        tnorm = jnp.linalg.norm(k3_mat, axis=1) + 1e-6
        cosine = (k3_mat @ query_vec) / (tnorm * qnorm)
        len_ratio = jnp.minimum(query_len, seq_len) / jnp.maximum(query_len, seq_len)
        return 0.85 * cosine + 0.15 * len_ratio

else:
    _coarse_scores_jax = None


def batch_coarse_scores(query_seq: str, template_index: TemplateIndex) -> np.ndarray:
    query_vec = kmer3_vector(query_seq)
    query_len = np.float32(max(1, len(query_seq)))
    if JAX_AVAILABLE and template_index.k3_mat_jax is not None:
        scores = _coarse_scores_jax(
            jnp.asarray(query_vec),
            query_len,
            template_index.k3_mat_jax,
            template_index.seq_len_jax,
        )
        return np.asarray(scores, dtype=np.float32)
    return _coarse_scores_numpy(
        query_vec, query_len, template_index.k3_mat, template_index.seq_len
    )
def fill_nan_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = coords.astype(np.float32).copy()
    obs = np.isfinite(out).all(axis=1)
    n = len(out)
    if n == 0:
        return out, obs
    if not obs.any():
        return np.zeros_like(out), obs
    idx = np.arange(n)
    for d in range(3):
        v = out[:, d]
        good = np.isfinite(v)
        out[:, d] = np.interp(idx, idx[good], v[good]).astype(np.float32)
    return out, obs
def quick_similarity(query: str, tpl: Template) -> float:
    qk3 = kmer_set(query, k=3)
    inter = len(qk3 & tpl.k3)
    union = max(1, len(qk3 | tpl.k3))
    jaccard = inter / union
    len_ratio = min(len(query), len(tpl.sequence)) / max(len(query), len(tpl.sequence))
    return 0.7 * jaccard + 0.3 * len_ratio
def alignment_similarity(query: str, target: str) -> float:
    return float(SequenceMatcher(a=query, b=target, autojunk=False).ratio())
def sequence_similarity(query: str, tpl: Template, quick_only: bool = False) -> float:
    coarse = quick_similarity(query, tpl)
    if quick_only:
        return coarse
    align = alignment_similarity(query, tpl.sequence)
    return 0.45 * coarse + 0.55 * align
def template_diversity_penalty(a: Template, b: Template) -> float:
    inter = len(a.k3 & b.k3)
    union = max(1, len(a.k3 | b.k3))
    return inter / union
def select_diverse_templates(scored: List[Tuple[float, Template]], k: int = 5) -> List[Template]:
    if not scored:
        return []
    selected: List[Template] = []
    remaining = scored.copy()
    # MMR: balance relevance and sequence-level diversity
    lambda_rel = 0.78
    while remaining and len(selected) < k:
        if not selected:
            best_idx = int(np.argmax([s for s, _ in remaining]))
            selected.append(remaining.pop(best_idx)[1])
            continue
        mmr_scores = []
        for rel, tpl in remaining:
            max_sim = max(template_diversity_penalty(tpl, s) for s in selected)
            mmr_scores.append(lambda_rel * rel - (1.0 - lambda_rel) * max_sim)
        best_idx = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best_idx)[1])
    return selected
def resample_coords_linear(coords: np.ndarray, new_len: int) -> np.ndarray:
    old_len = len(coords)
    if old_len == new_len:
        return coords.copy()
    if old_len <= 1:
        return np.repeat(coords, new_len, axis=0)
    old_x = np.linspace(0.0, 1.0, old_len)
    new_x = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((new_len, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(new_x, old_x, coords[:, d])
    return out
def nw_alignment_map(query: str, target: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> np.ndarray:
    n, m = len(query), len(target)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if m == 0:
        return np.full(n, -1, dtype=np.int32)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0=diag,1=up,2=left
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap
        trace[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap
        trace[0, j] = 2
    for i in range(1, n + 1):
        qi = query[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if qi == target[j - 1] else mismatch)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            if diag >= up and diag >= left:
                score[i, j] = diag
                trace[i, j] = 0
            elif up >= left:
                score[i, j] = up
                trace[i, j] = 1
            else:
                score[i, j] = left
                trace[i, j] = 2
    q_to_t = np.full(n, -1, dtype=np.int32)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i, j] == 0:
            q_to_t[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i, j] == 1):
            i -= 1
        else:
            j -= 1
    return q_to_t
def estimate_bond_len(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 6.0
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 6.0
    return float(np.clip(np.median(d), 4.5, 7.5))
def unit_with_scale(vec: np.ndarray, scale: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return np.array([scale, 0.0, 0.0], dtype=np.float32)
    return (vec / n * scale).astype(np.float32)
def map_coords_by_alignment(query_seq: str, tpl: Template) -> Tuple[np.ndarray, np.ndarray]:
    n = len(query_seq)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)
    q_to_t = nw_alignment_map(query_seq, tpl.sequence)
    out = np.full((n, 3), np.nan, dtype=np.float32)
    mapped = np.zeros(n, dtype=bool)
    for qi, tj in enumerate(q_to_t):
        if 0 <= tj < len(tpl.coords):
            out[qi] = tpl.coords[tj]
            mapped[qi] = True
    if mapped.sum() == 0:
        return resample_coords_linear(tpl.coords, n), np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for d in range(3):
        out[:, d] = np.interp(idx, idx[mapped], out[mapped, d]).astype(np.float32)
    # Extrapolate ends with near-constant bond length instead of flat extension.
    bond = estimate_bond_len(tpl.coords)
    first = int(np.argmax(mapped))
    last = int(n - 1 - np.argmax(mapped[::-1]))
    if first > 0:
        if first + 1 < n:
            direction = unit_with_scale(out[first] - out[first + 1], bond)
        else:
            direction = np.array([bond, 0.0, 0.0], dtype=np.float32)
        for i in range(first - 1, -1, -1):
            out[i] = out[i + 1] + direction
    if last < n - 1:
        if last - 1 >= 0:
            direction = unit_with_scale(out[last] - out[last - 1], bond)
        else:
            direction = np.array([bond, 0.0, 0.0], dtype=np.float32)
        for i in range(last + 1, n):
            out[i] = out[i - 1] + direction
    # Keep local steps in a plausible C1' range.
    for i in range(1, n):
        step = out[i] - out[i - 1]
        dist = float(np.linalg.norm(step))
        if dist < 1e-6:
            out[i] = out[i - 1] + np.array([bond, 0.0, 0.0], dtype=np.float32)
            continue
        clipped = float(np.clip(dist, 3.5, 8.5))
        if abs(clipped - dist) > 1e-3:
            out[i] = out[i - 1] + step / dist * clipped
    return out.astype(np.float32), mapped
def smooth_noise(noise: np.ndarray) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
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
    out = coords.astype(np.float32).copy()
    rng = np.random.default_rng(seed + 97 * model_idx)
    rigid_scales = [0.02, 0.04, 0.05, 0.06, 0.07]
    flex_scales = [0.08, 0.18, 0.28, 0.38, 0.50]
    rigid = rigid_scales[min(model_idx, 4)]
    flex = flex_scales[min(model_idx, 4)]
    sigma = np.where(flex_mask[:, None], flex, rigid).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    noise = smooth_noise(noise)
    out = out + sigma * noise
    centered = out - out.mean(axis=0, keepdims=True)
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    if model_idx in (1, 3, 4):
        angle = 0.06 * model_idx
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        centered = centered @ r.T
    if model_idx in (2, 3, 4):
        centered[:, 2] += (0.5 + 0.25 * model_idx) * t
        centered[:, 0] += 0.35 * model_idx * (t * t - np.mean(t * t))
    return centered.astype(np.float32)
def make_helix(length: int, model_idx: int) -> np.ndarray:
    i = np.arange(length, dtype=np.float32)
    radius = 8.0 + 0.7 * model_idx
    theta = 0.56 + 0.02 * model_idx
    phase = 0.6 * model_idx
    rise = 2.3 + 0.06 * model_idx
    x = radius * np.cos(theta * i + phase)
    y = radius * np.sin(theta * i + phase)
    z = rise * i
    return np.stack([x, y, z], axis=1).astype(np.float32)
def load_msa_conservation(target_id: str, query_seq: str, max_rows: int = 128) -> Optional[np.ndarray]:
    msa_path = os.path.join(DATA_DIR, "MSA", f"{target_id}.MSA.fasta")
    if not os.path.exists(msa_path):
        return None
    seqs = []
    cur = []
    with open(msa_path, "r", encoding="utf-8", errors="ignore") as f:
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
    # Prefer a row that ungaps to the current query.
    query_aln = seqs[0]
    for s in seqs:
        ungapped = normalize_sequence(s.replace("-", "").replace(".", ""))
        if ungapped == query_seq:
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
            if base is None:
                continue
            counts[base] += 1
            total += 1
        if total > 0:
            cons_col[c] = max(counts.values()) / total
    out = np.full(len(query_seq), np.nan, dtype=np.float32)
    qi = 0
    for c, ch in enumerate(query_aln):
        if ch in ("-", "."):
            continue
        if qi < len(query_seq):
            out[qi] = cons_col[c]
            qi += 1
    if np.isnan(out).all():
        return None
    fill_val = float(np.nanmean(out)) if not np.isnan(np.nanmean(out)) else 0.5
    out = np.where(np.isfinite(out), out, fill_val).astype(np.float32)
    return out
def build_templates(seq_df: pd.DataFrame, labels_df: pd.DataFrame) -> List[Template]:
    seq_map = {
        row.target_id: normalize_sequence(row.sequence)
        for row in seq_df.itertuples(index=False)
    }
    release_map = infer_release_date_map(seq_df)
    labels_df = infer_target_id_and_resid(labels_df)
    triplets = discover_xyz_triplets(labels_df.columns)
    if not triplets:
        raise ValueError("labels ???? x_i/y_i/z_i ???")
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
            centered = filled - filled.mean(axis=0, keepdims=True)
            templates.append(
                Template(
                    template_id=f"{tid}_c{ci}",
                    target_id=tid,
                    conformer_idx=ci,
                    sequence=seq[:max_len],
                    coords=centered.astype(np.float32),
                    obs_mask=obs[:max_len],
                    k3=kmer_set(seq[:max_len], 3),
                    k3_vec=kmer3_vector(seq[:max_len]),
                    release_date=release_map.get(tid, pd.NaT),
                )
            )
    return templates
def filter_templates_by_cutoff(templates: List[Template], cutoff: Optional[str]) -> List[Template]:
    if cutoff is None or (isinstance(cutoff, float) and np.isnan(cutoff)):
        return templates
    cutoff_ts = pd.to_datetime(cutoff, errors="coerce")
    if pd.isna(cutoff_ts):
        return templates
    filtered = [tpl for tpl in templates if pd.isna(tpl.release_date) or tpl.release_date < cutoff_ts]
    return filtered if filtered else templates
def predict_five_models(
    target_id: str,
    query_seq: str,
    templates: List[Template],
    cutoff: Optional[str] = None,
    template_index: Optional[TemplateIndex] = None,
    top_k: int = 5,
) -> np.ndarray:
    query_seq = normalize_sequence(query_seq)
    n = len(query_seq)
    if n == 0:
        return np.zeros((5, 0, 3), dtype=np.float32)
    pool = filter_templates_by_cutoff(templates, cutoff)
    if not pool:
        return np.stack([make_helix(n, i) for i in range(5)], axis=0)

    # Two-stage retrieval: fast vectorized prefilter (TPU/JAX when available), then alignment rerank.
    if template_index is not None and len(template_index.k3_mat) == len(templates):
        all_scores = batch_coarse_scores(query_seq, template_index)
        coarse_scored = [
            (float(all_scores[template_index.id_to_idx[tpl.template_id]]), tpl)
            for tpl in pool
            if tpl.template_id in template_index.id_to_idx
        ]
    else:
        coarse_scored = [(sequence_similarity(query_seq, tpl, quick_only=True), tpl) for tpl in pool]

    if not coarse_scored:
        return np.stack([make_helix(n, i) for i in range(5)], axis=0)

    coarse_scored.sort(key=lambda x: x[0], reverse=True)
    pre_n = min(len(coarse_scored), max(64, top_k * 16))
    pre_candidates = [tpl for _, tpl in coarse_scored[:pre_n]]
    reranked = [(sequence_similarity(query_seq, tpl, quick_only=False), tpl) for tpl in pre_candidates]
    reranked.sort(key=lambda x: x[0], reverse=True)
    chosen = select_diverse_templates(reranked, k=top_k)
    msa_cons = load_msa_conservation(target_id, query_seq)
    preds = []
    for i, tpl in enumerate(chosen):
        base, mapped = map_coords_by_alignment(query_seq, tpl)
        # Flexible regions: alignment gaps + low-conservation positions from MSA.
        flex_mask = ~mapped
        if msa_cons is not None and len(msa_cons) == len(flex_mask):
            flex_mask = flex_mask | (msa_cons < 0.60)
        seed = hash_seed(f"{target_id}|{tpl.template_id}")
        preds.append(diversify(base, i, flex_mask, seed))
    while len(preds) < 5:
        if preds:
            base = preds[0].copy()
        else:
            base = make_helix(n, 0)
        flex_mask = np.ones(n, dtype=bool)
        seed = hash_seed(f"{target_id}|fallback|{len(preds)}")
        preds.append(diversify(base, len(preds), flex_mask, seed))
    return np.stack(preds[:5], axis=0).astype(np.float32)
def kabsch_align(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    true_c = true - true.mean(axis=0, keepdims=True)
    h = pred_c.T @ true_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    pred_rot = pred_c @ r
    return pred_rot, true_c
def tm_d0(length: int) -> float:
    if length < 12:
        return 0.3
    if length < 16:
        return 0.4
    if length < 20:
        return 0.5
    if length < 24:
        return 0.6
    if length < 30:
        return 0.7
    return 0.6 * (length - 0.5) ** 0.5 - 2.5
def tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    m = min(len(pred), len(true))
    if m == 0:
        return 0.0
    pred_a, true_a = kabsch_align(pred[:m], true[:m])
    d0 = tm_d0(m)
    dist = np.sqrt(((pred_a - true_a) ** 2).sum(axis=1))
    return float(np.mean(1.0 / (1.0 + (dist / d0) ** 2)))
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
            coords = coords[valid]
            if len(coords) >= 8:
                conformers.append(coords.astype(np.float32))
        if conformers:
            out[tid] = conformers
    return out
start = time.time()
required_files = [
    "train_sequences.csv",
    "train_labels.csv",
    "validation_sequences.csv",
    "validation_labels.csv",
    "test_sequences.csv",
]
for fn in required_files:
    p = os.path.join(DATA_DIR, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")
train_sequences = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
val_sequences = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
val_labels = pd.read_csv(os.path.join(DATA_DIR, "validation_labels.csv"))
test_sequences = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
print("train_sequences:", train_sequences.shape)
print("train_labels:", train_labels.shape)
print("validation_sequences:", val_sequences.shape)
print("validation_labels:", val_labels.shape)
print("test_sequences:", test_sequences.shape)
print("retrieval_backend:", f"jax-{JAX_BACKEND}" if JAX_AVAILABLE else "numpy-cpu")
# ---------- ????????????????????????? ----------
templates = build_templates(train_sequences, train_labels)
template_index = build_template_index(templates)
print(f"templates built: {len(templates)}")
# ---------- ?????best-of-5 TM-score? ----------
val_truth = labels_to_target_conformers(val_labels)
val_meta = {row.target_id: row for row in val_sequences.itertuples(index=False)}
tm_list = []
for tid, true_list in val_truth.items():
    row = val_meta.get(tid)
    if row is None:
        continue
    cutoff = getattr(row, "temporal_cutoff", None)
    preds5 = predict_five_models(
        tid, row.sequence, templates, cutoff=cutoff, template_index=template_index, top_k=5
    )
    best_target = 0.0
    for k in range(5):
        pred = preds5[k]
        best_model = 0.0
        for true_coords in true_list:
            m = min(len(pred), len(true_coords))
            if m < 8:
                continue
            score = tm_score(pred[:m], true_coords[:m])
            if score > best_model:
                best_model = score
        if best_model > best_target:
            best_target = best_model
    if best_target > 0:
        tm_list.append(best_target)
if tm_list:
    print(
        f"Validation best-of-5 TM-score | mean={np.mean(tm_list):.4f}, "
        f"median={np.median(tm_list):.4f}, n={len(tm_list)}"
    )
else:
    print("Validation ????????????????")
# ---------- ?? submission.csv ----------
rows = []
for row in test_sequences.itertuples(index=False):
    target_id = row.target_id
    seq = normalize_sequence(row.sequence)
    cutoff = getattr(row, "temporal_cutoff", None)
    preds5 = predict_five_models(
        target_id, seq, templates, cutoff=cutoff, template_index=template_index, top_k=5
    )
    for resid, resname in enumerate(seq, start=1):
        rec = {
            "ID": f"{target_id}_{resid}",
            "resname": resname,
            "resid": resid,
        }
        for k in range(5):
            rec[f"x_{k + 1}"] = float(preds5[k, resid - 1, 0])
            rec[f"y_{k + 1}"] = float(preds5[k, resid - 1, 1])
            rec[f"z_{k + 1}"] = float(preds5[k, resid - 1, 2])
        rows.append(rec)
submission = pd.DataFrame(rows)
coord_cols = [c for c in submission.columns if c.startswith(("x_", "y_", "z_"))]
submission[coord_cols] = submission[coord_cols].clip(lower=-999.999, upper=9999.999)
submission.to_csv("submission.csv", index=False)
print("submission saved:", submission.shape, "-> submission.csv")
print(f"elapsed: {time.time() - start:.1f}s")
display(submission.head(10))

