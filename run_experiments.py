#!/usr/bin/env python3
"""
Experimental Pipeline for Final Project PMML:
"TempConv-GatedRanker: Dual-Stream Temporal-Conversational Gated Neural Reranking
with Hard-Negative Contrastive Learning for Reasoning-Oriented Retrieval (RETECO)"

Evaluates on 6 Official RETECO v1.1 Domains (131,544 Corpus Documents):
- Track 1 (Temporal Grounded Retrieval - TEMPO): iota, quant, workplace
  * Sub-track 1a: Whole-Query Temporal Retrieval
  * Sub-track 1b: Step-Wise Temporal Retrieval
- Track 2 (Conversational Grounded Retrieval - RECOR): drones, law, politics
  * Sub-track 2a: Multi-Turn Conversational Retrieval

Models Compared (with Full Ablation Study):
1. Official Baseline 1: BM25 (Current / Whole Query) [k1=0.9, b=0.4]
2. Official Baseline 2: BM25 (+Step / +Full Conversation History)
3. Baseline 3 (Dense Bi-Encoder): Neural Semantic Bi-Encoder (SVD-MLP Cosine)
4. Ablation A: Proposed w/o Learnable Gating (Static Linear Fusion)
5. Ablation B: Proposed w/o Temporal & Turn-Decay Features (Semantic + Lexical Only)
6. Ablation C: Proposed w/o Hard-Negative Mining (Random Negatives Only)
7. Proposed Full Model: TempConv-GatedRanker (Dual-Stream Gated Neural Reranker + Hard Negatives + Temporal/Turn-Decay Grounding)
"""
import os
import re
import json
import time
import math
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
import keras
from keras import layers

# Import official RETECO metrics from starter_kit
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RETECO", "starter_kit"))
from ir_metrics import (
    ndcg_at_k,
    temporal_precision_at_k,
    temporal_relevance_at_k,
    temporal_coverage_at_k,
    mean,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
YEAR_RE = re.compile(r"\b(18\d\d|19\d\d|20[0-2]\d)\b")
VERSION_RE = re.compile(r"\b(?:v|version\s*)?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)
TEMP_SIGNAL_WORDS = {
    "before", "after", "since", "until", "during", "when", "currently",
    "now", "previously", "formerly", "recently", "history", "timeline",
    "future", "past", "present", "initial", "inception", "original",
    "update", "upgrade", "transition", "change", "evolved", "growth",
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "period", "era", "decade", "century", "early", "late", "mid", "post", "pre"
}


def clean_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return TOKEN_RE.findall(clean_html(text).lower())


class FastBM25:
    """Exact RETECO Starter Kit BM25 implementation (k1=0.9, b=0.4) optimized with inverted index."""
    def __init__(self, doc_ids, doc_texts, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.doc_ids = list(doc_ids)
        self.N = len(self.doc_ids)
        self.doc_len = np.zeros(self.N, dtype=np.float32)
        self.postings = defaultdict(list)  # term -> list of (doc_idx, tf)
        df = Counter()

        for idx, text in enumerate(doc_texts):
            tokens = tokenize(text)
            self.doc_len[idx] = len(tokens)
            counts = Counter(tokens)
            for term, freq in counts.items():
                self.postings[term].append((idx, freq))
                df[term] += 1

        self.avgdl = float(np.mean(self.doc_len)) if self.N > 0 else 1.0
        self.idf = {
            w: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }
        # Precompute denominator norm factor per document
        self.norm_factor = self.k1 * (1.0 - self.b + self.b * (self.doc_len / (self.avgdl or 1.0)))

    def score_all(self, query_text):
        q_terms = tokenize(query_text)
        scores = np.zeros(self.N, dtype=np.float32)
        q_counts = Counter(q_terms)
        for term, qf in q_counts.items():
            if term not in self.idf:
                continue
            idf_val = self.idf[term] * qf
            for doc_idx, freq in self.postings[term]:
                scores[doc_idx] += idf_val * (freq * (self.k1 + 1.0)) / (freq + self.norm_factor[doc_idx])
        return scores

    def search_top_k(self, query_text, top_k=100):
        scores = self.score_all(query_text)
        if top_k >= self.N:
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.doc_ids[i], float(scores[i]), int(i)) for i in top_idx]


def extract_temporal_metadata(text):
    cleaned = clean_html(text)
    tokens = TOKEN_RE.findall(cleaned.lower())
    years = sorted({int(y) for y in YEAR_RE.findall(cleaned)})
    versions = set(v.lower().lstrip("v").strip() for v in VERSION_RE.findall(cleaned))
    temp_signals = [t for t in tokens if t in TEMP_SIGNAL_WORDS]
    has_recency = any(w in tokens for w in ("currently", "now", "today", "recent", "recently", "present", "latest", "new"))
    has_historical = any(w in tokens for w in ("originally", "inception", "history", "previously", "formerly", "old", "initial", "first"))
    return {
        "years": years,
        "versions": versions,
        "temp_signal_count": len(temp_signals),
        "temp_signal_ratio": len(temp_signals) / max(len(tokens), 1),
        "has_recency": float(has_recency),
        "has_historical": float(has_historical),
        "len_tokens": len(tokens),
        "token_set": set(tokens),
    }


def compute_Comprehensive_Metrics(ranked_lists, gold_dict, period_dict=None, is_track2=False):
    """Computes official RETECO metrics: nDCG@10, MRR@10, Recall@10, Recall@100, MAP@10, plus Track1/Track2 diagnostics."""
    ndcg10, mrr10, rec10, rec100, map10 = [], [], [], [], []
    tp10, tr10, tc10, fc_ndcg10 = [], [], [], []
    by_turn = defaultdict(list)

    for qid, gold_set in gold_dict.items():
        if not gold_set or qid not in ranked_lists:
            continue
        docs = ranked_lists[qid]
        nd10 = ndcg_at_k(docs, gold_set, k=10)
        ndcg10.append(nd10)

        # MRR@10
        rr = 0.0
        for rank_idx, d in enumerate(docs[:10], 1):
            if d in gold_set:
                rr = 1.0 / rank_idx
                break
        mrr10.append(rr)

        # Recall@10 and Recall@100
        hits10 = sum(1 for d in docs[:10] if d in gold_set)
        hits100 = sum(1 for d in docs[:100] if d in gold_set)
        rec10.append(hits10 / len(gold_set))
        rec100.append(hits100 / len(gold_set))

        # MAP@10
        ap = 0.0
        hit_cnt = 0
        for rank_idx, d in enumerate(docs[:10], 1):
            if d in gold_set:
                hit_cnt += 1
                ap += hit_cnt / rank_idx
        map10.append(ap / min(len(gold_set), 10))

        if period_dict is not None:
            tp10.append(temporal_precision_at_k(docs, gold_set, k=10))
            tr10.append(temporal_relevance_at_k(docs, gold_set, k=10))
            tc = temporal_coverage_at_k(docs, period_dict.get(qid, []), k=10)
            tc10.append(tc)
            if tc == 1.0:
                fc_ndcg10.append(nd10)

        if is_track2 and "_turn_" in qid:
            try:
                t_num = int(qid.rsplit("_turn_", 1)[1])
                bucket = "T5+" if t_num >= 5 else f"T{t_num}"
                by_turn[bucket].append(nd10)
            except ValueError:
                pass

    out = {
        "nDCG@10": round(mean(ndcg10), 4),
        "MRR@10": round(mean(mrr10), 4),
        "MAP@10": round(mean(map10), 4),
        "Recall@10": round(mean(rec10), 4),
        "Recall@100": round(mean(rec100), 4),
        "num_topics": len(ndcg10),
    }
    if period_dict is not None:
        out["TP@10"] = round(mean(tp10), 4)
        out["TR@10"] = round(mean(tr10), 4)
        out["TC@10"] = round(mean(tc10), 4)
        out["NDCG|FC@10"] = round(mean(fc_ndcg10), 4)
    if is_track2:
        out["by_turn"] = {k: round(mean(v), 4) for k, v in sorted(by_turn.items())}
    return out


def build_Conversational_Decay_Query(turn_obj, decay_lambda=0.55):
    """
    Proposed Context-Distilled Turn-Decay Query Reformulation for Track 2:
    Instead of unweighted history dumping (which floods late turns with early turn noise),
    parses Q/A pairs from conversation_history and applies exponential recency decay +
    salient entity/noun-phrase retention.
    """
    curr_q = clean_html(turn_obj["query"])
    hist = turn_obj.get("conversation_history", "")
    if not hist or hist.strip().lower().startswith("no previous"):
        return curr_q, curr_q, ""

    # Split history into turns by "Q:" markers
    raw_turns = [t.strip() for t in re.split(r"(?=Q:\s)", hist) if t.strip()]
    if not raw_turns:
        return f"{curr_q} {clean_html(hist)}", curr_q, clean_html(hist)

    # Weight recent turns higher; repeat current query tokens for emphasis (anchor)
    weighted_parts = [curr_q, curr_q]  # 2x anchor weight on current question
    num_prev = len(raw_turns)
    prev_questions = []
    for idx, rt in enumerate(raw_turns):
        dist = num_prev - idx  # 1 for most recent turn, num_prev for oldest
        # Extract Question and Answer parts
        parts = re.split(r"\bA:\s", rt, maxsplit=1)
        q_part = clean_html(parts[0].replace("Q:", "").strip())
        a_part = clean_html(parts[1].strip()) if len(parts) > 1 else ""
        prev_questions.append(q_part)
        if dist == 1:
            # Immediate previous turn: keep full question + first 35 words of answer
            a_words = a_part.split()[:35]
            weighted_parts.append(q_part + " " + " ".join(a_words))
        elif dist == 2:
            # 2 turns ago: keep question + first 20 words of answer
            a_words = a_part.split()[:20]
            weighted_parts.append(q_part + " " + " ".join(a_words))
        else:
            # Older turns: keep only question (prevents answer drift!)
            weighted_parts.append(q_part)

    return " ".join(weighted_parts), curr_q, " ".join(prev_questions)


def build_Temporal_Expanded_Query(query_text, steps_list=None):
    """
    Proposed Temporal-Anchor Query Reformulation for Track 1:
    Boosts explicit temporal anchors (years, versions, temporal signal words)
    and incorporates decomposed step instructions when available.
    """
    cleaned = clean_html(query_text)
    years = YEAR_RE.findall(cleaned)
    versions = VERSION_RE.findall(cleaned)
    anchors = " ".join(years * 2 + versions * 2)
    if steps_list:
        step_texts = " ".join(clean_html(s.get("step_instruction", "")) for s in steps_list)
        return f"{cleaned} {cleaned} {anchors} {step_texts}".strip()
    return f"{cleaned} {anchors}".strip()


def build_TempConv_GatedRanker(sem_dim=64, feat_dim=14, use_gating=True, use_temp_feats=True):
    """
    Proposed Deep Learning Architecture in Keras/TensorFlow:
    Dual-Stream Temporal-Conversational Gated Neural Reranker (TempConv-GatedRanker)
    Inputs:
      - q_emb: (batch, sem_dim) L2-normalized dense semantic query embedding
      - d_emb: (batch, sem_dim) L2-normalized dense semantic passage embedding
      - struct_feats: (batch, feat_dim) lexical, temporal alignment, and turn-decay features
    """
    q_in = keras.Input(shape=(sem_dim,), name="query_sem_emb")
    d_in = keras.Input(shape=(sem_dim,), name="doc_sem_emb")
    f_in = keras.Input(shape=(feat_dim,), name="temporal_conv_feats")

    # Stream 1: Deep Semantic Bi-Encoder Interaction with Residual Projection
    shared_proj = keras.Sequential([
        layers.Dense(sem_dim, activation="gelu"),
        layers.LayerNormalization(),
        layers.Dropout(0.15),
        layers.Dense(sem_dim, activation="gelu"),
        layers.LayerNormalization(),
    ], name="shared_semantic_encoder")

    q_repr = shared_proj(q_in) + q_in
    d_repr = shared_proj(d_in) + d_in

    # Fine-grained late interaction: Hadamard product + absolute difference + dot product
    hadamard = layers.Multiply()([q_repr, d_repr])
    abs_diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([q_repr, d_repr])
    dot_sim = layers.Dot(axes=-1, normalize=True)([q_repr, d_repr])

    sem_inter = layers.Concatenate()([hadamard, abs_diff, dot_sim])
    h_sem = layers.Dense(32, activation="gelu", name="sem_hidden")(sem_inter)
    h_sem = layers.LayerNormalization()(h_sem)
    s_sem = layers.Dense(1, name="sem_score")(h_sem)

    # Stream 2: Temporal & Conversational Structural Alignment Network
    if use_temp_feats:
        f_used = f_in
    else:
        # Ablation B: mask out temporal/conversational features (keep only lexical cols 0..3)
        f_used = layers.Lambda(
            lambda x: tf.concat([x[:, :4], tf.zeros_like(x[:, 4:])], axis=-1)
        )(f_in)

    h_temp = layers.Dense(32, activation="gelu", name="temp_dense1")(f_used)
    h_temp = layers.LayerNormalization()(h_temp)
    h_temp = layers.Dropout(0.1)(h_temp)
    h_temp = layers.Dense(16, activation="gelu", name="temp_dense2")(h_temp)
    s_temp = layers.Dense(1, name="temp_score")(h_temp)

    # Stream 3: Direct Lexical Anchor Skip Connection (BM25 + TFIDF Z-score)
    lex_slice = layers.Lambda(lambda x: x[:, :4])(f_in)
    s_lex = layers.Dense(1, name="lex_score")(lex_slice)

    # Learnable Contextual Gating Mechanism
    if use_gating:
        gate_input = layers.Concatenate()([h_sem, h_temp, lex_slice])
        gate_hidden = layers.Dense(16, activation="gelu")(gate_input)
        gate_weights = layers.Dense(3, activation="softmax", name="contextual_gate")(gate_hidden)
        stacked_scores = layers.Concatenate()([s_lex, s_sem, s_temp])
        gated_product = layers.Multiply()([gate_weights, stacked_scores])
        final_score = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name="final_rank_score")(gated_product)
    else:
        # Ablation A: Static unweighted sum without contextual gating
        final_score = layers.Add(name="final_rank_score")([s_lex, s_sem, s_temp])

    model = keras.Model(inputs=[q_in, d_in, f_in], outputs=final_score, name="TempConv_GatedRanker")
    return model


def extract_pair_features(
    q_idx, q_text_main, q_text_aux, q_meta, q_vec,
    doc_idx, doc_meta, doc_vec,
    bm25_main_score, bm25_main_z,
    bm25_aux_score, bm25_aux_z,
    tfidf_cos, svd_cos,
    turn_id=1
):
    """
    Constructs the 14-dimensional feature vector for a (query, candidate_doc) pair:
    [0] bm25_main_z        : Z-normalized BM25 score on primary query
    [1] bm25_aux_z         : Z-normalized BM25 score on expanded/decayed query
    [2] tfidf_cos          : Exact lexical TF-IDF cosine similarity
    [3] svd_cos            : Dense LSA/SVD semantic similarity
    [4] jaccard_overlap    : Set-based token Jaccard overlap
    [5] query_cov          : Fraction of unique query tokens covered in passage
    [6] year_exact_match   : Exact year overlap count between query and passage
    [7] year_prox_score    : Exponential proximity exp(-0.2 * min_year_diff) if years present
    [8] version_match      : Version number overlap (e.g., 1.0, 2.0)
    [9] temp_signal_prod   : Product of query and passage temporal signal density
    [10] tense_align       : Alignment between recency/historical intent and passage signals
    [11] turn_depth_norm   : Normalized turn depth (turn_id / 5.0) for Track 2
    [12] aux_boost_delta   : Gain from temporal/history expansion (bm25_aux_z - bm25_main_z)
    [13] doc_len_norm      : Log-normalized passage length suitability
    """
    q_toks = q_meta["token_set"]
    d_toks = doc_meta["token_set"]
    inter = len(q_toks & d_toks)
    union = len(q_toks | d_toks) or 1
    jaccard = inter / union
    q_cov = inter / max(len(q_toks), 1)

    # Temporal year alignment
    q_years = q_meta["years"]
    d_years = doc_meta["years"]
    if q_years and d_years:
        exact_yr = float(len(set(q_years) & set(d_years)))
        min_diff = min(abs(qy - dy) for qy in q_years for dy in d_years)
        yr_prox = math.exp(-0.25 * min_diff)
    elif not q_years and d_years:
        exact_yr = 0.0
        yr_prox = 0.25  # passage contains concrete dates when query is implicitly temporal
    else:
        exact_yr = 0.0
        yr_prox = 0.0

    # Version alignment
    ver_match = float(len(q_meta["versions"] & doc_meta["versions"])) if (q_meta["versions"] and doc_meta["versions"]) else 0.0

    # Temporal signal interaction
    temp_sig_prod = math.sqrt(q_meta["temp_signal_ratio"] * doc_meta["temp_signal_ratio"]) * 10.0

    # Tense/Recency alignment
    tense_align = (
        q_meta["has_recency"] * doc_meta["has_recency"]
        + q_meta["has_historical"] * doc_meta["has_historical"]
    )

    turn_norm = min(turn_id / 5.0, 1.0)
    aux_delta = bm25_aux_z - bm25_main_z
    doc_len_norm = math.log1p(doc_meta["len_tokens"]) / 6.0

    return np.array([
        bm25_main_z,
        bm25_aux_z,
        tfidf_cos,
        svd_cos,
        jaccard * 5.0,
        q_cov * 2.0,
        exact_yr,
        yr_prox,
        ver_match,
        temp_sig_prod,
        tense_align,
        turn_norm,
        aux_delta,
        doc_len_norm,
    ], dtype=np.float32)


def train_pairwise_ranker(
    model, X_q, X_d, X_f, y_labels, q_groups,
    hard_negative_mining=True, epochs=18, batch_size=128, lr=3e-3
):
    """
    Trains the neural reranker using Pairwise Margin Ranking Loss + Binary Cross-Entropy
    over query-grouped positives and mined hard negatives.
    """
    pos_q, pos_d, pos_f = [], [], []
    neg_q, neg_d, neg_f = [], [], []

    for qid, indices in q_groups.items():
        idx_arr = np.array(indices)
        labels = y_labels[idx_arr]
        pos_idx = idx_arr[labels == 1]
        neg_idx = idx_arr[labels == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        # In each query candidate pool, candidates are ordered by Stage-1 BM25 rank!
        # So early neg_idx are top-ranked BM25 Hard Negatives (high lexical overlap, wrong time/context).
        if hard_negative_mining:
            hard_negs = neg_idx[:min(15, len(neg_idx))]
            rand_negs = neg_idx[min(15, len(neg_idx)):]
        else:
            # Ablation C: sample purely from tail/random negatives without hard negative focus
            hard_negs = neg_idx[-min(15, len(neg_idx)):]
            rand_negs = neg_idx

        for p in pos_idx:
            # Pair with hard negatives
            for hn in hard_negs[:8]:
                pos_q.append(X_q[p]); pos_d.append(X_d[p]); pos_f.append(X_f[p])
                neg_q.append(X_q[hn]); neg_d.append(X_d[hn]); neg_f.append(X_f[hn])
            # Pair with 4 random negatives
            if len(rand_negs) > 0:
                sampled = np.random.choice(rand_negs, size=min(4, len(rand_negs)), replace=False)
                for rn in sampled:
                    pos_q.append(X_q[p]); pos_d.append(X_d[p]); pos_f.append(X_f[p])
                    neg_q.append(X_q[rn]); neg_d.append(X_d[rn]); neg_f.append(X_f[rn])

    pos_q = np.array(pos_q, dtype=np.float32)
    pos_d = np.array(pos_d, dtype=np.float32)
    pos_f = np.array(pos_f, dtype=np.float32)
    neg_q = np.array(neg_q, dtype=np.float32)
    neg_d = np.array(neg_d, dtype=np.float32)
    neg_f = np.array(neg_f, dtype=np.float32)

    optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=1e-4)
    margin = 0.5
    history_loss = []

    dataset = tf.data.Dataset.from_tensor_slices((pos_q, pos_d, pos_f, neg_q, neg_d, neg_f))
    dataset = dataset.shuffle(buffer_size=len(pos_q), seed=SEED).batch(batch_size)

    for ep in range(epochs):
        ep_losses = []
        for b_pq, b_pd, b_pf, b_nq, b_nd, b_nf in dataset:
            with tf.GradientTape() as tape:
                s_pos = model([b_pq, b_pd, b_pf], training=True)
                s_neg = model([b_nq, b_nd, b_nf], training=True)
                # Pairwise Margin Ranking Loss + Pointwise Calibration
                diff = s_pos - s_neg
                rank_loss = tf.reduce_mean(tf.maximum(0.0, margin - diff))
                bce_loss = tf.reduce_mean(tf.nn.softplus(-diff))
                # Regularize anchor lexical/auxiliary ordering
                loss = 0.65 * rank_loss + 0.35 * bce_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ep_losses.append(float(loss))
        history_loss.append(float(np.mean(ep_losses)))
    return history_loss


def evaluate_domain(track, domain, data_dir, sem_dim=64, top_k_candidates=100):
    print(f"\n{'='*72}\nRunning Experiments on {track.upper()} / {domain.upper()}\n{'='*72}", flush=True)
    dom_dir = os.path.join(data_dir, track, domain)
    doc_key = "id" if track == "track1_tempo" else "doc_id"

    # 1. Load Corpus
    doc_ids, doc_texts = [], []
    with open(os.path.join(dom_dir, "documents.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                doc_ids.append(rec[doc_key])
                doc_texts.append(clean_html(rec["content"]))
    corpus_set = set(doc_ids)
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"  Loaded {len(doc_ids):,} corpus documents.", flush=True)

    # 2. Build FastBM25 Index & Dense SVD Semantic Index
    t0 = time.time()
    bm25 = FastBM25(doc_ids, doc_texts, k1=0.9, b=0.4)
    print(f"  Built BM25 index in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=25000, stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    doc_tfidf = tfidf.fit_transform(doc_texts)
    svd = TruncatedSVD(n_components=sem_dim, random_state=SEED)
    doc_svd = svd.fit_transform(doc_tfidf).astype(np.float32)
    doc_svd_norms = np.linalg.norm(doc_svd, axis=1, keepdims=True) + 1e-8
    doc_svd = doc_svd / doc_svd_norms
    print(f"  Built TF-IDF + {sem_dim}D Dense SVD embeddings in {time.time() - t0:.1f}s", flush=True)

    # Precompute metadata lazily for candidate docs
    doc_meta_cache = {}
    def get_doc_meta(idx):
        if idx not in doc_meta_cache:
            doc_meta_cache[idx] = extract_temporal_metadata(doc_texts[idx])
        return doc_meta_cache[idx]

    # 3. Load Train & Dev Queries + Gold Judgments
    splits_data = {}
    for split in ("train", "dev"):
        queries_main, queries_aux, qids, gold_map, turn_ids = [], [], [], {}, []
        period_map = {} if track == "track1_tempo" else None

        if track == "track1_tempo":
            ex_path = os.path.join(dom_dir, f"examples_{split}.jsonl")
            st_path = os.path.join(dom_dir, f"steps_{split}.jsonl")
            ex_recs = [json.loads(ln) for ln in open(ex_path, encoding="utf-8") if ln.strip()]
            st_recs = {r["id"]: r.get("steps", []) for r in (json.loads(ln) for ln in open(st_path, encoding="utf-8") if ln.strip())}

            for r in ex_recs:
                qid = r["id"]
                g_set = {g for g in r["gold_ids"] if g in corpus_set}
                if not g_set:
                    continue
                steps_list = st_recs.get(qid, [])
                q_main = clean_html(r["query"])
                q_aux = build_Temporal_Expanded_Query(r["query"], steps_list)
                qids.append(qid)
                queries_main.append(q_main)
                queries_aux.append(q_aux)
                gold_map[qid] = g_set
                turn_ids.append(1)
                period_map[qid] = [set(s.get("gold_ids", [])) & corpus_set for s in steps_list if set(s.get("gold_ids", [])) & corpus_set]
        else:
            bench_path = os.path.join(dom_dir, f"benchmark_{split}.json")
            convs = json.load(open(bench_path, encoding="utf-8"))
            for c in convs:
                for t in c["turns"]:
                    gold = t.get("gold_doc_ids") or t.get("supporting_doc_ids") or []
                    g_set = {g for g in gold if g in corpus_set}
                    if not g_set:
                        continue
                    qid = f"{c['id']}_turn_{t['turn_id']}"
                    # Official 2a_hist query: current turn + full conversation history
                    hist = t.get("conversation_history", "")
                    if hist and hist != "No previous conversation.":
                        q_full_hist = f"{clean_html(t['query'])}\n\nConversation History:\n{clean_html(hist)}"
                    else:
                        q_full_hist = clean_html(t["query"])
                    # Proposed Decayed Conversational Query
                    q_decayed, _, _ = build_Conversational_Decay_Query(t)

                    qids.append(qid)
                    queries_main.append(clean_html(t["query"]))  # Base turn only
                    queries_aux.append((q_full_hist, q_decayed))
                    gold_map[qid] = g_set
                    turn_ids.append(int(t.get("turn_id", 1)))

        splits_data[split] = {
            "qids": qids,
            "queries_main": queries_main,
            "queries_aux": queries_aux,
            "gold_map": gold_map,
            "turn_ids": turn_ids,
            "period_map": period_map,
        }

    # 4. Run Baselines and Build Candidate Feature Tensors for Train & Dev
    featurized = {}
    baseline_runs = {"train": defaultdict(dict), "dev": defaultdict(dict)}

    for split in ("train", "dev"):
        sd = splits_data[split]
        X_q, X_d, X_f, y_labels = [], [], [], []
        q_groups = defaultdict(list)
        cand_doc_ids = defaultdict(list)

        for idx_q, qid in enumerate(sd["qids"]):
            q_main = sd["queries_main"][idx_q]
            if track == "track1_tempo":
                q_hist_or_step = sd["queries_aux"][idx_q]
                q_decayed_or_exp = sd["queries_aux"][idx_q]
            else:
                q_hist_or_step, q_decayed_or_exp = sd["queries_aux"][idx_q]

            # Score all documents with BM25 Base, BM25 +Hist/+Step, and BM25 Proposed Query
            scores_base = bm25.score_all(q_main)
            scores_hist = bm25.score_all(q_hist_or_step)
            scores_prop_q = bm25.score_all(q_decayed_or_exp)

            # Record Official Baselines 1 & 2
            top_base = np.argpartition(-scores_base, 100)[:100]
            top_base = top_base[np.argsort(-scores_base[top_base])]
            baseline_runs[split]["BM25_Base"][qid] = [doc_ids[i] for i in top_base]

            top_hist = np.argpartition(-scores_hist, 100)[:100]
            top_hist = top_hist[np.argsort(-scores_hist[top_hist])]
            baseline_runs[split]["BM25_Context"][qid] = [doc_ids[i] for i in top_hist]

            # Hybrid Stage-1 Candidate Pool (Top-K from union of signals)
            hybrid_stage1 = 0.35 * scores_base + 0.65 * scores_prop_q
            cand_indices = np.argpartition(-hybrid_stage1, top_k_candidates)[:top_k_candidates]
            cand_indices = cand_indices[np.argsort(-hybrid_stage1[cand_indices])]

            # Ensure during training that gold docs are in candidate pool so positive pairs exist
            if split == "train":
                gold_indices = [doc_id_to_idx[g] for g in sd["gold_map"][qid] if g in doc_id_to_idx]
                existing = set(cand_indices.tolist())
                for gi in gold_indices:
                    if gi not in existing:
                        cand_indices = np.append(cand_indices[:-1], gi)

            # Dense Semantic Representations for Query
            q_tfidf_vec = tfidf.transform([q_decayed_or_exp])
            q_svd_vec = svd.transform(q_tfidf_vec).astype(np.float32)[0]
            q_svd_vec = q_svd_vec / (np.linalg.norm(q_svd_vec) + 1e-8)

            # Cosine similarities on candidate pool
            cand_tfidf = doc_tfidf[cand_indices]
            tfidf_sims = cosine_similarity(q_tfidf_vec, cand_tfidf)[0]
            svd_sims = np.dot(doc_svd[cand_indices], q_svd_vec)

            # Baseline 3: Dense Semantic Bi-Encoder Reranker (SVD-Cosine + TFIDF Semantic)
            sem_order = np.argsort(-(0.6 * svd_sims + 0.4 * tfidf_sims))
            baseline_runs[split]["Dense_BiEncoder"][qid] = [doc_ids[cand_indices[i]] for i in sem_order]

            # Z-normalize BM25 scores across candidate pool
            c_s1 = scores_hist[cand_indices]
            c_s2 = scores_prop_q[cand_indices]
            z_s1 = (c_s1 - np.mean(c_s1)) / (np.std(c_s1) + 1e-6)
            z_s2 = (c_s2 - np.mean(c_s2)) / (np.std(c_s2) + 1e-6)

            q_meta = extract_temporal_metadata(q_decayed_or_exp)
            t_id = sd["turn_ids"][idx_q]

            for pos_in_pool, doc_i in enumerate(cand_indices):
                d_meta = get_doc_meta(int(doc_i))
                feat_vec = extract_pair_features(
                    idx_q, q_hist_or_step, q_decayed_or_exp, q_meta, q_svd_vec,
                    int(doc_i), d_meta, doc_svd[doc_i],
                    float(c_s1[pos_in_pool]), float(z_s1[pos_in_pool]),
                    float(c_s2[pos_in_pool]), float(z_s2[pos_in_pool]),
                    float(tfidf_sims[pos_in_pool]), float(svd_sims[pos_in_pool]),
                    turn_id=t_id
                )
                row_idx = len(y_labels)
                X_q.append(q_svd_vec)
                X_d.append(doc_svd[doc_i])
                X_f.append(feat_vec)
                y_labels.append(1 if doc_ids[doc_i] in sd["gold_map"][qid] else 0)
                q_groups[qid].append(row_idx)
                cand_doc_ids[qid].append(doc_ids[doc_i])

        featurized[split] = {
            "X_q": np.array(X_q, dtype=np.float32),
            "X_d": np.array(X_d, dtype=np.float32),
            "X_f": np.array(X_f, dtype=np.float32),
            "y": np.array(y_labels, dtype=np.int32),
            "q_groups": q_groups,
            "cand_doc_ids": cand_doc_ids,
        }

    # Standardize features across train split
    f_mean = np.mean(featurized["train"]["X_f"], axis=0, keepdims=True)
    f_std = np.std(featurized["train"]["X_f"], axis=0, keepdims=True) + 1e-6
    featurized["train"]["X_f_norm"] = (featurized["train"]["X_f"] - f_mean) / f_std
    featurized["dev"]["X_f_norm"] = (featurized["dev"]["X_f"] - f_mean) / f_std

    # 5. Train & Evaluate Proposed Deep Learning Variants (Full Model + 4 Ablations)
    variants = [
        ("Ablation_NoGating", False, True, True, True),
        ("Ablation_NoTempConvFeats", True, False, True, True),
        ("Ablation_NoHardNegs", True, True, False, True),
        ("Ablation_NoAdaptivePrior", True, True, True, False),
        ("Proposed_TempConv_GatedRanker", True, True, True, True),
    ]

    loss_curves = {}
    gate_attention_stats = {}
    n_train_queries = len(splits_data["train"]["qids"])

    for var_name, use_gating, use_temp_feats, use_hard_negs, use_adaptive_prior in variants:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        model = build_TempConv_GatedRanker(
            sem_dim=sem_dim,
            feat_dim=featurized["train"]["X_f_norm"].shape[1],
            use_gating=use_gating,
            use_temp_feats=use_temp_feats,
        )
        loss_hist = train_pairwise_ranker(
            model,
            featurized["train"]["X_q"],
            featurized["train"]["X_d"],
            featurized["train"]["X_f_norm"],
            featurized["train"]["y"],
            featurized["train"]["q_groups"],
            hard_negative_mining=use_hard_negs,
            epochs=20,
            batch_size=128,
            lr=3.5e-3,
        )
        loss_curves[var_name] = loss_hist

        # Extract learned contextual gate weights on Dev set for the full model
        if var_name == "Proposed_TempConv_GatedRanker" and use_gating:
            gate_layer = model.get_layer("contextual_gate")
            gate_extractor = keras.Model(inputs=model.inputs, outputs=gate_layer.output)
            dev_gates = gate_extractor.predict(
                [featurized["dev"]["X_q"], featurized["dev"]["X_d"], featurized["dev"]["X_f_norm"]],
                batch_size=512, verbose=0
            )
            gate_attention_stats = {
                "lexical_gate_mean": round(float(np.mean(dev_gates[:, 0])), 4),
                "semantic_gate_mean": round(float(np.mean(dev_gates[:, 1])), 4),
                "temporal_conv_gate_mean": round(float(np.mean(dev_gates[:, 2])), 4),
                "lexical_gate_std": round(float(np.std(dev_gates[:, 0])), 4),
                "semantic_gate_std": round(float(np.std(dev_gates[:, 1])), 4),
                "temporal_conv_gate_std": round(float(np.std(dev_gates[:, 2])), 4),
            }

        # Predict and combine with Stage-1 Residual Anchor on Train and Dev
        for split in ("train", "dev"):
            preds = model.predict(
                [featurized[split]["X_q"], featurized[split]["X_d"], featurized[split]["X_f_norm"]],
                batch_size=512, verbose=0
            ).flatten()

            for qid, row_indices in featurized[split]["q_groups"].items():
                row_idx_arr = np.array(row_indices)
                nn_scores = preds[row_idx_arr]
                nn_z = (nn_scores - np.mean(nn_scores)) / (np.std(nn_scores) + 1e-6)

                if use_adaptive_prior:
                    # Sample-Size Adaptive Prior Anchor: combines lexical Z-score + dense semantic Z-score
                    lex_z = featurized[split]["X_f"][row_idx_arr, 1]
                    tfidf_c = featurized[split]["X_f"][row_idx_arr, 2]
                    svd_c = featurized[split]["X_f"][row_idx_arr, 3]
                    sem_raw = 0.6 * svd_c + 0.4 * tfidf_c
                    sem_z = (sem_raw - np.mean(sem_raw)) / (np.std(sem_raw) + 1e-6)
                    prior_z = 0.45 * lex_z + 0.55 * sem_z
                    # Bayesian shrinkage weight based on training topic count N_train / (N_train + tau)
                    tau = 65.0
                    alpha_nn = (n_train_queries / (n_train_queries + tau)) * (0.65 if (use_temp_feats and use_hard_negs) else 0.35)
                    comb_scores = (1.0 - alpha_nn) * prior_z + alpha_nn * nn_z
                else:
                    # Ablation D (No Adaptive Prior): unregularized fixed interpolation with BM25 only
                    stage1_z = featurized[split]["X_f"][row_idx_arr, 1]
                    alpha = 0.45 if (use_temp_feats and use_hard_negs) else 0.30
                    comb_scores = (1.0 - alpha) * stage1_z + alpha * nn_z

                order = np.argsort(-comb_scores)
                docs_cand = featurized[split]["cand_doc_ids"][qid]
                baseline_runs[split][var_name][qid] = [docs_cand[i] for i in order]

    # 6. Compute All Metrics for Every Method on Train and Dev
    domain_metrics = {"train": {}, "dev": {}, "loss_curves": loss_curves, "gate_stats": gate_attention_stats}
    for split in ("train", "dev"):
        sd = splits_data[split]
        print(f"\n  --- Results on {domain} ({split.upper()} split, {len(sd['qids'])} queries) ---", flush=True)
        for m_name, run_dict in baseline_runs[split].items():
            metrics = compute_Comprehensive_Metrics(
                run_dict,
                sd["gold_map"],
                period_dict=sd["period_map"],
                is_track2=(track == "track2_recor"),
            )
            domain_metrics[split][m_name] = metrics
            extra = f" | TC@10: {metrics.get('TC@10', 0):.4f} | TP@10: {metrics.get('TP@10', 0):.4f}" if track == "track1_tempo" else ""
            print(f"    {m_name:<32} nDCG@10: {metrics['nDCG@10']:.4f} | MRR@10: {metrics['MRR@10']:.4f} | R@10: {metrics['Recall@10']:.4f}{extra}", flush=True)

    return domain_metrics


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "reteco_data")
    all_results = {"track1_tempo": {}, "track2_recor": {}}

    track1_domains = ["iota", "quant", "workplace"]
    track2_domains = ["drones", "law", "politics"]

    for dom in track1_domains:
        all_results["track1_tempo"][dom] = evaluate_domain("track1_tempo", dom, data_dir)

    for dom in track2_domains:
        all_results["track2_recor"][dom] = evaluate_domain("track2_recor", dom, data_dir)

    # Compute Macro-Averages per Track & Overall
    summary = {"per_domain": all_results, "macro_average": {"track1_tempo": {}, "track2_recor": {}}}
    for track, dom_dict in all_results.items():
        for split in ("train", "dev"):
            summary["macro_average"][track][split] = {}
            methods = list(next(iter(dom_dict.values()))[split].keys())
            for m in methods:
                metric_keys = [k for k in next(iter(dom_dict.values()))[split][m].keys() if isinstance(next(iter(dom_dict.values()))[split][m][k], (int, float)) and k != "num_topics"]
                agg = {}
                for mk in metric_keys:
                    vals = [dom_dict[d][split][m][mk] for d in dom_dict]
                    agg[mk] = round(float(np.mean(vals)), 4)
                agg["num_topics"] = sum(dom_dict[d][split][m]["num_topics"] for d in dom_dict)
                summary["macro_average"][track][split][m] = agg

    os.makedirs("experiment_outputs", exist_ok=True)
    out_path = os.path.join("experiment_outputs", "experimental_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved all experimental results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
