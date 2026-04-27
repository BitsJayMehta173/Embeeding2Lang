"""
sentence_validator.py
Validates full sentences (not just word pairs) using the H4 CP-CLE Guardrail.

Approach:
  1. Tokenise both English and Hindi sentences into content words.
  2. Look up each word in Glove (English) and H4 (Hindi).
  3. Compute a "Semantic Fingerprint": average of all pairwise cosine similarities
     within a sentence. High fingerprint = words are thematically coherent.
  4. Compute a "Cross-Lingual Agreement" score: how closely the Hindi fingerprint
     matches the expected English fingerprint (using sim_E from Glove as reference).
  5. Flag if the agreement is below a threshold -> SEMANTIC MISMATCH.

Also measures and compares latency vs mBERT for sentence-level scoring.
"""

import os, json, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
import gensim.downloader as api
from transformers import BertTokenizer, BertModel

MODELS_DIR = "models"

# ── load models ───────────────────────────────────────────────────────────────

print("Loading Glove (English reference)...")
glove = api.load("glove-wiki-gigaword-100")

print("Loading H4 (Hindi Guardrail)...")
h4 = KeyedVectors.load(os.path.join(MODELS_DIR, "H5_compressed.kv"))

print("Loading bilingual dictionary...")
dictionary = {}
rev_dict   = {}
with open("data/en-hi.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            en, hi = parts
            dictionary[en]  = hi
            rev_dict[hi]    = en


# ── helpers ───────────────────────────────────────────────────────────────────

def sentence_fingerprint_h4(sentence_hindi):
    """
    Compute average pairwise cosine similarity among Hindi words in a sentence
    using H4 embeddings. Returns (score, found_words, missing_words).
    """
    words   = [w.strip(".,!?;:\"'()") for w in sentence_hindi.lower().split()]
    found   = [w for w in words if w in h4]
    missing = [w for w in words if w not in h4]

    if len(found) < 2:
        return 0.0, found, missing

    sims = []
    for w1, w2 in itertools.combinations(found, 2):
        sims.append(h4.similarity(w1, w2))
    return float(np.mean(sims)), found, missing


def sentence_fingerprint_glove(sentence_english):
    """
    Compute average pairwise cosine similarity among English words using Glove.
    """
    words = [w.strip(".,!?;:\"'()") for w in sentence_english.lower().split()]
    found = [w for w in words if w in glove]

    if len(found) < 2:
        return 0.0, found, []

    sims = []
    for w1, w2 in itertools.combinations(found, 2):
        sims.append(glove.similarity(w1, w2))
    return float(np.mean(sims)), found, []


def cross_lingual_agreement(sentence_en, sentence_hi):
    """
    Cross-lingual agreement: for each English word with a known Hindi translation,
    compare sim_Glove(en_i, en_j) vs sim_H4(hi_i, hi_j).
    Agreement = 1 - mean_squared_error between the two similarity vectors.
    Returns agreement score in [0, 1]. Higher = better translation.
    """
    en_words = [w.strip(".,!?;:\"'()") for w in sentence_en.lower().split()]
    hi_words = [w.strip(".,!?;:\"'()") for w in sentence_hi.lower().split()]

    # Build aligned pairs: English word + its Hindi translation, both in vocab
    aligned_en = []
    aligned_hi = []
    for ew in en_words:
        hw = dictionary.get(ew)
        if hw and ew in glove and hw in h4:
            aligned_en.append(ew)
            aligned_hi.append(hw)
    # Also check if the actual Hindi words in the sentence have a reverse mapping
    for hw in hi_words:
        ew = rev_dict.get(hw)
        if ew and ew in glove and hw in h4 and ew not in aligned_en:
            aligned_en.append(ew)
            aligned_hi.append(hw)

    if len(aligned_en) < 2:
        return None, aligned_en, aligned_hi

    sim_e_list = []
    sim_h_list = []
    for (e1, h1_), (e2, h2_) in itertools.combinations(zip(aligned_en, aligned_hi), 2):
        sim_e_list.append(glove.similarity(e1, e2))
        sim_h_list.append(h4.similarity(h1_, h2_))

    mse  = float(np.mean([(se - sh)**2 for se, sh in zip(sim_e_list, sim_h_list)]))
    agreement = max(0.0, 1.0 - mse * 5)   # scale MSE to 0-1 range
    return agreement, aligned_en, aligned_hi


def validate_sentence(english, hindi_translation, threshold=0.40, label=None):
    """Full validation pipeline for a sentence pair."""
    start = time.perf_counter()

    fp_en, en_found,  _          = sentence_fingerprint_glove(english)
    fp_hi, hi_found, hi_missing  = sentence_fingerprint_h4(hindi_translation)
    agreement, aligned_en, aligned_hi = cross_lingual_agreement(english, hindi_translation)
    elapsed_ms = (time.perf_counter() - start) * 1000

    decision = "VALID" if (agreement is not None and agreement >= threshold) else "MISMATCH"

    return {
        "english":       english,
        "hindi":         hindi_translation,
        "label":         label,
        "en_fingerprint": round(fp_en, 4),
        "hi_fingerprint": round(fp_hi, 4),
        "agreement":     round(agreement, 4) if agreement is not None else None,
        "decision":      decision,
        "aligned_en":    aligned_en,
        "aligned_hi":    aligned_hi,
        "hi_missing":    hi_missing,
        "latency_ms":    round(elapsed_ms, 4),
    }


# ── mBERT sentence-level validator ───────────────────────────────────────────

def load_mbert():
    print("\nLoading H5 (Compressed cc.hi.300) model...")
    tok = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    mdl = BertModel.from_pretrained("bert-base-multilingual-cased")
    mdl.eval()
    return tok, mdl

def mbert_agreement(tok, mdl, english, hindi):
    start = time.perf_counter()
    with torch.no_grad():
        inp_en = tok(english, return_tensors="pt", truncation=True)
        inp_hi = tok(hindi,   return_tensors="pt", truncation=True)
        out_en = mdl(**inp_en).last_hidden_state[:, 0, :]
        out_hi = mdl(**inp_hi).last_hidden_state[:, 0, :]
        sim    = F.cosine_similarity(out_en, out_hi).item()
    elapsed_ms = (time.perf_counter() - start) * 1000
    decision   = "VALID" if sim >= 0.80 else "MISMATCH"
    return round(sim, 4), decision, round(elapsed_ms, 4)


# ── test sentences ────────────────────────────────────────────────────────────

TEST_CASES = [
    # (English, Hindi, is_correct, description)
    (
        "the student studies hard at the university",
        "छात्र विश्वविद्यालय में कठिन पढ़ाई करता है",
        True,
        "Correct: Student studying at university"
    ),
    (
        "the student studies hard at the university",
        "रसोई में खाना बना रहा है",
        False,
        "Wrong: Replaced with 'cooking in kitchen'"
    ),
    (
        "the doctor carefully treats the sick patient",
        "चिकित्सक बीमार रोगी का ध्यान से उपचार करता है",
        True,
        "Correct: Doctor treating patient"
    ),
    (
        "the doctor carefully treats the sick patient",
        "खिलाड़ी मैदान में तेज दौड़ता है",
        False,
        "Wrong: Replaced with 'player running on field'"
    ),
    (
        "the fast runner won the race with great speed",
        "तेज धावक ने बहुत तेजी से दौड़ जीती",
        True,
        "Correct: Fast runner winning race"
    ),
    (
        "the fast runner won the race with great speed",
        "धीरे चलने वाला सो गया",
        False,
        "Wrong: Opposite meaning - slow walker fell asleep"
    ),
    (
        "scientists discovered a new planet in the solar system",
        "वैज्ञानिकों ने सौर मंडल में एक नया ग्रह खोजा",
        True,
        "Correct: Scientists discovering planet"
    ),
    (
        "scientists discovered a new planet in the solar system",
        "बच्चे पार्क में खेल रहे हैं",
        False,
        "Wrong: Replaced with 'children playing in park'"
    ),
]


# ── run benchmark ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SENTENCE-LEVEL SEMANTIC GUARDRAIL BENCHMARK")
    print("="*70 + "\n")

    results_h4    = []
    results_mbert = []

    for (en, hi, is_correct, desc) in TEST_CASES:
        r = validate_sentence(en, hi, threshold=0.40, label="VALID" if is_correct else "INVALID")
        results_h4.append((desc, is_correct, r))
        print(f"[H4] {desc}")
        print(f"     EN fingerprint : {r['en_fingerprint']}")
        print(f"     HI fingerprint : {r['hi_fingerprint']}")
        print(f"     Agreement      : {r['agreement']}")
        print(f"     Decision       : {r['decision']}  (True: {'VALID' if is_correct else 'INVALID'})")
        print(f"     Latency        : {r['latency_ms']} ms")
        aligned_display = [(e, h) for e, h in zip(r['aligned_en'], r['aligned_hi'])]
        print(f"     Aligned pairs  : {[(e, h.encode('utf-8')) for e, h in aligned_display]}")
        print()

    # mBERT
    print("\nRunning mBERT comparison...")
    tok, mdl = load_mbert()
    for (en, hi, is_correct, desc) in TEST_CASES:
        sim, dec, lat = mbert_agreement(tok, mdl, en, hi)
        results_mbert.append((desc, is_correct, sim, dec, lat))
        print(f"[mBERT] {desc}")
        print(f"     Agreement : {sim}  | Decision: {dec} | Latency: {lat} ms\n")

    # Accuracy
    h4_correct    = sum(1 for (_, gt, r) in results_h4    if (r["decision"]=="VALID") == gt)
    mbert_correct = sum(1 for (_, gt, _, dec, _) in results_mbert if (dec=="VALID") == gt)

    print(f"\nH4    Sentence Accuracy: {h4_correct}/{len(TEST_CASES)}")
    print(f"mBERT Sentence Accuracy: {mbert_correct}/{len(TEST_CASES)}")

    # Save results for PDF
    out = {
        "h4":    [(d, gt, r) for (d, gt, r) in results_h4],
        "mbert": [(d, gt, s, dec, lat) for (d, gt, s, dec, lat) in results_mbert],
        "h4_acc": h4_correct,
        "mbert_acc": mbert_correct,
        "total": len(TEST_CASES),
    }
    import pickle
    with open("results/sentence_benchmark_results.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\nResults saved to results/sentence_benchmark_results.pkl")
