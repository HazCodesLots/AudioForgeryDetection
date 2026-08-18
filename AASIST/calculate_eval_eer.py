"""
calculate_eval_eer.py
Computes EER and minDCF from the scores written by AASIST/evaluate_eval_set.py.

Usage:
    python AASIST/calculate_eval_eer.py
    python AASIST/calculate_eval_eer.py --scores path/to/eval_scores_epoch13.txt
    python AASIST/calculate_eval_eer.py --scores ... --protocol path/to/ASVspoof5.eval.track_1.tsv

Score file format (one line per utterance):
    <file_id> <bonafide_probability>

Protocol file format (whitespace-separated, no header):
    col 0 = speaker_id
    col 1 = file_id
    ...
    col 8 = label  ('bonafide' or 'spoof')
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SCORES = (
    r"M:\Results\ASVspoof5\AASIST3Wav2Vec2\aasist3_wav2vec2\eval_scores_epoch13.txt"
)
DEFAULT_PROTOCOL = r"M:\Datasets\ASVspoof5\ASVspoof5.eval.track_1.tsv"


# ── Metric functions (identical to TrainAASIST3 in AASIST3.py) ─────────────
def compute_eer(labels, scores):
    """
    labels : 0=bonafide, 1=spoof (numpy int array)
    scores : spoof score (higher → more likely spoof)
    Returns (eer_percent, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    roc_interp = interp1d(fpr, tpr)
    eer_fpr = brentq(lambda x: 1.0 - x - roc_interp(x), 0.0, 1.0)
    eer = 100.0 * (1.0 - roc_interp(eer_fpr))

    idx = np.nanargmin(np.abs(fnr - fpr))
    threshold = thresholds[idx]
    return eer, threshold


def compute_min_dcf(labels, scores, c_miss=1, c_fa=10, pi_spf=0.05):
    """Official ASVspoof5 Track 1 minDCF (no normalization).

    DCF(τ) = β·P_miss(τ) + P_fa(τ)
    β = C_miss·(1 − π_spf) / (C_fa·π_spf) = 1·0.95 / (10·0.05) = 1.9

    With roc_curve(labels, scores, pos_label=1) [spoof=positive]:
      fpr = P(spoof decision | bonafide) = P_miss (false rejection of bonafide)
      fnr = P(bonafide decision | spoof) = P_fa   (false acceptance of spoof)
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    beta = c_miss * (1 - pi_spf) / (c_fa * pi_spf)  # = 1.9
    return float(np.min(beta * fpr + fnr))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",   default=DEFAULT_SCORES,
                        help="Path to the scores .txt produced by evaluate_eval_set.py")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL,
                        help="Path to ASVspoof5 eval protocol TSV (col 1=file_id, col 8=label)")
    args = parser.parse_args()

    # ── Load scores ──────────────────────────────────────────────────────────
    print(f"Loading scores : {args.scores}")
    scores_df = pd.read_csv(args.scores, sep=r"\s+", header=None,
                            names=["file_id", "bonafide_score"], engine="python")
    print(f"  {len(scores_df):,} utterances scored")
    scores_map = dict(zip(scores_df["file_id"], scores_df["bonafide_score"]))

    # ── Load protocol ─────────────────────────────────────────────────────────
    print(f"Loading protocol: {args.protocol}")
    proto_df = pd.read_csv(args.protocol, sep=r"\s+", header=None, engine="python")
    file_ids = proto_df[1].astype(str).values
    raw_labels = proto_df[8].astype(str).values
    labels = (raw_labels == "spoof").astype(np.int64)   # 0=bonafide, 1=spoof
    print(f"  {len(file_ids):,} utterances in protocol")
    print(f"  bonafide: {int((labels==0).sum()):,}  spoof: {int((labels==1).sum()):,}")

    # ── Align scores to protocol order ────────────────────────────────────────
    matched_scores = []
    matched_labels = []
    missing = 0
    for fid, lbl in zip(file_ids, labels):
        if fid in scores_map:
            # Convert bonafide probability → spoof score (matches pos_label=1 in roc_curve)
            matched_scores.append(1.0 - scores_map[fid])
            matched_labels.append(lbl)
        else:
            missing += 1

    if missing:
        print(f"  [WARNING] {missing:,} protocol entries had no score — excluded from metrics.")

    matched_scores = np.array(matched_scores, dtype=np.float32)
    matched_labels = np.array(matched_labels, dtype=np.int64)
    print(f"  {len(matched_scores):,} utterances used for metric computation")

    # ── Compute metrics ───────────────────────────────────────────────────────
    eer, threshold = compute_eer(matched_labels, matched_scores)
    min_dcf = compute_min_dcf(matched_labels, matched_scores)

    print()
    print("=" * 46)
    print("  EVAL SET RESULTS")
    print("=" * 46)
    print(f"  EER           : {eer:.4f} %")
    print(f"  minDCF        : {min_dcf:.4f}  (official Track 1: β=1.9, C_miss=1, C_fa=10, π_spf=0.05, no normalisation)")
    print(f"  EER threshold : {threshold:.6f}")
    print("=" * 46)

    # ── Write JSON results ────────────────────────────────────────────────────
    scores_stem = os.path.splitext(args.scores)[0]  # strip .txt
    json_path = scores_stem + "_results.json"
    results = {
        "EER": round(eer, 6),
        "minDCF": round(min_dcf, 6),
        "EER threshold": round(float(threshold), 6),
    }
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"\n  Results saved to: {json_path}")


if __name__ == "__main__":
    main()
