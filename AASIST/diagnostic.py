# diagnostics.py
# Run: python diagnostics.py
# Edit the 4 paths below to match your setup

TRAIN_PROTOCOL = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5\ASVspoof5.train.tsv"
DEV_PROTOCOL   = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5\ASVspoof5.dev.track_1.tsv"
TRAIN_AUDIO    = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5\flac_T"
DEV_AUDIO      = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5\flac_D"
N_AUDIO_CHECK  = 200   # how many files to probe for load failures

import pandas as pd
import numpy as np
from pathlib import Path
import librosa

# ─────────────────────────────────────────────
# 1. PROTOCOL DELIMITER + FIRST ROWS
# ─────────────────────────────────────────────
print("=" * 60)
print("CHECK 1 — PROTOCOL PARSING")
print("=" * 60)

for name, path in [("TRAIN", TRAIN_PROTOCOL), ("DEV", DEV_PROTOCOL)]:
    print(f"\n[{name}] {path}")
    with open(path, 'r') as f:
        raw_lines = [f.readline() for _ in range(3)]
    print("  Raw first 3 lines:")
    for i, l in enumerate(raw_lines):
        print(f"    [{i}] {repr(l[:120])}")

    # Check what delimiter the file actually uses
    for sep, label in [(' ', 'SPACE'), ('\t', 'TAB'), (',', 'COMMA')]:
        try:
            df_test = pd.read_csv(path, sep=sep, header=None, nrows=3)
            print(f"  With sep={repr(sep)} ({label}): {df_test.shape[1]} columns")
        except Exception as e:
            print(f"  With sep={repr(sep)}: FAILED — {e}")

    # Load with the code's current sep=' '
    df = pd.read_csv(path, sep=' ', header=None)
    print(f"\n  Loaded with sep=SPACE: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  First 5 rows:")
    print(df.head())
    print(f"\n  Column 1 (file_ids) sample: {df[1].values[:5]}")

    # Label detection — mirrors your code exactly
    label_col = None
    for col in [8, 6, 7]:
        if col in df.columns:
            test_val = df[col].iloc[0]
            if test_val in ['spoof', 'bonafide', '-']:
                label_col = col
                break
    print(f"  Detected label column: {label_col}  (value='{df[label_col].iloc[0] if label_col is not None else 'NOT FOUND'}')")

    if label_col is not None:
        labels = df[label_col].apply(lambda x: 1 if x == 'spoof' else 0).values
        n_bonafide = (labels == 0).sum()
        n_spoof    = (labels == 1).sum()
        total      = len(labels)
        print(f"  Class counts → bonafide=0: {n_bonafide} ({100*n_bonafide/total:.1f}%)")
        print(f"                 spoof   =1: {n_spoof}    ({100*n_spoof/total:.1f}%)")
        print(f"\n  Sample (file_id, label) pairs:")
        for i in range(min(8, total)):
            print(f"    [{i}] file_id={df[1].values[i]!r:30s}  raw_label={df[label_col].values[i]!r}  → int={labels[i]}")
    else:
        print("  !! LABEL COLUMN NOT FOUND — protocol parsing is broken !!")
        print(f"  All column 0 unique values: {df[0].unique()[:10]}")
        if df.shape[1] > 6:
            for c in range(df.shape[1]):
                print(f"    col {c}: {df[c].unique()[:5]}")

# ─────────────────────────────────────────────
# 2. CLASS COUNTS CROSS-CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 2 — CLASS BALANCE SUMMARY")
print("=" * 60)

for name, path in [("TRAIN", TRAIN_PROTOCOL), ("DEV", DEV_PROTOCOL)]:
    df = pd.read_csv(path, sep=' ', header=None)
    label_col = next((c for c in [8, 6, 7]
                      if c in df.columns and df[c].iloc[0] in ['spoof','bonafide','-']), None)
    if label_col:
        labels = df[label_col].apply(lambda x: 1 if x == 'spoof' else 0).values
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n[{name}]  total={len(labels)}")
        for u, c in zip(unique, counts):
            tag = 'bonafide' if u == 0 else 'spoof'
            print(f"  label {u} ({tag}): {c:>8}  ({100*c/len(labels):.2f}%)")

        # Check for '-' entries being silently mapped to bonafide
        raw_vals = df[label_col].unique()
        print(f"  Unique raw label values: {raw_vals}")
        dash_count = (df[label_col] == '-').sum()
        if dash_count > 0:
            print(f"  !! WARNING: {dash_count} entries with '-' are being mapped to bonafide=0 !!")

# ─────────────────────────────────────────────
# 3. AUDIO LOAD FAILURE COUNT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 3 — AUDIO LOAD FAILURES")
print("=" * 60)

for name, proto_path, audio_dir in [
    ("TRAIN", TRAIN_PROTOCOL, TRAIN_AUDIO),
    ("DEV",   DEV_PROTOCOL,   DEV_AUDIO)
]:
    df = pd.read_csv(proto_path, sep=' ', header=None)
    file_ids = df[1].values
    n_check  = min(N_AUDIO_CHECK, len(file_ids))
    indices  = np.random.choice(len(file_ids), n_check, replace=False)

    failures   = []
    zero_waves = []

    print(f"\n[{name}] Checking {n_check} random files from {audio_dir}...")
    for i in indices:
        fid  = file_ids[i]
        fpath = Path(audio_dir) / f"{fid}.flac"
        if not fpath.exists():
            failures.append(f"MISSING: {fpath}")
            continue
        try:
            wav, _ = librosa.load(str(fpath), sr=16000, mono=True)
            if np.abs(wav).max() < 1e-6:
                zero_waves.append(str(fpath))
        except Exception as e:
            failures.append(f"LOAD_ERR {fpath}: {e}")

    print(f"  Files checked   : {n_check}")
    print(f"  Load failures   : {len(failures)}  ({100*len(failures)/n_check:.1f}%)")
    print(f"  Near-zero waves : {len(zero_waves)} ({100*len(zero_waves)/n_check:.1f}%)")
    if failures:
        print(f"  First 5 failures:")
        for f in failures[:5]:
            print(f"    {f}")
    if zero_waves:
        print(f"  First 5 zero-waveforms:")
        for f in zero_waves[:5]:
            print(f"    {f}")
    if not failures and not zero_waves:
        print("  ✓ All files loaded cleanly")

# ─────────────────────────────────────────────
# 4. CONFUSION MATRIX SIMULATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 4 — CONFUSION MATRIX (given val spoof acc ~0%)")
print("=" * 60)

df = pd.read_csv(DEV_PROTOCOL, sep=' ', header=None)
label_col = next((c for c in [8, 6, 7]
                  if c in df.columns and df[c].iloc[0] in ['spoof','bonafide','-']), None)
if label_col:
    labels = df[label_col].apply(lambda x: 1 if x == 'spoof' else 0).values
    # Simulate model predicting everything as bonafide (class 0)
    all_bonafide_preds = np.zeros_like(labels)
    tn = ((all_bonafide_preds == 0) & (labels == 0)).sum()  # predict bonafide, IS bonafide
    fp = ((all_bonafide_preds == 1) & (labels == 0)).sum()  # predict spoof,    IS bonafide
    fn = ((all_bonafide_preds == 0) & (labels == 1)).sum()  # predict bonafide, IS spoof
    tp = ((all_bonafide_preds == 1) & (labels == 1)).sum()  # predict spoof,    IS spoof

    print(f"\n  If model predicts ALL as bonafide:")
    print(f"  {'':20s}  Pred bonafide  Pred spoof")
    print(f"  {'True bonafide':20s}  TN={tn:>8}     FP={fp:>8}")
    print(f"  {'True spoof':20s}  FN={fn:>8}     TP={tp:>8}")
    print(f"  → Bonafide acc = {100*tn/(tn+fp+1e-8):.2f}%  (expected ~100%)")
    print(f"  → Spoof acc    = {100*tp/(fn+tp+1e-8):.2f}%  (expected ~0%)")
    print(f"  → Total acc    = {100*(tn+tp)/len(labels):.2f}%")
    print(f"\n  If your val Total Acc ~= {100*((labels==0).sum()/len(labels)):.1f}%,")
    print(f"  the model IS predicting everything as bonafide.")