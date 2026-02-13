"""
Diagnose WaveFake split integrity using the current WaveFakeDatasetFixed API.
Checks for overlap between train and test sets across all vocoders and real data.
"""

import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
splits_json = os.path.join(script_dir, 'wavefake_splits.json')

print(f"Loading splits from: {splits_json}")
with open(splits_json, 'r') as f:
    splits = json.load(f)

print(f"Metadata: {splits.get('metadata', 'NO METADATA')}\n")

# ============================================================
# 1. Per-category overlap check (train vs test within each category)
# ============================================================
print("=" * 60)
print("1. INTRA-CATEGORY OVERLAP CHECK (train ∩ test per category)")
print("=" * 60)

total_overlap = 0

# Real data
real_train = set(splits['real']['train'])
real_test = set(splits['real']['test'])
overlap = real_train & real_test
total_overlap += len(overlap)
total = len(real_train) + len(real_test)
ratio = len(real_train) / total
print(f"  real: train={len(real_train)}, test={len(real_test)}, "
      f"overlap={len(overlap)}, ratio={ratio:.3f}")

# Each vocoder
for voc_name, voc_data in splits['vocoders'].items():
    train_set = set(voc_data['train'])
    test_set = set(voc_data['test'])
    overlap = train_set & test_set
    total_overlap += len(overlap)
    total = len(train_set) + len(test_set)
    ratio = len(train_set) / total
    print(f"  {voc_name}: train={len(train_set)}, test={len(test_set)}, "
          f"overlap={len(overlap)}, ratio={ratio:.3f}")

print(f"\n  Total overlapping files: {total_overlap}")

# ============================================================
# 2. Cross-category check (real vs fake files)
# ============================================================
print("\n" + "=" * 60)
print("2. CROSS-CATEGORY CHECK (real files in fake sets?)")
print("=" * 60)

all_real = real_train | real_test
cross_issues = 0
for voc_name, voc_data in splits['vocoders'].items():
    all_fake = set(voc_data['train']) | set(voc_data['test'])
    cross = all_real & all_fake
    if cross:
        print(f"  🚨 {voc_name}: {len(cross)} files appear in both real and fake!")
        cross_issues += len(cross)

if cross_issues == 0:
    print("  ✅ No cross-category leakage detected")

# ============================================================
# 3. Cross-vocoder check (file overlap between vocoders)
# ============================================================
print("\n" + "=" * 60)
print("3. CROSS-VOCODER CHECK (file overlap between vocoders?)")
print("=" * 60)

voc_names = list(splits['vocoders'].keys())
cross_voc_issues = 0
for i, v1 in enumerate(voc_names):
    all_v1 = set(splits['vocoders'][v1]['train']) | set(splits['vocoders'][v1]['test'])
    for j, v2 in enumerate(voc_names):
        if i < j:
            all_v2 = set(splits['vocoders'][v2]['train']) | set(splits['vocoders'][v2]['test'])
            overlap = all_v1 & all_v2
            if overlap:
                print(f"  🚨 {v1} ∩ {v2}: {len(overlap)} shared files!")
                cross_voc_issues += len(overlap)

if cross_voc_issues == 0:
    print("  ✅ No cross-vocoder overlap detected")

# ============================================================
# 4. PHYSICAL INTEGRITY CHECK (Do files exist on disk?)
# ============================================================
print("\n" + "=" * 60)
print("4. PHYSICAL INTEGRITY CHECK (Do files exist on disk?)")
print("=" * 60)

import random

def check_existence(file_list, category_name, num_samples=10):
    if not file_list:
        return 0
    
    samples = random.sample(file_list, min(len(file_list), num_samples))
    missing = 0
    for f in samples:
        if not os.path.exists(f):
            missing += 1
    
    if missing > 0:
        print(f"  🚨 {category_name}: {missing}/{len(samples)} samples CHECKED MISSING!")
    else:
        print(f"  ✅ {category_name}: All {len(samples)} checked samples exist")
    return missing

total_missing = 0
total_missing += check_existence(splits['real']['train'], "real_train")
total_missing += check_existence(splits['real']['test'], "real_test")

for voc_name, voc_data in splits['vocoders'].items():
    total_missing += check_existence(voc_data['train'], f"{voc_name}_train")
    total_missing += check_existence(voc_data['test'], f"{voc_name}_test")

if total_missing == 0:
    print("\n  ✅ Physical integrity looks good!")
else:
    print(f"\n  🚨 FOUND MISSING FILES! Total missing in samples: {total_missing}")

# ============================================================
# 5. DataLoader integration test
# ============================================================
print("\n" + "=" * 60)
print("5. DATALOADER INTEGRATION TEST")
print("=" * 60)

from WaveFakeLoader import WaveFakeDatasetFixed

try:
    train_ds = WaveFakeDatasetFixed(
        splits_json=splits_json, split_type='train',
        vocoders_to_include=None, include_real=True
    )
    test_ds = WaveFakeDatasetFixed(
        splits_json=splits_json, split_type='test',
        vocoders_to_include=None, include_real=True
    )

    # Check first item to trigger file load
    _ = train_ds[0]
    
    train_paths = set(s['path'] for s in train_ds.samples)
    test_paths = set(s['path'] for s in test_ds.samples)
    loader_overlap = train_paths & test_paths

    print(f"  Train loader samples: {len(train_ds)}")
    print(f"  Test loader samples:  {len(test_ds)}")
    print(f"  File-level overlap:   {len(loader_overlap)}")

    total = len(train_ds) + len(test_ds)
    print(f"  Train ratio: {len(train_ds)/total:.3f}")
    
except Exception as e:
    print(f"  🚨 DataLoader Test Failed: {e}")
    loader_overlap = [1] # Force failure status

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 60)
if total_overlap == 0 and cross_issues == 0 and cross_voc_issues == 0 and len(loader_overlap) == 0 and total_missing == 0:
    print("✅ ALL CHECKS PASSED — No data leakage detected and files exist")
else:
    print("🚨 ISSUES FOUND — See above for details")
print("=" * 60)
