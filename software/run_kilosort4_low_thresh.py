#!/usr/bin/env python
# coding: utf-8
"""
Run Kilosort4 on reconstructed recording with lowered detection thresholds
so that more spikes are detected, then compute F1 score against GT.
"""
import numpy as np
import os, sys, csv
from scipy.optimize import linear_sum_assignment
import kilosort
from kilosort import run_kilosort
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
KS_DIR = os.path.join(SAVE_DIR, 'kilosort4_lowthresh')
ORIG_DIR = os.path.join(BASE_DIR, 'transfer', 'kilosort4_full')
BIN_PATH = os.path.join(SAVE_DIR, 'recon_recording.bin')

os.makedirs(KS_DIR, exist_ok=True)

# ==================== Step 1: Run Kilosort4 with low thresholds ====================
print("=" * 70)
print("Running Kilosort4 on recon recording (low thresholds)")
print("=" * 70)

# Load original ops as template, modify thresholds
orig_ops_path = os.path.join(SAVE_DIR, 'kilosort4', 'ops.npy')
ops_data = np.load(orig_ops_path, allow_pickle=True)
ops = ops_data.item()

# Lower detection thresholds to detect more spikes
ops['Th_universal'] = 3.0      # default 9.0
ops['Th_learned'] = 3.0        # default 8.0
ops['Th_single_ch'] = 3.0      # default 6.0
ops['results_dir'] = KS_DIR
ops['data_file_path'] = [BIN_PATH]
ops['data_dir'] = SAVE_DIR
ops['filename'] = [BIN_PATH]
ops['torch_device'] = 'cpu'

print(f"\nThresholds: Th_universal={ops['Th_universal']}, "
      f"Th_learned={ops['Th_learned']}, Th_single_ch={ops['Th_single_ch']}")
print(f"Results dir: {KS_DIR}")
print(f"Input: {BIN_PATH}")

# Run Kilosort4
print("\nStarting Kilosort4 (this may take 1-2 hours)...")
try:
    run_kilosort(ops)
    print("\nKilosort4 completed!")
except Exception as e:
    print(f"\nError during Kilosort4: {e}")
    sys.exit(1)

# ==================== Step 2: Compute F1 Score ====================
print("\n" + "=" * 70)
print("Computing F1 Score")
print("=" * 70)

# Load recon results
st_recon = np.load(os.path.join(KS_DIR, 'spike_times.npy')).flatten()
sc_recon = np.load(os.path.join(KS_DIR, 'spike_clusters.npy')).flatten()

# Load original GT (good clusters only)
st_orig = np.load(os.path.join(ORIG_DIR, 'spike_times.npy')).flatten()
sc_orig = np.load(os.path.join(ORIG_DIR, 'spike_clusters.npy')).flatten()

with open(os.path.join(ORIG_DIR, 'cluster_group.tsv'), 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    good_clusters = set(int(row[0]) for row in reader if row[1] == 'good')

good_mask = np.isin(sc_orig, list(good_clusters))
st_orig_good = st_orig[good_mask]
sc_orig_good = sc_orig[good_mask]

# Remap GT to 0-based
unique_gt = np.unique(sc_orig_good)
gt_to_label = {c: i for i, c in enumerate(unique_gt)}
gt_labels = np.array([gt_to_label[c] for c in sc_orig_good])
n_clusters_gt = len(unique_gt)

# Match by time proximity
window = 10
matched_gt = []
matched_recon = []

print(f"Matching {len(st_recon)} recon spikes to {len(st_orig_good)} GT spikes...")
for i in range(len(st_recon)):
    t_recon = st_recon[i]
    c_recon = sc_recon[i]
    diff = np.abs(st_orig_good - t_recon)
    nearest = np.argmin(diff)
    if diff[nearest] <= window:
        matched_gt.append(gt_labels[nearest])
        matched_recon.append(c_recon)

print(f"  Matched: {len(matched_gt)} / {len(st_recon)} spikes")

if len(matched_gt) == 0:
    print("No matches! F1 = 0")
    sys.exit(0)

matched_gt = np.array(matched_gt)
matched_recon = np.array(matched_recon)

# Confusion matrix
unique_recon = np.unique(matched_recon)
recon_to_label = {c: i for i, c in enumerate(unique_recon)}
recon_labels = np.array([recon_to_label[c] for c in matched_recon])
n_clusters_recon = len(unique_recon)

conf_mat = np.zeros((n_clusters_gt, n_clusters_recon), dtype=int)
for g, r in zip(matched_gt, recon_labels):
    conf_mat[g, r] += 1

# Hungarian alignment
row_ind, col_ind = linear_sum_assignment(-conf_mat)

correct = int(conf_mat[row_ind, col_ind].sum())
total = int(conf_mat.sum())
acc = correct / total * 100

# Weighted F1
f1_w = 0.0
for r in range(n_clusters_gt):
    if r in row_ind:
        c = col_ind[list(row_ind).index(r)]
        tp = conf_mat[r, c]
    else:
        tp = 0
    fp_val = conf_mat[:, c].sum() - tp
    fn_val = conf_mat[r, :].sum() - tp
    prec = tp / (tp + fp_val) if (tp + fp_val) > 0 else 0
    rec = tp / (tp + fn_val) if (tp + fn_val) > 0 else 0
    f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1_w += f1_c * conf_mat[r, :].sum()
f1_w /= total

print(f"\n  Recon spikes:     {len(st_recon)}")
print(f"  Recon clusters:   {n_clusters_recon}")
print(f"  GT good clusters: {n_clusters_gt}")
print(f"  Accuracy:         {acc:.1f}% (chance: {100/n_clusters_gt:.1f}%)")
print(f"  Weighted F1:      {f1_w:.4f}")

# Save results
result_path = os.path.join(SAVE_DIR, 'f1_score_lowthresh.txt')
with open(result_path, 'w') as f:
    f.write(f"Kilosort4 on recon recording (low thresholds)\n")
    f.write(f"  Th_universal={ops['Th_universal']}, Th_learned={ops['Th_learned']}\n")
    f.write(f"  Recon spikes: {len(st_recon)}\n")
    f.write(f"  Matched to GT: {len(matched_gt)}\n")
    f.write(f"  Recon clusters: {n_clusters_recon}\n")
    f.write(f"  GT clusters: {n_clusters_gt}\n")
    f.write(f"  Accuracy: {acc:.1f}%\n")
    f.write(f"  Weighted F1: {f1_w:.4f}\n")
print(f"\nResults saved: {result_path}")
print("Done!")
