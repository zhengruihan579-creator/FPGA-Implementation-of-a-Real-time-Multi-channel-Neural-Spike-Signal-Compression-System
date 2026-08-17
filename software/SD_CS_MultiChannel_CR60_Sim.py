#!/usr/bin/env python
# coding: utf-8
"""
Multi-channel SD+CS Simulation — CR=60%
- Neuropixel dataset (374 channels, 30kHz)
- Spike extraction using Kilosort4 results
- MDC matrix compression (CR=0.60)
- IRLS reconstruction
- Kilosort labels as GT, PCA+K-Means for comparison
- Save all results to disk
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
CR_TARGET = 0.60
WINDOW_SIZE = 50
SEGMENT_LENGTH = 2 * WINDOW_SIZE
P_SIGMA = 0.4  # sigma parameter
SEED = 42
N_GROUPS = 14      # number of test groups for F1 evaluation
NOISE_THRESH = 1.0 # signal norm threshold for meaningful channels

# ====== Plot-Only Mode ======
PLOT_ONLY = False  # Set True to skip re-processing and just plot from saved results
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(BASE_DIR, 'transfer')
RESULTS_DIR = os.path.join(TRANSFER_DIR, 'kilosort4_full')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# Use CR-specific save files
CR_SUFFIX = f'_cr{int(CR_TARGET*100)}'
BEST_CH_FILE = f'best_channel_data{CR_SUFFIX}.npz'
SEGMENTS_FILE = f'neuropixel_raw_segments{CR_SUFFIX}.npz'
RECON_FILE = f'neuropixel_recon_segments{CR_SUFFIX}.npz'
RESULTS_FILE = f'neuropixel_results{CR_SUFFIX}.npz'
BAR_CHART_FILE = f'SD_CS_MultiChannel_BestChannel_BarChart{CR_SUFFIX}.png'
WAVEFORM_FILE = f'SD_CS_MultiChannel_Waveforms{CR_SUFFIX}.png'

# Font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Font sizes for plots
FS_TITLE = 11
FS_SUBTITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 8
FS_ANNOT = 9


# ==================== MDC functions (MultiChannel version) ====================
def Distance(input_signal, CR, p):
    """Compute sigma for MDC matrix generation"""
    _max = np.max(input_signal)
    _min = np.min(input_signal)
    N = len(input_signal)
    M = N * (1 - CR)
    ave_width = p * (_max - _min) / M
    sigma = (_max - _min - (M - 1) * ave_width) / M
    return sigma


def MDC_UMDC_Gen(input_signal, sigma):
    """Generate MDC matrix from input signal and sigma"""
    N = len(input_signal)
    MDC = np.empty([1, N])
    i = 0
    cluster = np.zeros(N)
    others = np.zeros(N)
    while i < N:
        if i == 0:
            core_data = input_signal[0]
            for j in range(N):
                dis = sigma - np.abs(core_data - input_signal[j])
                if dis >= 0:
                    cluster[j] = input_signal[j]
            others = input_signal - cluster
            MDC = cluster.reshape(1, N)
        else:
            core_data = core_data_next
            for j in core_data_ind:
                dis = sigma - np.abs(core_data - input_signal[j])
                if dis >= 0:
                    cluster[j] = input_signal[j]
            others = input_signal_new - cluster
            MDC = np.append(MDC, cluster)
        if not others.any():
            break
        core_data_ind = np.where(others != 0)[0]
        core_data_next = input_signal[core_data_ind[0]]
        input_signal_new = others
        i = i + 1
        cluster = np.zeros(N)
        others = np.zeros(N)
    cluster_num = MDC.size // N
    MDC = MDC.reshape(cluster_num, N)
    return MDC


def CS_IRLS(y, T_Mat, m):
    """IRLS reconstruction algorithm"""
    hat_x_tp = T_Mat.T.dot(y)
    epsilong = 1
    p = 1
    times = 1
    max_iter = max(5, int(len(y) / 4))
    while (epsilong > 10e-9) and (times < max_iter):
        AA = hat_x_tp * hat_x_tp + epsilong
        a = np.shape(AA)[0]
        b = np.shape(AA)[1]
        BB = np.ones((a, b)) * (p / 2 - 1)
        weight = AA ** BB
        n_y = T_Mat.shape[1]
        CC = np.ones((n_y, 1)) / weight
        Q_Mat = np.zeros((n_y, n_y))
        CC_flat = CC.flatten()
        for i in range(n_y):
            Q_Mat[i, i] = CC_flat[i] if i < len(CC_flat) else CC_flat[-1]
        A = Q_Mat.dot(T_Mat.T)
        B = T_Mat.dot(A)
        C = np.linalg.pinv(B).dot(y)
        hat_x = A.dot(C)
        if norm2(hat_x - hat_x_tp) < math.sqrt(epsilong) / 100:
            epsilong = epsilong / 10
        hat_x_tp = hat_x
        times = times + 1
    return hat_x


def norm2(x):
    return np.linalg.norm(x, 2)


def Signal_to_noise_Distortion_Ratio(re_signal, input_signal):
    """Compute SNDR in dB"""
    error = input_signal - re_signal
    signal_norm = norm2(input_signal)
    error_norm = norm2(error)
    sndr = 20 * math.log(signal_norm / error_norm, 10)
    return sndr


# ==================== Data Loading ====================
def load_neuropixel_data():
    print("Loading Neuropixel data...")
    raw_data = np.load(os.path.join(TRANSFER_DIR, 'dataSample_BPF_300_5000.npy'))
    print(f"  Raw data: {raw_data.shape}")

    # Load original results file (shared data: spike_times, pc_feature_ind, etc.)
    orig_results_file = 'neuropixel_results.npz'
    old_results = np.load(os.path.join(SAVE_DIR, orig_results_file), allow_pickle=True)
    spike_times = old_results['spike_times']
    spike_clusters = old_results['spike_clusters']
    pc_feature_ind = old_results['pc_feature_ind']
    gt_labels = old_results['gt_labels']
    cluster_labels = old_results['kilosort_labels']

    # Build channel map
    n_channels_total = raw_data.shape[0]
    channel_map = np.arange(n_channels_total)
    print(f"  {len(spike_times)} spikes, {len(np.unique(spike_clusters))} clusters")
    print(f"  PC feature indices: {pc_feature_ind.shape}")
    return raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels


# ==================== Processing ====================
def process_multichan_dataset(raw_data, spike_times, spike_clusters,
                               pc_feature_ind, cluster_labels):
    """
    Best-channel approach: find the best channel among 16 associated channels,
    compress and reconstruct only that channel.
    """
    print("\nProcessing spikes (best-channel only)...")
    W = WINDOW_SIZE
    seg_length = SEGMENT_LENGTH

    n_spikes = len(spike_times)
    st_f = spike_times.astype(int)
    sc_f = spike_clusters.astype(int)

    # Build cluster→label mapping
    unique_clusters = np.unique(spike_clusters)
    cluster_to_label = {}
    for cl in unique_clusters:
        mask = spike_clusters == cl
        if np.any(mask):
            cluster_to_label[cl] = int(cluster_labels[mask][0])

    orig_best_list = []
    recon_best_list = []
    best_ch_phys_list = []
    best_sndr_list = []
    best_cr_list = []
    spike_times_list = []
    spike_clusters_list = []
    gt_labels_list = []

    BATCH = 2000

    for i in range(n_spikes):
        t = int(st_f[i])
        cid = int(sc_f[i])

        if cid < pc_feature_ind.shape[0]:
            channels = pc_feature_ind[cid]
        else:
            channels = np.arange(min(16, raw_data.shape[0]))
        n_chan = len(channels)

        start_t = t - W
        end_t = t + W
        if start_t < 0 or end_t >= raw_data.shape[1]:
            continue
        raw_seg = raw_data[channels, start_t:end_t].astype(np.float64)

        signal_norms = np.array([np.linalg.norm(raw_seg[ch, :]) for ch in range(n_chan)])
        best_local = int(np.argmax(signal_norms))
        best_phys = int(channels[best_local])
        best_signal = raw_seg[best_local, :]

        _sigma = Distance(best_signal, CR_TARGET, P_SIGMA)
        _MDC = MDC_UMDC_Gen(best_signal, _sigma)
        com_signal = np.dot(_MDC, best_signal)
        m = len(com_signal)
        actual_cr_val = 1 - m / seg_length

        re_signal = CS_IRLS(com_signal.reshape(m, 1), _MDC, seg_length).flatten()

        sndr_val = Signal_to_noise_Distortion_Ratio(
            re_signal.reshape(-1, 1), best_signal.reshape(-1, 1))

        orig_best_list.append(best_signal.astype(np.float32))
        recon_best_list.append(re_signal.astype(np.float32))
        best_ch_phys_list.append(best_phys)
        best_sndr_list.append(sndr_val)
        best_cr_list.append(actual_cr_val)
        spike_times_list.append(t)
        spike_clusters_list.append(cid)
        gt_labels_list.append(cluster_to_label[cid])

        if (i + 1) % BATCH == 0:
            finite = [s for s in best_sndr_list if np.isfinite(s)]
            ms = f"{np.mean(finite):.1f}" if finite else "N/A"
            print(f"    ... {i+1}/{n_spikes} spikes (SNDR={ms}dB)", flush=True)

    orig_best = np.array(orig_best_list, dtype=np.float32)
    recon_best = np.array(recon_best_list, dtype=np.float32)
    best_sndr = np.array(best_sndr_list)
    best_cr = np.array(best_cr_list)
    gt_labels = np.array(gt_labels_list)

    sndr_finite = best_sndr[np.isfinite(best_sndr)]
    n_inf = np.sum(~np.isfinite(best_sndr))

    print(f"\n  Best-channel processing complete!")
    print(f"  Total spikes: {len(orig_best)}")
    if len(sndr_finite) > 0:
        print(f"  Mean SNDR (finite): {np.mean(sndr_finite):.2f} dB")
        print(f"  SNDR range: [{sndr_finite.min():.1f}, {sndr_finite.max():.1f}] dB")
    print(f"  Inf SNDR: {n_inf}")
    print(f"  Mean CR: {np.mean(best_cr):.4f}")

    return {
        'orig_best': orig_best,
        'recon_best': recon_best,
        'best_ch_phys': np.array(best_ch_phys_list, dtype=np.uint16),
        'best_sndr': best_sndr,
        'best_cr': best_cr,
        'spike_times': np.array(spike_times_list),
        'spike_clusters': np.array(spike_clusters_list),
        'gt_labels': gt_labels,
    }


def save_best_channel_results(result):
    save_path = os.path.join(SAVE_DIR, BEST_CH_FILE)
    np.savez_compressed(save_path,
                        orig_best=result['orig_best'],
                        recon_best=result['recon_best'],
                        best_ch_phys=result['best_ch_phys'],
                        best_sndr=result['best_sndr'],
                        best_cr=result['best_cr'],
                        spike_times=result['spike_times'],
                        spike_clusters=result['spike_clusters'],
                        gt_labels=result['gt_labels'])
    print(f"  Saved: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")


def load_best_channel_results():
    save_path = os.path.join(SAVE_DIR, BEST_CH_FILE)
    if not os.path.exists(save_path):
        return None
    data = np.load(save_path, allow_pickle=True)
    print(f"  Loaded {BEST_CH_FILE}", flush=True)
    return {
        'orig_best': data['orig_best'],
        'recon_best': data['recon_best'],
        'best_ch_phys': data['best_ch_phys'],
        'best_sndr': data['best_sndr'],
        'best_cr': data['best_cr'],
        'spike_times': data['spike_times'],
        'spike_clusters': data['spike_clusters'],
        'gt_labels': data['gt_labels'],
    }


def get_or_process_best_channel(raw_data, spike_times, spike_clusters,
                                 pc_feature_ind, channel_map, cluster_labels):
    if PLOT_ONLY:
        result = load_best_channel_results()
        if result is not None:
            print("  Loaded saved best-channel results")
            return result
        print("  No saved results found. Set PLOT_ONLY=False to process.")
        import sys
        sys.exit(1)
    result = process_multichan_dataset(
        raw_data, spike_times, spike_clusters, pc_feature_ind, cluster_labels)
    save_best_channel_results(result)
    return result


# ==================== Plotting Functions ====================

def get_multichan_context(raw_data, spike_time, pc_feature_ind, cluster_id,
                          n_channels_show=8, half_window=60):
    channels_all = pc_feature_ind[cluster_id] if cluster_id < pc_feature_ind.shape[0] \
        else np.arange(min(16, raw_data.shape[0]))
    channels_show = channels_all[:n_channels_show]
    start = max(0, spike_time - half_window)
    end = min(raw_data.shape[1], spike_time + half_window)
    time_axis = np.arange(start - spike_time, end - spike_time)
    channel_traces = raw_data[channels_show, start:end]
    return time_axis, channel_traces


def compute_per_channel_sndr_cr(results):
    if 'best_sndr' not in results:
        print("  No best_sndr found, recomputing...", flush=True)
        orig = results['orig_best']
        recon = results['recon_best']
        sndr_list = []
        for i in range(len(orig)):
            o = orig[i].astype(np.float64)
            r = recon[i].astype(np.float64)
            sn = np.linalg.norm(o)
            en = np.linalg.norm(o - r)
            s = 20 * np.log10(sn / (en + 1e-10)) if en > 1e-12 else np.inf
            sndr_list.append(s)
        results['best_sndr'] = np.array(sndr_list)
    if 'best_cr' not in results:
        results['best_cr'] = np.full(len(results['orig_best']), CR_TARGET)


def evaluate_f1_score(results):
    """
    Best-channel F1 evaluation: 14 groups.
    Uses original KMeans labels as pseudo-GT for recon evaluation.
    """
    print("\n" + "=" * 70)
    print(f"Best-Channel Evaluation (CR={CR_TARGET:.0%}): {N_GROUPS} Groups")
    print("=" * 70)

    orig_best = results['orig_best']
    recon_best = results['recon_best']
    gt_labels = results['gt_labels']
    best_sndr = results['best_sndr']
    best_cr = results.get('best_cr', np.full(len(orig_best), CR_TARGET))

    n_total = len(orig_best)
    n_clusters = len(np.unique(gt_labels))
    group_size = n_total // N_GROUPS

    assert n_total % N_GROUPS == 0, f"{n_total} not divisible by {N_GROUPS}"
    print(f"  {n_total} spikes, {n_clusters} GT clusters, {N_GROUPS} groups x {group_size}")

    FEAT_S = 20
    FEAT_E = 81

    group_results = []

    for g in range(N_GROUPS):
        s = g * group_size
        e = (g + 1) * group_size
        idx = np.arange(s, e)
        n_use = len(idx)

        print(f"\n  --- Group {g+1}/{N_GROUPS}: {s}-{e-1} ({n_use}) ---", flush=True)
        t0 = time.time()

        Xo = orig_best[idx, FEAT_S:FEAT_E]
        Xr = recon_best[idx, FEAT_S:FEAT_E]
        gt = gt_labels[idx]

        scaler = StandardScaler()
        Xo_s = scaler.fit_transform(Xo)
        Xr_s = scaler.transform(Xr)

        # KMeans on original (fit and predict)
        km_o = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3,
                               batch_size=min(1000, n_use), verbose=0)
        lo = km_o.fit_predict(Xo_s)

        # Predict on reconstructed using original KMeans cluster centers
        lr = km_o.predict(Xr_s)

        def f1_vs_gt(pred_labels, true_labels):
            conf = np.zeros((n_clusters, n_clusters), dtype=int)
            for t, p in zip(true_labels, pred_labels):
                conf[t, p] += 1
            ri, ci = linear_sum_assignment(-conf)
            correct = int(conf[ri, ci].sum())
            total = int(conf.sum())
            acc = correct / max(total, 1) * 100
            f1_w = 0.0
            for r in range(n_clusters):
                c = ci[list(ri).index(r)]
                tp = conf[r, c]
                fp = conf[:, c].sum() - tp
                fn = conf[r, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                f1_w += f1_c * conf[r, :].sum()
            f1_w /= max(total, 1)
            return f1_w, acc

        f1_orig, acc_orig = f1_vs_gt(lo, gt)
        f1_recon, acc_recon = f1_vs_gt(lr, gt)
        f1_recon_vs_orig, acc_recon_vs_orig = f1_vs_gt(lr, lo)

        g_sndr = best_sndr[idx]
        fs = g_sndr[np.isfinite(g_sndr)]
        mean_sndr = float(np.mean(fs)) if len(fs) > 0 else np.nan
        mean_cr = float(np.mean(best_cr[idx]))

        group_results.append((g + 1, n_use, f1_orig, acc_orig, f1_recon, acc_recon,
                              f1_recon_vs_orig, acc_recon_vs_orig, mean_sndr, mean_cr))
        print(f"    F1(orig->GT)={f1_orig:.4f}  F1(recon->origKM)={f1_recon_vs_orig:.4f}  "
              f"SNDR={mean_sndr:.1f}dB  ({time.time()-t0:.1f}s)", flush=True)

    # Summary table
    f1o = [r[2] for r in group_results]
    f1r = [r[4] for r in group_results]
    f1rv = [r[6] for r in group_results]
    sndr_vals = [r[8] for r in group_results]
    cr_vals = [r[9] for r in group_results]
    print(f"\n{'='*105}")
    print(f"{'Grp':>4s}  {'Samples':>8s}  {'F1(orig->GT)':>13s}  {'F1(recon->GT)':>14s}  "
          f"{'F1(recon->origKM)':>18s}  {'SNDR':>8s}  {'CR':>8s}")
    print(f"{'-'*105}")
    for r in group_results:
        print(f"{r[0]:>4d}  {r[1]:>8d}  {r[2]:>13.4f}  {r[4]:>14.4f}  "
              f"{r[6]:>17.4f}  {r[8]:>7.1f}dB  {r[9]:>8.4f}")
    print(f"{'-'*105}")
    print(f"{'Mean':>4s}  {'':>8s}  {np.mean(f1o):>13.4f}  {np.mean(f1r):>14.4f}  "
          f"{np.mean(f1rv):>17.4f}  {np.mean(sndr_vals):>7.1f}dB  {np.mean(cr_vals):>8.4f}")
    print(f"{'Std':>4s}  {'':>8s}  {np.std(f1o):>13.4f}  {np.std(f1r):>14.4f}  "
          f"{np.std(f1rv):>17.4f}  {np.std(sndr_vals):>7.1f}dB  {np.std(cr_vals):>8.4f}")
    print(f"{'='*105}")

    results['_group_results'] = [(r[0], r[1], r[6], r[7], r[8], r[9]) for r in group_results]
    results['_group_results_orig'] = [(r[0], r[1], r[2], r[3], r[8], r[9]) for r in group_results]
    results['_group_results_recon_vs_gt'] = [(r[0], r[1], r[4], r[5], r[8], r[9]) for r in group_results]
    return group_results


# ==================== Waveform Comparison ====================
def plot_waveform_comparison(results):
    orig_best = results['orig_best']
    recon_best = results['recon_best']
    best_sndr = results['best_sndr']
    best_ch_phys = results['best_ch_phys']
    best_cr = results.get('best_cr', np.full(len(orig_best), CR_TARGET))
    spike_times = results['spike_times']
    spike_clusters = results['spike_clusters']

    orig_std = np.std(orig_best, axis=1)
    signal_mask = orig_std >= 10.0
    idx_signal = np.where(signal_mask)[0]
    sndr_signal = best_sndr[signal_mask]

    global_mean = np.mean(best_sndr[np.isfinite(best_sndr)])
    print(f"  Global SNDR mean: {global_mean:.1f} dB ({len(best_sndr)} spikes, "
          f"signal-filtered: {len(idx_signal)} spikes)", flush=True)

    if len(idx_signal) == 0:
        print("  No valid spikes for waveform comparison.")
        return

    def pick_centered_spike(target_sndr, tol=0.5):
        candidates = np.where(np.abs(sndr_signal - target_sndr) <= tol)[0]
        if len(candidates) == 0:
            candidates = np.where(np.abs(sndr_signal - target_sndr) <= tol * 5)[0]
        if len(candidates) == 0:
            return idx_signal[np.argmin(np.abs(sndr_signal - target_sndr))]
        best_centered = candidates[0]
        best_dist = 999
        for ci in candidates:
            si = idx_signal[ci]
            seg = orig_best[si]
            peak_pos = np.argmax(np.abs(seg))
            dist = abs(peak_pos - 50)
            if dist < best_dist:
                best_dist = dist
                best_centered = ci
        return idx_signal[best_centered]

    sndr_max_val = np.max(sndr_signal)
    idx_max = pick_centered_spike(sndr_max_val)
    idx_min = pick_centered_spike(np.min(sndr_signal))
    idx_mean = pick_centered_spike(global_mean)

    spike_idx = [idx_max, idx_min, idx_mean]
    row_labels = ['(a) Max SNDR', '(b) Min SNDR', '(c) Mean SNDR']

    print("  Loading raw data for multi-channel context...", flush=True)
    raw_data = np.load(os.path.join(TRANSFER_DIR, 'dataSample_BPF_300_5000.npy'))
    old_data = np.load(os.path.join(SAVE_DIR, 'neuropixel_results.npz'), allow_pickle=True)
    pc_feature_ind = old_data['pc_feature_ind']

    FS = 30000
    W = WINDOW_SIZE
    time_axis = np.arange(-W, W)
    CTX_W = W + 10

    width = 3.5
    fig, axes = plt.subplots(3, 2, figsize=(width, width * 1.7), dpi=300,
                             gridspec_kw={'width_ratios': [1.0, 0.85]})
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.96,
                        hspace=0.50, wspace=0.30)

    SFS_ST = 8
    SFS_LB = 7
    SFS_TK = 6
    SFS_AN = 7
    SFS_LG = 6

    for i, (spk_idx, row_label) in enumerate(zip(spike_idx, row_labels)):
        t_spike = int(spike_times[spk_idx])
        cid = int(spike_clusters[spk_idx])
        best_phys = int(best_ch_phys[spk_idx])

        # ===== Left: Multi-channel context =====
        ax_ctx = axes[i, 0]

        if cid < pc_feature_ind.shape[0]:
            channels_show = sorted(list(pc_feature_ind[cid][:6]))
        else:
            channels_show = list(range(min(6, raw_data.shape[0])))

        start = max(0, t_spike - CTX_W)
        end = min(raw_data.shape[1], t_spike + CTX_W)
        t_ctx_rel = np.arange(start - t_spike, end - t_spike)
        ctx_abs_s = (t_spike + t_ctx_rel) / FS

        for ch_idx in channels_show:
            tr = raw_data[ch_idx, start:end].astype(np.float64)
            tr_norm = (tr - np.mean(tr)) / (np.std(tr) + 1e-10)
            ch_pos = channels_show.index(ch_idx)
            if ch_idx == best_phys:
                ax_ctx.plot(ctx_abs_s, tr_norm + ch_pos * 2.5, 'b-', lw=1.2, alpha=1.0)
            else:
                ax_ctx.plot(ctx_abs_s, tr_norm + ch_pos * 2.5, 'k-', lw=0.3, alpha=0.7)

        spike_abs_s = t_spike / FS
        ax_ctx.axvspan(spike_abs_s - W/FS, spike_abs_s + W/FS, color='red', alpha=0.08)
        ax_ctx.axvline(spike_abs_s, color='red', linestyle=':', lw=0.5, alpha=0.5)
        ax_ctx.set_ylabel('Ch', fontsize=SFS_LB)
        ax_ctx.set_yticks([channels_show.index(ch) * 2.5 for ch in channels_show])
        ax_ctx.set_yticklabels([str(ch) for ch in channels_show], fontsize=SFS_TK)
        ax_ctx.tick_params(labelsize=SFS_TK)
        ax_ctx.set_xlim(ctx_abs_s[0], ctx_abs_s[-1])
        if i == 2:
            ax_ctx.set_xlabel('Time (s)', fontsize=SFS_LB)

        # ===== Right: Best-channel waveform comparison =====
        ax_wf = axes[i, 1]
        orig = orig_best[spk_idx, :]
        recon = recon_best[spk_idx, :]
        err_sig = np.abs(orig - recon)

        abs_time_s = (t_spike + time_axis) / FS

        ax_wf.plot(abs_time_s, orig, 'b-', label='Original', lw=0.8)
        ax_wf.plot(abs_time_s, recon, 'r--', label='Reconstructed', lw=0.8)

        y_range = max(orig.max(), recon.max()) - min(orig.min(), recon.min())
        if y_range < 1e-10:
            y_range = 1.0
        bar_base = min(orig.min(), recon.min()) - 0.3 * y_range
        bar_max_h = 0.2 * y_range
        bar_h = (err_sig / (err_sig.max() + 1e-10)) * bar_max_h if err_sig.max() > 0 else err_sig * 0
        bar_w = (abs_time_s[1] - abs_time_s[0]) * 0.6
        ax_wf.bar(abs_time_s, bar_h, bottom=bar_base, width=bar_w,
                  color='gray', alpha=0.5, edgecolor='gray', linewidth=0.1,
                  label='|Error|')

        if i == 2:
            y0, dy = 0.25, 0.10
            ax_wf.text(0.04, y0, f'Ch.{best_phys} @ {t_spike/FS:.3f}s',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='blue')
            ax_wf.text(0.04, y0 + dy, f'CR={best_cr[spk_idx]*100:.1f}%',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='purple')
            ax_wf.text(0.04, y0 + 2*dy, f'SNDR={best_sndr[spk_idx]:.1f} dB',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='green')
        else:
            y0, dy = 0.95, -0.10
            ax_wf.text(0.04, y0, f'SNDR={best_sndr[spk_idx]:.1f} dB',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='green')
            ax_wf.text(0.04, y0 + dy, f'CR={best_cr[spk_idx]*100:.1f}%',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='purple')
            ax_wf.text(0.04, y0 + 2*dy, f'Ch.{best_phys} @ {t_spike/FS:.3f}s',
                       transform=ax_wf.transAxes, ha='left', va='top',
                       fontsize=SFS_AN, color='blue')

        y_bottom = bar_base - 0.05 * y_range
        y_top = max(orig.max(), recon.max()) * 1.35
        ax_wf.set_ylim(y_bottom, y_top)
        ax_wf.set_ylabel('Amplitude', fontsize=SFS_LB, labelpad=0)
        ax_wf.tick_params(labelsize=SFS_TK)
        if i == 2:
            ax_wf.set_xlabel('Time (s)', fontsize=SFS_LB, labelpad=0)

        ax_ctx.text(-0.22, 1.0, row_label, transform=ax_ctx.transAxes,
                    fontsize=SFS_ST, fontweight='bold',
                    ha='left', va='bottom')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', lw=0.8, label='Original'),
        Line2D([0], [0], color='r', linestyle='--', lw=0.8, label='Reconstructed'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='gray', label='|Error|'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=SFS_LG, frameon=True,
               handlelength=1.5, handletextpad=0.5,
               bbox_to_anchor=(0.5, 0.0))

    save_path = os.path.join(BASE_DIR, WAVEFORM_FILE)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)
    del raw_data


# ==================== Best-Channel Bar Chart ====================
def plot_best_channel_barchart(results):
    group_results = results.get('_group_results')
    if group_results is None:
        print("  No group results to plot. Run evaluate_f1_score first.")
        return

    f1_vals = [r[2] for r in group_results]
    sndr_vals = [r[4] for r in group_results]
    cr_vals = [r[5] for r in group_results]

    SFS_TITLE = 10
    SFS_SUBTITLE = 9
    SFS_LABEL = 8
    SFS_LEGEND = 8
    SFS_TICK = 6
    SFS_ANNOT = 9

    C_SNDR = '#348ABD'
    C_CR   = '#E24A33'
    C_F1   = '#988ED5'

    width = 3.5
    fig, axes = plt.subplots(3, 1, figsize=(width, width * 1.0), dpi=300,
                             sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.08, top=0.92, hspace=0.45)

    x = np.arange(len(f1_vals))
    bw = 0.55
    group_labels = [str(r[0]) for r in group_results]

    n_total = len(results['best_sndr'])
    group_size = n_total // len(f1_vals)

    def get_per_spike_data(values_array):
        data = []
        for g in range(len(f1_vals)):
            s = g * group_size
            e = (g + 1) * group_size
            data.append(values_array[s:e])
        return data

    # ===== (a) SNDR box plot =====
    ax = axes[0]
    sndr_groups = get_per_spike_data(results['best_sndr'])
    sndr_box = [sg[np.isfinite(sg)] for sg in sndr_groups]
    bp_s = ax.boxplot(sndr_box, positions=x, widths=0.55, patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=C_SNDR, alpha=0.7, edgecolor='black', linewidth=0.5),
                      medianprops=dict(color='white', linewidth=1.5),
                      whiskerprops=dict(color=C_SNDR, linewidth=1.0),
                      capprops=dict(color=C_SNDR, linewidth=1.0))
    means_s = [np.mean(d) for d in sndr_box]
    ax.scatter(x, means_s, marker='D', s=15, color='white',
               edgecolors='black', linewidths=0.5, zorder=5)
    all_finite = np.concatenate(sndr_box)
    gmax_s, gmin_s = float(np.max(all_finite)), float(np.min(all_finite))
    for i in range(len(x)):
        top = bp_s['caps'][i*2+1].get_ydata()[0]
        bot = bp_s['caps'][i*2].get_ydata()[0]
        ax.text(x[i], top + 0.3, f'MAX={gmax_s:.1f}', ha='center', va='bottom',
                fontsize=5, color=C_SNDR)
        ax.text(x[i], bot - 0.3, f'MIN={gmin_s:.1f}', ha='center', va='top',
                fontsize=5, color=C_SNDR)
        ax.text(x[i], means_s[i], f'{means_s[i]:.1f}', ha='left', va='bottom',
                fontsize=5, color='black', fontweight='bold')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color=C_SNDR,linestyle='--',lw=1.2),
               Line2D([0],[0],color=C_SNDR,linestyle='-.',lw=1.2),
               plt.scatter([],[],marker='D',s=10,color='white',edgecolors='black',lw=0.5)],
              [f'MAX={gmax_s:.1f}dB', f'MIN={gmin_s:.1f}dB', 'Mean'],
              fontsize=5.5, loc='upper right', framealpha=0.8, ncol=3)
    ax.set_ylabel('SNDR (dB)', fontsize=SFS_LABEL)
    ax.set_title('SNDR', fontsize=SFS_SUBTITLE)
    ax.set_ylim(max(0, gmin_s - 1), min(50, gmax_s + 2))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=SFS_TICK)
    ax.tick_params(labelsize=SFS_TICK)

    # ===== (b) CR (as %) box plot =====
    target_cr_pct = CR_TARGET * 100
    ax = axes[1]
    cr_groups = get_per_spike_data(results['best_cr'])
    cr_box = [cg * 100 for cg in cr_groups]
    bp_c = ax.boxplot(cr_box, positions=x, widths=0.55, patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=C_CR, alpha=0.7, edgecolor='black', linewidth=0.5),
                      medianprops=dict(color='white', linewidth=1.5),
                      whiskerprops=dict(color=C_CR, linewidth=1.0),
                      capprops=dict(color=C_CR, linewidth=1.0))
    means_c = [np.mean(d) for d in cr_box]
    ax.scatter(x, means_c, marker='D', s=15, color='white',
               edgecolors='black', linewidths=0.5, zorder=5)
    h_target = ax.axhline(target_cr_pct, color='gray', linestyle='--', linewidth=0.8,
               label=f'Target={target_cr_pct:.0f}%')
    all_cr_flat = np.concatenate(cr_box)
    gmax_c, gmin_c = float(np.max(all_cr_flat)), float(np.min(all_cr_flat))
    for i in range(len(x)):
        top = bp_c['caps'][i*2+1].get_ydata()[0]
        bot = bp_c['caps'][i*2].get_ydata()[0]
        ax.text(x[i], top + 0.1, f'MAX={gmax_c:.1f}', ha='center', va='bottom',
                fontsize=5, color=C_CR)
        ax.text(x[i], bot - 0.1, f'MIN={gmin_c:.1f}', ha='center', va='top',
                fontsize=5, color=C_CR)
        ax.text(x[i], means_c[i], f'{means_c[i]:.1f}', ha='left', va='bottom',
                fontsize=5, color='black', fontweight='bold')
    ax.legend([h_target,
               Line2D([0],[0],color=C_CR,linestyle='--',lw=1.2),
               Line2D([0],[0],color=C_CR,linestyle='-.',lw=1.2),
               plt.scatter([],[],marker='D',s=10,color='white',edgecolors='black',lw=0.5)],
              [f'Target={target_cr_pct:.0f}%', f'MAX={gmax_c:.1f}%',
               f'MIN={gmin_c:.1f}%', 'Mean'],
              fontsize=5.5, loc='upper right', framealpha=0.8, ncol=2)
    ax.set_ylabel('Actual CR (%)', fontsize=SFS_LABEL)
    ax.set_title('Actual CR', fontsize=SFS_SUBTITLE)
    ax.set_ylim(30, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=SFS_TICK)
    ax.tick_params(labelsize=SFS_TICK)

    # ===== (c) F1 bar chart =====
    f1_pct = [v * 100 for v in f1_vals]
    ax = axes[2]
    bars = ax.bar(x, f1_pct, bw, color=C_F1, edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 105)
    y_lo = ax.get_ylim()[0]
    all_f1 = []
    for bar, val in zip(bars, f1_pct):
        all_f1.append(val)
        y_vis_bottom = max(y_lo, bar.get_y())
        y_vis_top = bar.get_y() + bar.get_height()
        y_text = y_vis_bottom + (y_vis_top - y_vis_bottom) * 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                f'{val:.1f}%', ha='center', va='center', fontsize=SFS_ANNOT, color='white', rotation=90)
    max_f1 = max(all_f1); min_f1 = min(all_f1)
    h_max = ax.axhline(max_f1, color=C_F1, linestyle='--', linewidth=1.2, alpha=0.8)
    h_min = ax.axhline(min_f1, color=C_F1, linestyle='-.', linewidth=1.2, alpha=0.8)
    ax.legend([h_max, h_min], [f'MAX={max_f1:.1f}%', f'MIN={min_f1:.1f}%'],
              fontsize=SFS_TICK, loc='upper right', framealpha=0.8, ncol=2)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=SFS_TICK)
    ax.set_ylabel('F1 Score (%)', fontsize=SFS_LABEL)
    ax.set_title('F1 Score', fontsize=SFS_SUBTITLE)
    ax.tick_params(labelsize=SFS_TICK)

    save_path = os.path.join(BASE_DIR, BAR_CHART_FILE)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Bar chart saved: {save_path}")
    plt.close(fig)


# ==================== Main ====================
def main():
    print("=" * 70)
    print(f"Best-Channel SD+CS Simulation — Neuropixel Dataset (CR={CR_TARGET:.0%})")
    print(f"Target CR: {CR_TARGET:.0%}, Window: {SEGMENT_LENGTH} samples")
    print("=" * 70)

    if not PLOT_ONLY:
        raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels = \
            load_neuropixel_data()

        print("\nProcessing spikes (best-channel only)...")
        results = get_or_process_best_channel(
            raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels)

        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"  Total spikes:       {len(results['orig_best'])}")
        print(f"  GT clusters:        {len(np.unique(results['gt_labels']))}")
        sndr_finite = results['best_sndr'][np.isfinite(results['best_sndr'])]
        if len(sndr_finite) > 0:
            print(f"  Mean SNDR (finite):  {np.mean(sndr_finite):.2f} dB")
            print(f"  SNDR range (finite): [{sndr_finite.min():.1f}, {sndr_finite.max():.1f}] dB")
        n_inf = np.sum(~np.isfinite(results['best_sndr']))
        if n_inf > 0:
            print(f"  Inf SNDR spikes:     {n_inf} (excluded from stats)")
        print(f"  Mean actual CR:     {np.mean(results['best_cr']):.4f}")
        print("=" * 70)

        evaluate_f1_score(results)
    else:
        results = get_or_process_best_channel(
            None, None, None, None, None, None)
        compute_per_channel_sndr_cr(results)
        evaluate_f1_score(results)

    plot_best_channel_barchart(results)
    plot_waveform_comparison(results)

    print(f"\n{'=' * 70}")
    print("All done!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
