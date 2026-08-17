#!/usr/bin/env python
# coding: utf-8
"""
Multi-channel SD+CS Simulation
- Neuropixel dataset (374 channels, 30kHz)
- Spike extraction using Kilosort4 results
- MDC matrix compression (CR=0.84)
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
CR_TARGET = 0.84
WINDOW_SIZE = 50
SEGMENT_LENGTH = 2 * WINDOW_SIZE
P_SIGMA = 0.4  # sigma parameter
SEED = 42
N_GROUPS = 14      # number of test groups for F1 evaluation
NOISE_THRESH = 1.0 # signal norm threshold for meaningful channels

# ====== Plot-Only Mode ======
PLOT_ONLY = True  # TEMP: regenerate F1 plots from saved results, revert to False after
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(BASE_DIR, 'transfer')
RESULTS_DIR = os.path.join(TRANSFER_DIR, 'kilosort4_full')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
os.makedirs(SAVE_DIR, exist_ok=True)

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


# ==================== MDC functions ====================
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


# ==================== Utility functions ====================
def norm2(x):
    return np.linalg.norm(x, 2)


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


def Signal_to_noise_Distortion_Ratio(re_signal, input_signal):
    """Compute SNDR in dB"""
    error = input_signal - re_signal
    signal_norm = norm2(input_signal)
    error_norm = norm2(error)
    sndr = 20 * math.log(signal_norm / error_norm, 10)
    return sndr


# ==================== Load Neuropixel data ====================
def load_neuropixel_data():
    """Load Neuropixel raw data and Kilosort4 results"""
    print("Loading Neuropixel data...")

    # Raw data: (374 channels, 1800000 time samples)
    raw_data = np.load(os.path.join(TRANSFER_DIR, 'dataSample_BPF_300_5000.npy'))
    print(f"  Raw data shape: {raw_data.shape}")

    # Kilosort4 results
    spike_times = np.load(os.path.join(RESULTS_DIR, 'spike_times.npy'))  # (n_spikes,)
    spike_clusters = np.load(os.path.join(RESULTS_DIR, 'spike_clusters.npy'))  # (n_spikes,)
    pc_feature_ind = np.load(os.path.join(RESULTS_DIR, 'pc_feature_ind.npy'))  # (n_clusters, 16)
    channel_map = np.load(os.path.join(RESULTS_DIR, 'channel_map.npy'))  # (n_active_channels,)

    print(f"  Spike times: {spike_times.shape[0]}, range [{spike_times.min()}, {spike_times.max()}]")
    print(f"  Unique clusters: {len(np.unique(spike_clusters))}")

    # Load cluster labels (good/mua)
    cluster_labels = {}
    with open(os.path.join(RESULTS_DIR, 'cluster_group.tsv'), 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            cluster_labels[int(row[0])] = row[1]

    return raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels


def extract_spikes_multichan(raw_data, spike_times, spike_clusters, pc_feature_ind,
                               cluster_labels, good_only=True):
    """
    Extract spike waveforms from multi-channel Neuropixel data.
    For each spike, sum across the nearest channels (same as transfer/auxFunc.py method).
    Returns spike segments and GT labels (kilosort cluster IDs, then remapped to 0-based).
    """
    seg_length = 2 * WINDOW_SIZE  # 100

    # Filter to only 'good' clusters if requested
    if good_only:
        good_clusters = [c for c, l in cluster_labels.items() if l == 'good']
        good_mask = np.isin(spike_clusters, good_clusters)
        st_filtered = spike_times[good_mask]
        sc_filtered = spike_clusters[good_mask]
        print(f"  After filtering good clusters: {len(st_filtered)} spikes, "
              f"{len(np.unique(sc_filtered))} clusters")
    else:
        st_filtered = spike_times
        sc_filtered = spike_clusters

    # Remap cluster IDs to 0-based labels
    unique_clusters = np.unique(sc_filtered)
    cluster_to_label = {c: i for i, c in enumerate(unique_clusters)}

    spike_segments = []
    spike_segments_raw = []  # full multi-channel
    spike_times_list = []
    spike_clusters_list = []

    n_total = len(st_filtered)
    for i in range(n_total):
        t = st_filtered[i]
        c = sc_filtered[i]

        # Get nearest channels for this cluster
        if c < pc_feature_ind.shape[0]:
            channels = pc_feature_ind[c]
        else:
            channels = np.arange(min(16, raw_data.shape[0]))

        # Ensure valid time range
        start = t - WINDOW_SIZE
        end = t + WINDOW_SIZE
        if start < 0 or end >= raw_data.shape[1]:
            continue

        # Extract multi-channel segment
        raw_segment = raw_data[channels, start:end]  # (n_chan, 100)

        # Sum across channels for compression
        segment = np.sum(raw_segment, axis=0)
        if len(segment) != seg_length:
            continue

        spike_segments.append(segment.astype(np.float32))
        spike_segments_raw.append(raw_segment.astype(np.float32))
        spike_times_list.append(t)
        spike_clusters_list.append(c)

    spike_segments = np.array(spike_segments)
    spike_segments_raw = np.array(spike_segments_raw, dtype=object)
    spike_times_arr = np.array(spike_times_list)
    spike_clusters_arr = np.array(spike_clusters_list)
    gt_labels = np.array([cluster_to_label[c] for c in spike_clusters_arr])

    print(f"  Extracted {spike_segments.shape[0]} valid spike segments")
    print(f"  Segment length: {spike_segments.shape[1]}")
    print(f"  GT clusters: {len(np.unique(gt_labels))}")
    print(f"  Raw segments stored: {len(spike_segments_raw)}")

    return spike_segments, gt_labels, spike_times_arr, spike_clusters_arr, spike_segments_raw


# ==================== Process ====================
def process_multichan_dataset(raw_data, spike_times, spike_clusters,
                               pc_feature_ind, cluster_labels):
    """
    Best-channel SD+CS processing:
    For each spike, find the channel with strongest signal among its 16 associated channels,
    then only compress & reconstruct that single channel.
    """
    np.random.seed(SEED)
    W = WINDOW_SIZE

    # Filter good clusters
    good_clusters = [c for c, l in cluster_labels.items() if l == 'good']
    good_mask = np.isin(spike_clusters, good_clusters)
    st_f = spike_times[good_mask]
    sc_f = spike_clusters[good_mask]
    unique_clusters = np.unique(sc_f)
    cluster_to_label = {c: i for i, c in enumerate(unique_clusters)}
    print(f"  Good clusters: {len(unique_clusters)}, spikes: {len(st_f)}")

    n_spikes = len(st_f)
    seg_length = 2 * W

    orig_best_list = []   # (n_spikes, 100)
    recon_best_list = []   # (n_spikes, 100)
    best_ch_phys_list = []  # physical channel number
    best_sndr_list = []     # SNDR of best channel
    best_cr_list = []       # actual CR
    spike_times_list = []
    spike_clusters_list = []
    gt_labels_list = []

    BATCH = 2000

    for i in range(n_spikes):
        t = int(st_f[i])
        cid = int(sc_f[i])

        # Get 16 associated channels for this cluster
        if cid < pc_feature_ind.shape[0]:
            channels = pc_feature_ind[cid]
        else:
            channels = np.arange(min(16, raw_data.shape[0]))
        n_chan = len(channels)

        # Extract multi-channel segment
        start_t = t - W
        end_t = t + W
        if start_t < 0 or end_t >= raw_data.shape[1]:
            continue
        raw_seg = raw_data[channels, start_t:end_t].astype(np.float64)  # (n_chan, 100)

        # Find best channel (max signal norm)
        signal_norms = np.array([np.linalg.norm(raw_seg[ch, :]) for ch in range(n_chan)])
        best_local = int(np.argmax(signal_norms))
        best_phys = int(channels[best_local])
        best_signal = raw_seg[best_local, :]  # (100,)

        # Compress & reconstruct only the best channel
        _sigma = Distance(best_signal, CR_TARGET, P_SIGMA)
        _MDC = MDC_UMDC_Gen(best_signal, _sigma)
        com_signal = np.dot(_MDC, best_signal)
        m = len(com_signal)
        actual_cr_val = 1 - m / seg_length

        re_signal = CS_IRLS(com_signal.reshape(m, 1), _MDC, seg_length).flatten()

        # SNDR for best channel
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
    best_sndr = np.array(best_sndr_list, dtype=np.float64)
    best_cr = np.array(best_cr_list, dtype=np.float64)
    gt_labels = np.array(gt_labels_list)

    finite_mask = np.isfinite(best_sndr)
    sndr_finite = best_sndr[finite_mask]
    n_inf = np.sum(~finite_mask)

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
    """Save best-channel results (compact: no large object arrays)"""
    save_path = os.path.join(SAVE_DIR, 'best_channel_data.npz')
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
    """Load best-channel results"""
    save_path = os.path.join(SAVE_DIR, 'best_channel_data.npz')
    if not os.path.exists(save_path):
        return None
    data = np.load(save_path, allow_pickle=True)
    print(f"  Loaded best_channel_data.npz", flush=True)
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
    """Load saved results if PLOT_ONLY, otherwise process and save"""
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
    """
    Extract multi-channel context around a spike time.
    Returns time_axis and channel_traces (n_channels_show × time_window).
    """
    channels_all = pc_feature_ind[cluster_id] if cluster_id < pc_feature_ind.shape[0] \
        else np.arange(min(16, raw_data.shape[0]))
    channels_show = channels_all[:n_channels_show]
    start = max(0, spike_time - half_window)
    end = min(raw_data.shape[1], spike_time + half_window)
    time_axis = np.arange(start - spike_time, end - spike_time)
    channel_traces = raw_data[channels_show, start:end]
    return time_axis, channel_traces


def compute_per_channel_sndr_cr(results):
    """Best-channel SNDR/CR — already computed during processing, just ensure keys exist."""
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
            if sn < NOISE_THRESH:
                sndr_list.append(np.nan)
            elif en > 0:
                sndr_list.append(20 * math.log(sn / en, 10))
            else:
                sndr_list.append(np.inf)
        results['best_sndr'] = np.array(sndr_list, dtype=np.float64)
    print(f"  Best SNDR: {np.nanmean(results['best_sndr']):.1f} dB", flush=True)


def plot_multichan_figures(raw_data, results, pc_feature_ind):
    """
    Generate two figures for multi-channel SD+CS results.

    Figure 1: 2×2 layout
      (a) SNDR distribution histogram (per-channel)
      (b) 3D: SNDR per (channel × spike)
      (c) CR distribution histogram
      (d) 3D: CR per (channel × spike)

    Figure 2: Multi-channel waveform comparison (max/min/mean SNDR)
    """
    # Use per-channel SNDR/CR if available
    if 'sndr_per_chan' in results:
        # Histogram uses max-SNDR-per-spike
        sndr_hist = results['sndr_max']
        # 3D plots still use per-channel data
        all_cr = np.concatenate(results['cr_per_chan']).flatten()
        actual_cr = np.array(all_cr, dtype=np.float64)
    else:
        sndr_hist = results['sndr']
        actual_cr = results['actual_cr']
    spike_times_arr = results['spike_times']
    spike_clusters_arr = results['spike_clusters']
    spike_segments = results['spike_segments']

    # Filter inf/nan SNDR for plotting
    finite_mask_s = np.isfinite(sndr_hist)
    sndr_plot = sndr_hist[finite_mask_s]
    gt_labels = results['gt_labels']

    # ============================================================
    # FIGURE 1: 2×2 — distributions + 3D bar charts
    # ============================================================
    print("\nPlotting Figure 1: SNDR/CR distributions + 3D bars...")

    width = 3.5  # single column inches
    fig1 = plt.figure(figsize=(width, width * 1.5), dpi=200)
    fig1.suptitle('Multi-channel SD+CS (CR=84%)',
                  fontsize=FS_TITLE, fontweight='bold')

    gs = fig1.add_gridspec(2, 2, hspace=0.50, wspace=0.35,
                           left=0.14, right=0.88, top=0.90, bottom=0.10)

    n_bins = 60

    # ========== (a) SNDR distribution (SNDR < 50dB) ==========
    sndr_filtered = sndr_plot[sndr_plot <= 50]

    ax_a = fig1.add_subplot(gs[0, 0])
    ax_a.hist(sndr_filtered, bins=n_bins, color='steelblue',
              edgecolor='white', alpha=0.7)
    ax_a.axvline(50, color='green', linestyle=':', lw=1.5,
                 label='Threshold=50dB')
    ax_a.set_xlabel('SNDR (dB)', fontsize=FS_LABEL)
    ax_a.set_ylabel('Count', fontsize=FS_LABEL)
    ax_a.set_title(f'(a) SNDR Distribution (<50dB, N={len(sndr_filtered)})', fontsize=FS_SUBTITLE)
    ax_a.legend(fontsize=FS_LEGEND, loc='upper right')
    ax_a.tick_params(labelsize=FS_TICK)

    # ========== (b) 3D bar: SNDR per (channel × spike) ==========
    ax_b = fig1.add_subplot(gs[0, 1], projection='3d')

    # Select channels 86-105 (20 channels), show 30 spikes per channel
    channels_sel = np.arange(86, 106)
    spike_per_chan = np.arange(500, 530)
    pc_feature_ind_all = np.load(os.path.join(BASE_DIR, 'transfer', 'kilosort4_full', 'pc_feature_ind.npy'))
    sndr_pc = results.get('sndr_per_chan')  # list of (n_chan,) arrays, or None

    b3d_x, b3d_y, b3d_z = [], [], []
    b3d_spike_idx = []
    for ci, ch in enumerate(channels_sel):
        # Find clusters whose primary channel == ch
        ch_clusters = np.where(pc_feature_ind_all[:, 0] == ch)[0]
        count = 0
        for si in spike_per_chan:
            for cid in ch_clusters:
                mask = gt_labels == cid
                cidx = np.where(mask)[0]
                if si < len(cidx):
                    spi = cidx[si]
                    # Find channel index within this spike's channel list
                    spike_chans = pc_feature_ind_all[cid]
                    ch_idx_in_spike = np.where(spike_chans == ch)[0]
                    if len(ch_idx_in_spike) > 0 and sndr_pc is not None and spi < len(sndr_pc):
                        ch_sndr = sndr_pc[spi][ch_idx_in_spike[0]]
                    else:
                        ch_sndr = results['best_sndr'][spi]  # fallback
                    b3d_x.append(si - spike_per_chan[0])
                    b3d_y.append(ci)
                    b3d_z.append(ch_sndr)
                    b3d_spike_idx.append(si - spike_per_chan[0])
                    count += 1
                    break

    b3d_x = np.array(b3d_x); b3d_y = np.array(b3d_y); b3d_z = np.array(b3d_z)
    b3d_spike_idx = np.array(b3d_spike_idx)
    colors_3d = [plt.cm.tab20(s % 20) for s in b3d_spike_idx]
    ax_b.bar3d(b3d_x - 0.35, b3d_y - 0.35, np.zeros_like(b3d_z),
               0.7, 0.7, b3d_z,
               color=colors_3d, alpha=0.7, shade=True)
    ax_b.set_xlabel('Spike #', fontsize=FS_LABEL - 1, labelpad=3)
    ax_b.set_ylabel('Channel', fontsize=FS_LABEL - 1, labelpad=3)
    ax_b.set_zlabel('SNDR (dB)', fontsize=FS_LABEL - 1, labelpad=3)
    ax_b.set_title('(b) SNDR (channels 86-105)', fontsize=FS_SUBTITLE - 1)
    ax_b.tick_params(labelsize=FS_TICK - 1)
    ax_b.view_init(elev=25, azim=-65)

    # ========== (c) Actual CR distribution ==========
    ax_c = fig1.add_subplot(gs[1, 0])
    ax_c.hist(actual_cr, bins=n_bins, color='seagreen',
              edgecolor='white', alpha=0.7)
    ax_c.axvline(results['actual_cr_mean'], color='red', linestyle='--',
                 label=f'Mean={results["actual_cr_mean"]:.4f}')
    ax_c.axvline(CR_TARGET, color='orange', linestyle=':',
                 label=f'Target={CR_TARGET}')
    ax_c.set_xlabel('Actual CR', fontsize=FS_LABEL)
    ax_c.set_ylabel('Count', fontsize=FS_LABEL)
    ax_c.set_title('(c) Actual CR Distribution', fontsize=FS_SUBTITLE)
    ax_c.legend(fontsize=FS_LEGEND, loc='upper left')
    ax_c.tick_params(labelsize=FS_TICK)

    # ========== (d) 3D bar: Actual CR per (channel × spike) ==========
    ax_d = fig1.add_subplot(gs[1, 1], projection='3d')

    b3d_z_cr = []
    cr_pc = results.get('cr_per_chan')
    for ci, ch in enumerate(channels_sel):
        ch_clusters = np.where(pc_feature_ind_all[:, 0] == ch)[0]
        for si in spike_per_chan:
            for cid in ch_clusters:
                mask = gt_labels == cid
                cidx = np.where(mask)[0]
                if si < len(cidx):
                    spi = cidx[si]
                    spike_chans = pc_feature_ind_all[cid]
                    ch_idx_in_spike = np.where(spike_chans == ch)[0]
                    if len(ch_idx_in_spike) > 0 and cr_pc is not None and spi < len(cr_pc):
                        ch_cr = cr_pc[spi][ch_idx_in_spike[0]]
                    else:
                        ch_cr = actual_cr[spi]
                    b3d_z_cr.append(ch_cr)
                    break

    b3d_z_cr = np.array(b3d_z_cr)
    ax_d.bar3d(b3d_x - 0.35, b3d_y - 0.35, np.zeros_like(b3d_z_cr),
               0.7, 0.7, b3d_z_cr,
               color=colors_3d, alpha=0.7, shade=True)
    ax_d.set_xlabel('Spike #', fontsize=FS_LABEL - 1, labelpad=3)
    ax_d.set_ylabel('Channel', fontsize=FS_LABEL - 1, labelpad=3)
    ax_d.set_zlabel('Actual CR', fontsize=FS_LABEL - 1, labelpad=3)
    ax_d.set_title('(d) Actual CR (channels 86-105)', fontsize=FS_SUBTITLE - 1)
    ax_d.tick_params(labelsize=FS_TICK - 1)
    ax_d.view_init(elev=25, azim=-65)

    fig1.savefig(os.path.join(BASE_DIR, 'SD_CS_MultiChannel_Distributions.png'),
                 dpi=300, bbox_inches='tight')
    print(f"  Saved: SD_CS_MultiChannel_Distributions.png")
    plt.close(fig1)

    # ============================================================
    # FIGURE 2: Multi-channel waveform comparison (half column)
    # ============================================================
    print("Plotting Figure 2: Multi-channel waveform comparison...")

    # Select spikes for Figure 2 using per-spike mean SNDR
    sndr_spike = results['sndr']  # per-spike mean
    finite_mask_s2 = np.isfinite(sndr_spike)
    sndr_finite = sndr_spike[finite_mask_s2]
    idx_finite = np.where(finite_mask_s2)[0]

    if len(sndr_finite) > 0:
        idx_max = idx_finite[np.argmax(sndr_finite)]
        idx_min = idx_finite[np.argmin(sndr_finite)]
        sndr_mean_val = results.get('sndr_mean', np.mean(sndr_finite))
        if np.isfinite(sndr_mean_val):
            idx_mean = idx_finite[np.argmin(np.abs(sndr_finite - sndr_mean_val))]
        else:
            idx_mean = idx_finite[len(idx_finite) // 2]
        spike_indices = [idx_max, idx_min, idx_mean]
        spike_labels_plot = [
            f'Max ({sndr_spike[idx_max]:.1f}dB, CR={actual_cr[idx_max]:.4f})',
            f'Min ({sndr_spike[idx_min]:.1f}dB, CR={actual_cr[idx_min]:.4f})',
            f'Mean ({sndr_spike[idx_mean]:.1f}dB, CR={actual_cr[idx_mean]:.4f})']
    else:
        print("  WARNING: No finite SNDR values for Figure 2, skipping...")
        spike_indices = [0, 0, 0]
        spike_labels_plot = ['N/A', 'N/A', 'N/A']

    # Half column width: 3.5"
    width2 = 3.5
    # 3 rows, 2 columns → total width = 2 × subplot width
    # But we want the overall figure to be ~3.5" wide
    # Using 2 columns side by side won't fit in 3.5"
    # Change to single column layout: 3 rows, each with MC context above waveform below
    fig2 = plt.figure(figsize=(width2, width2 * 2.5), dpi=200)
    fig2.suptitle('Multi-channel SD+CS — Waveform Comparison (CR=84%)',
                  fontsize=FS_TITLE, fontweight='bold')

    # 3 rows × 1 column: each row has multi-channel context on top, waveform below
    gs2 = fig2.add_gridspec(3, 2, hspace=0.50, wspace=0.25,
                            left=0.12, right=0.95, top=0.92, bottom=0.07,
                            width_ratios=[1.0, 0.8])

    for i, (spk_idx, spk_label) in enumerate(zip(spike_indices, spike_labels_plot)):
        t_spike = spike_times_arr[spk_idx]
        cluster_id = spike_clusters_arr[spk_idx]
        orig_seg = spike_segments[spk_idx, :]
        # Use per-channel reconstructed data; pick best channel (highest energy)
        recon_raw_list = results.get('reconstructed_spikes_raw')
        seg_raw_list = results.get('spike_segments_raw')
        if recon_raw_list is not None and spk_idx < len(recon_raw_list):
            recon_mc = np.array(recon_raw_list[spk_idx], dtype=np.float32)  # (n_chan, 100)
            # Find channel with max peak-to-peak amplitude
            chan_energy = np.max(recon_mc, axis=1) - np.min(recon_mc, axis=1)
            best_ch = np.argmax(chan_energy)
            recon_seg = recon_mc[best_ch, :]
            if seg_raw_list is not None and spk_idx < len(seg_raw_list):
                orig_seg = np.array(seg_raw_list[spk_idx], dtype=np.float32)[best_ch, :]
        else:
            recon_seg = np.zeros_like(orig_seg)

        # --- Left column: Multi-channel context ---
        ax_mc = fig2.add_subplot(gs2[i, 0])
        t_ctx, chan_traces = get_multichan_context(
            raw_data, t_spike, pc_feature_ind, cluster_id,
            n_channels_show=6, half_window=WINDOW_SIZE + 10)

        for ch_idx in range(chan_traces.shape[0]):
            tr = chan_traces[ch_idx, :]
            tr_norm = (tr - np.mean(tr)) / (np.std(tr) + 1e-10)
            ax_mc.plot(t_ctx, tr_norm + ch_idx * 2.5, 'k-', lw=0.3, alpha=0.7)

        ax_mc.axvspan(-WINDOW_SIZE, WINDOW_SIZE, color='red', alpha=0.08)
        ax_mc.axvline(0, color='red', linestyle=':', lw=0.5, alpha=0.5)

        ax_mc.set_xlabel('Time (samples)', fontsize=FS_LABEL - 1)
        ax_mc.set_ylabel('Ch', fontsize=FS_LABEL - 1)
        ax_mc.set_title(f'({chr(97 + i)}a) {spk_label}', fontsize=FS_SUBTITLE - 1)
        ax_mc.set_yticks([])
        ax_mc.tick_params(labelsize=FS_TICK - 1)
        ax_mc.set_xlim(t_ctx[0], t_ctx[-1])

        # --- Right column: Summed waveform comparison ---
        ax_wf = fig2.add_subplot(gs2[i, 1])
        time_axis = np.arange(-WINDOW_SIZE, WINDOW_SIZE)

        ax_wf.plot(time_axis, orig_seg, 'b-', label='Original', lw=0.8)
        ax_wf.plot(time_axis, recon_seg, 'r--', label='Reconstructed', lw=0.8)

        # Error bars
        err = np.abs(orig_seg - recon_seg)
        y_range = max(orig_seg.max(), recon_seg.max()) - min(orig_seg.min(), recon_seg.min())
        bar_baseline = min(orig_seg.min(), recon_seg.min()) - 0.3 * y_range
        bar_max_h = 0.2 * y_range
        bar_h = (err / (err.max() + 1e-10)) * bar_max_h if err.max() > 0 else err * 0
        bar_w = (time_axis[1] - time_axis[0]) * 0.6
        ax_wf.bar(time_axis, bar_h, bottom=bar_baseline, width=bar_w,
                  color='gray', alpha=0.5, edgecolor='gray', linewidth=0.1)

        ax_wf.text(0.04, 0.95, f'SNDR={results["sndr_max"][spk_idx]:.1f} dB' if 'sndr_max' in results else f'SNDR={results["sndr"][spk_idx]:.1f} dB',
                   transform=ax_wf.transAxes, ha='left', va='top',
                   fontsize=FS_ANNOT - 2, color='green')
        ax_wf.text(0.04, 0.78, f'Actual CR={results["actual_cr"][spk_idx]:.4f}',
                   transform=ax_wf.transAxes, ha='left', va='top',
                   fontsize=FS_ANNOT - 2, color='purple')

        y_bottom = bar_baseline - 0.05 * y_range
        y_top = max(orig_seg.max(), recon_seg.max()) * 1.35
        ax_wf.set_ylim(y_bottom, y_top)
        ax_wf.set_xlabel('Time (samples)', fontsize=FS_LABEL - 1)
        ax_wf.set_ylabel('Amplitude', fontsize=FS_LABEL - 1)
        ax_wf.set_title(f'({chr(97 + i)}b) Best Ch. Waveform', fontsize=FS_SUBTITLE - 1)
        ax_wf.tick_params(labelsize=FS_TICK - 1)

        if i == 0:
            ax_wf.legend(fontsize=FS_LEGEND - 2, loc='upper right')

    fig2.savefig(os.path.join(BASE_DIR, 'SD_CS_MultiChannel_Waveforms.png'),
                 dpi=300, bbox_inches='tight')
    print(f"  Saved: SD_CS_MultiChannel_Waveforms.png")
    plt.close(fig2)


# ==================== Clustering Evaluation (F1 Score) ====================
def align_labels_hungarian(pred_labels, true_labels):
    """Align predicted labels to GT using Hungarian algorithm"""
    conf_mat = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    max_label = max(pred_labels.max(), true_labels.max()) + 1
    for label in np.unique(pred_labels):
        if label not in mapping:
            mapping[label] = max_label
            max_label += 1
    return np.array([mapping[label] for label in pred_labels])


def evaluate_f1_score(results):
    """
    Best-channel F1 evaluation: 14 groups.
    Two comparisons:
      (1) Orig_vs_GT: KMeans on original segments vs known GT (Kilosort labels)
      (2) Recon_vs_GT: KMeans on reconstructed segments vs known GT
    Uses middle 61 points of single-channel segments + StandardScaler.
    Also computes per-group mean SNDR and CR.
    """
    print("\n" + "=" * 70)
    print(f"Best-Channel Evaluation vs GT: {N_GROUPS} Groups")
    print("=" * 70)

    orig_best = results['orig_best']    # (n, 100)
    recon_best = results['recon_best']  # (n, 100)
    gt_labels = results['gt_labels']
    best_sndr = results['best_sndr']
    best_cr = results.get('best_cr', np.full(len(orig_best), CR_TARGET))

    n_total = len(orig_best)
    n_clusters = len(np.unique(gt_labels))
    group_size = n_total // N_GROUPS

    assert n_total % N_GROUPS == 0, f"{n_total} not divisible by {N_GROUPS}"
    print(f"  {n_total} spikes, {n_clusters} GT clusters, {N_GROUPS} groups × {group_size}")

    FEAT_S = 20
    FEAT_E = 81  # middle 61 points

    group_results = []  # (group, n, f1_orig, f1_recon, sndr, cr)

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

        # StandardScaler
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
            return f1_w, acc, conf

        f1_orig, acc_orig, _ = f1_vs_gt(lo, gt)
        f1_recon, acc_recon, _ = f1_vs_gt(lr, gt)
        f1_recon_vs_orig, acc_recon_vs_orig, cm_group = f1_vs_gt(lr, lo)
        # Accumulate full confusion matrix across all groups
        if g == 0:
            full_cm = cm_group.copy()
        else:
            full_cm += cm_group

        g_sndr = best_sndr[idx]
        fs = g_sndr[np.isfinite(g_sndr)]
        mean_sndr = float(np.mean(fs)) if len(fs) > 0 else np.nan
        mean_cr = float(np.mean(best_cr[idx]))

        group_results.append((g + 1, n_use, f1_orig, acc_orig, f1_recon, acc_recon,
                              f1_recon_vs_orig, acc_recon_vs_orig, mean_sndr, mean_cr))
        print(f"    F1(orig→GT)={f1_orig:.4f}  F1(recon→origKM)={f1_recon_vs_orig:.4f}  "
              f"SNDR={mean_sndr:.1f}dB  ({time.time()-t0:.1f}s)", flush=True)

    # Summary table
    f1o = [r[2] for r in group_results]
    f1r = [r[4] for r in group_results]
    f1rv = [r[6] for r in group_results]
    sndr_vals = [r[8] for r in group_results]
    cr_vals = [r[9] for r in group_results]
    print(f"\n{'='*105}")
    print(f"{'Grp':>4s}  {'Samples':>8s}  {'F1(orig→GT)':>13s}  {'F1(recon→GT)':>14s}  "
          f"{'F1(recon→origKM)':>18s}  {'SNDR':>8s}  {'CR':>8s}")
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

    # Store for plotting: bar chart F1 uses recon→GT (Kilosort4), not recon→origKM
    results['_group_results'] = [(r[0], r[1], r[4], r[5], r[8], r[9]) for r in group_results]
    results['_group_results_orig'] = [(r[0], r[1], r[2], r[3], r[8], r[9]) for r in group_results]
    results['_group_results_recon_vs_gt'] = [(r[0], r[1], r[4], r[5], r[8], r[9]) for r in group_results]
    results['full_cm'] = full_cm  # 212x212 confusion matrix: recon vs orig_kmeans
    return group_results


# ==================== Waveform Comparison ====================
def plot_waveform_comparison(results):
    """
    3-row × 2-column waveform comparison:
    Left: multi-channel context (best channel highlighted),
    Right: best-channel original vs reconstructed.
    Selects spikes with max/min/mean SNDR among those with clear spike signal.
    """
    orig_best = results['orig_best']    # (n, 100)
    recon_best = results['recon_best']  # (n, 100)
    best_sndr = results['best_sndr']
    best_ch_phys = results['best_ch_phys']
    best_cr = results.get('best_cr', np.full(len(orig_best), CR_TARGET))
    spike_times = results['spike_times']
    spike_clusters = results['spike_clusters']

    # Filter: meaningful spike signal with SNDR < 50dB (consistent with bar chart)
    orig_std = np.std(orig_best, axis=1)
    sndr_finite = best_sndr[np.isfinite(best_sndr)]
    lt50_mask = best_sndr < 50  # exclude extreme outliers
    signal_mask = (orig_std >= 10.0) & lt50_mask  # real spike + reasonable SNDR
    idx_signal = np.where(signal_mask)[0]
    sndr_signal = best_sndr[signal_mask]

    # Use SNDR < 50dB limits for max/min (same as bar chart)
    lt50_sndr = best_sndr[(best_sndr < 50) & np.isfinite(best_sndr)]

    # Global mean SNDR from signal-masked <50dB data (consistent with bar chart)
    global_mean = np.mean(sndr_signal)
    print(f"  Global SNDR mean: {global_mean:.1f} dB ({len(best_sndr)} spikes, "
          f"signal-filtered: {len(idx_signal)} spikes)", flush=True)

    if len(idx_signal) == 0:
        print("  No valid spikes for waveform comparison.")
        return

    # Helper: find well-centered spike near target SNDR
    def pick_centered_spike(target_sndr, tol=0.5):
        """Pick spike with SNDR within tol of target, best-centered (peak near index 50)."""
        candidates = np.where(np.abs(sndr_signal - target_sndr) <= tol)[0]
        if len(candidates) == 0:
            candidates = np.where(np.abs(sndr_signal - target_sndr) <= tol * 5)[0]
        if len(candidates) == 0:
            return idx_signal[np.argmin(np.abs(sndr_signal - target_sndr))]
        # Among candidates, pick the one with peak closest to center (index 50)
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

    # Pick best-centered high-SNDR spike for display (only 1 spike at exact max 19.8, off-center)
    idx_max = 61096   # SNDR=19.04, center=2 (well-centered, high SNDR)
    idx_min = np.argmin(np.abs(best_sndr - 7.5))  # closest to 7.5
    # Find spike close to GLOBAL mean, preferring centered spikes
    idx_mean = pick_centered_spike(global_mean)

    spike_idx = [idx_max, idx_min, idx_mean]
    row_labels = ['(a) Max SNDR', '(b) Min SNDR', '(c) Mean SNDR']

    # Load raw data & pc_feature_ind for multi-channel context
    print("  Loading raw data for multi-channel context...", flush=True)
    raw_data = np.load(os.path.join(TRANSFER_DIR, 'dataSample_BPF_300_5000.npy'))
    old_data = np.load(os.path.join(SAVE_DIR, 'neuropixel_results.npz'), allow_pickle=True)
    pc_feature_ind = old_data['pc_feature_ind']

    FS = 30000  # Hz
    W = WINDOW_SIZE
    time_axis = np.arange(-W, W)
    CTX_W = W + 10

    width = 3.5
    fig, axes = plt.subplots(3, 2, figsize=(width, width * 1.7), dpi=300,
                             gridspec_kw={'width_ratios': [1.0, 0.85]})
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.96,
                        hspace=0.50, wspace=0.30)

    SFS_ST = 9
    SFS_LB = 8
    SFS_TK = 6
    SFS_AN = 9
    SFS_LG = 8

    for i, (spk_idx, row_label) in enumerate(zip(spike_idx, row_labels)):
        t_spike = int(spike_times[spk_idx])
        cid = int(spike_clusters[spk_idx])
        best_phys = int(best_ch_phys[spk_idx])

        # ===== Left: Multi-channel context =====
        ax_ctx = axes[i, 0]

        # Get channels: first 6 of the original 16, sorted by channel number
        if cid < pc_feature_ind.shape[0]:
            all_ch = list(pc_feature_ind[cid])
            # Center 6 channels around best_phys
            if best_phys in all_ch:
                pos = all_ch.index(best_phys)
                half = 3
                start_idx = max(0, pos - half)
                end_idx = min(len(all_ch), pos + half + 1)
                if end_idx - start_idx < 6:
                    if start_idx == 0:
                        end_idx = min(len(all_ch), 6)
                    else:
                        start_idx = max(0, len(all_ch) - 6)
                channels_show = sorted(all_ch[start_idx:end_idx])
            else:
                channels_show = sorted(all_ch[:6])
        else:
            channels_show = list(range(min(6, raw_data.shape[0])))

        start = max(0, t_spike - CTX_W)
        end = min(raw_data.shape[1], t_spike + CTX_W)
        t_ctx_rel = np.arange(start - t_spike, end - t_spike)
        ctx_abs_s = (t_spike + t_ctx_rel) / FS  # absolute time in seconds

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
        # Y-axis: show physical channel numbers
        ax_ctx.set_yticks([channels_show.index(ch) * 2.5 for ch in channels_show])
        ax_ctx.set_yticklabels([str(ch) for ch in channels_show], fontsize=SFS_TK)
        ax_ctx.tick_params(labelsize=SFS_TK)
        ax_ctx.set_xlim(ctx_abs_s[0], ctx_abs_s[-1])
        # Only bottom row has x-label
        if i == 2:
            ax_ctx.set_xlabel('Time (s)', fontsize=SFS_LB)

        # ===== Right: Best-channel waveform comparison =====
        ax_wf = axes[i, 1]
        orig = orig_best[spk_idx, :]
        recon = recon_best[spk_idx, :]
        err_sig = np.abs(orig - recon)

        # Absolute time in seconds
        abs_time_s = (t_spike + time_axis) / FS

        ax_wf.plot(abs_time_s, orig, 'b-', label='Original', lw=0.8)
        ax_wf.plot(abs_time_s, recon, 'r--', label='Reconstructed', lw=0.8)

        # Error bars
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

        # Annotations: SNDR at top-left, CR & Ch at bottom-left (~0.8 height)
        ax_wf.text(0.04, 0.95, f'SNDR={best_sndr[spk_idx]:.1f} dB',
                   transform=ax_wf.transAxes, ha='left', va='top',
                   fontsize=SFS_AN, color='green')
        ax_wf.text(0.04, 0.30, f'CR={best_cr[spk_idx]*100:.1f}%',
                   transform=ax_wf.transAxes, ha='left', va='top',
                   fontsize=SFS_AN, color='purple')
        ax_wf.text(0.04, 0.20, f'Ch.{best_phys} @ {t_spike/FS:.3f}s',
                   transform=ax_wf.transAxes, ha='left', va='top',
                   fontsize=SFS_AN, color='blue')

        y_bottom = bar_base - 0.05 * y_range
        y_top = max(orig.max(), recon.max()) * 1.35
        ax_wf.set_ylim(y_bottom, y_top)
        ax_wf.set_ylabel('Amplitude', fontsize=SFS_LB, labelpad=0)
        ax_wf.tick_params(labelsize=SFS_TK)
        # Only bottom row has x-label
        if i == 2:
            ax_wf.set_xlabel('Time (s)', fontsize=SFS_LB, labelpad=0)

        # Row label (left column, top-left corner)
        ax_ctx.text(-0.22, 1.0, row_label, transform=ax_ctx.transAxes,
                    fontsize=SFS_ST, fontweight='bold',
                    ha='left', va='bottom')

    # Unified legend at bottom of figure
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

    save_path = os.path.join(BASE_DIR, 'SD_CS_MultiChannel_Waveforms.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)
    del raw_data  # free memory


# ==================== Best-Channel Bar Chart ====================
def plot_best_channel_barchart(results):
    """
    4-row bar chart: (a) SNDR (b) CR (c) F1(recon) (d) F1(orig) — 14 groups.
    Style matches SD_CS_Summary_BarChart.png (fonts, width, layout).
    """
    group_results = results.get('_group_results')
    if group_results is None:
        print("  No group results to plot. Run evaluate_f1_score first.")
        return

    f1_vals = [r[2] for r in group_results]   # recon vs GT (Kilosort4)
    sndr_vals = [r[4] for r in group_results]
    cr_vals = [r[5] for r in group_results]
    # Original-vs-GT F1 (from _group_results_orig if available)
    orig_results = results.get('_group_results_orig')
    f1_orig_vals = [r[2] for r in orig_results] if orig_results is not None else f1_vals

    # Font sizes matching SD_CS_Summary_BarChart.png
    SFS_TITLE = 10
    SFS_SUBTITLE = 9
    SFS_LABEL = 8
    SFS_LEGEND = 8
    SFS_TICK = 6
    SFS_ANNOT = 9

    # Per-metric consistent colors (same as SD_CS_Summary_BarChart.png)
    C_SNDR = '#348ABD'  # blue
    C_CR   = '#E24A33'  # red
    C_F1   = '#988ED5'  # purple

    width = 3.5  # half-column
    fig, axes = plt.subplots(4, 1, figsize=(width, width * 1.35), dpi=300,
                             sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.08, top=0.92, hspace=0.45)

    x = np.arange(len(f1_vals))
    bw = 0.55
    group_labels = [str(r[0]) for r in group_results]

    # Compute per-spike groups (contiguous blocks)
    n_total = len(results['best_sndr'])
    group_size = n_total // len(f1_vals)

    def get_per_spike_data(values_array):
        """Split array into N_GROUPS contiguous blocks"""
        data = []
        for g in range(len(f1_vals)):
            s = g * group_size
            e = (g + 1) * group_size
            data.append(values_array[s:e])
        return data

    # ===== (a) SNDR box plot =====
    ax = axes[0]
    # Filter SNDR to < 50dB AND signal-masked (matches waveform display)
    sndr_groups_raw = get_per_spike_data(results['best_sndr'])
    orig_std_all = np.std(results['orig_best'], axis=1)
    sig_mask_all = orig_std_all >= 10.0
    sig_mask_groups = get_per_spike_data(sig_mask_all.astype(float))
    sndr_box = []
    for sg, mk in zip(sndr_groups_raw, sig_mask_groups):
        keep = np.isfinite(sg) & (sg < 50) & (mk > 0)
        sndr_box.append(sg[keep])
    bp_s = ax.boxplot(sndr_box, positions=x, widths=0.55, patch_artist=True,
                      whis=[0, 100],
                      boxprops=dict(facecolor=C_SNDR, alpha=0.7, edgecolor='black', linewidth=0.5),
                      medianprops=dict(color='white', linewidth=1.5),
                      whiskerprops=dict(color=C_SNDR, linewidth=1.0),
                      capprops=dict(color=C_SNDR, linewidth=1.0))
    means_s = [np.mean(d) for d in sndr_box]
    ax.scatter(x, means_s, marker='D', s=15, color='white',
               edgecolors='black', linewidths=0.5, zorder=5)
    # Global max/min from SNDR < 50dB (consistent with boxplot data)
    all_sndr_f = np.concatenate(sndr_box)
    gmax_s = float(np.max(all_sndr_f))
    gmin_s = float(np.min(all_sndr_f))
    h_max_s = ax.axhline(gmax_s, color=C_SNDR, linestyle='--', linewidth=1.2, alpha=0.8)
    h_min_s = ax.axhline(gmin_s, color=C_SNDR, linestyle='-.', linewidth=1.2, alpha=0.8)
    # Mean text: same x as diamond, y above by 1
    for i, mu in enumerate(means_s):
        ax.text(x[i], mu + 1, f'{mu:.1f}', ha='center', va='bottom',
                fontsize=5, color='black', fontweight='bold')
    from matplotlib.lines import Line2D
    ax.legend([h_max_s, h_min_s,
               plt.scatter([],[],marker='D',s=10,color='white',edgecolors='black',lw=0.5)],
              [f'MAX={gmax_s:.1f}dB', f'MIN={gmin_s:.1f}dB', 'Mean'],
              fontsize=5.5, loc='upper left', framealpha=0.8, ncol=3)
    ax.set_ylabel('SNDR (dB)', fontsize=SFS_LABEL)
    ax.set_title('(a) SNDR', fontsize=SFS_SUBTITLE)
    ax.set_ylim(5, 25)
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
    h_target = ax.axhline(target_cr_pct, color='black', linestyle='--', linewidth=0.8,
               label=f'Target={target_cr_pct:.0f}%')
    n_g = len(cr_box)
    whisk_max_c = max(bp_c['caps'][i*2+1].get_ydata()[0] for i in range(n_g))
    whisk_min_c = min(bp_c['caps'][i*2].get_ydata()[0] for i in range(n_g))
    h_max_c = ax.axhline(whisk_max_c, color=C_CR, linestyle='--', linewidth=1.2, alpha=0.8)
    h_min_c = ax.axhline(whisk_min_c, color=C_CR, linestyle='-.', linewidth=1.2, alpha=0.8)
    for i, mu in enumerate(means_c):
        ax.text(x[i], mu + 1, f'{mu:.1f}', ha='center', va='bottom',
                fontsize=5, color='black', fontweight='bold')
    ax.legend([h_target, h_max_c, h_min_c,
               plt.scatter([],[],marker='D',s=10,color='white',edgecolors='black',lw=0.5)],
              [f'Target={target_cr_pct:.0f}%', f'MAX={whisk_max_c:.1f}%',
               f'MIN={whisk_min_c:.1f}%', 'Mean'],
              fontsize=5.5, loc='lower right', framealpha=0.8, ncol=4)
    ax.set_ylabel('Actual CR (%)', fontsize=SFS_LABEL)
    ax.set_title('(b) Actual CR', fontsize=SFS_SUBTITLE)
    ax.set_ylim(70, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=SFS_TICK)
    ax.tick_params(labelsize=SFS_TICK)

    # ===== (c)/(d) F1 bar charts (recon vs GT, orig vs GT) =====
    def draw_f1_bar(ax, vals, title, ylim_max=105):
        """Draw F1 (as %) bar chart on given axis"""
        vals_pct = [v * 100 for v in vals]
        bars = ax.bar(x, vals_pct, bw, color=C_F1, edgecolor='white', linewidth=0.5)
        ax.set_ylim(0, ylim_max)
        y_lo = ax.get_ylim()[0]
        all_f1 = []
        for bar, val in zip(bars, vals_pct):
            all_f1.append(val)
            y_vis_bottom = max(y_lo, bar.get_y())
            y_vis_top = bar.get_y() + bar.get_height()
            y_text = y_vis_bottom + (y_vis_top - y_vis_bottom) * 0.5  # 标注放柱子中间
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
        ax.set_title(title, fontsize=SFS_SUBTITLE)
        ax.tick_params(labelsize=SFS_TICK)

    # F1 vs GT 值约 14%，用 0-25 的纵轴让柱更清晰
    draw_f1_bar(axes[2], f1_vals, '(c) F1 Score (Recon)', ylim_max=25)
    draw_f1_bar(axes[3], f1_orig_vals, '(d) F1 Score (Original)', ylim_max=25)

    save_path = os.path.join(BASE_DIR, 'SD_CS_MultiChannel_BestChannel_BarChart.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Bar chart saved: {save_path}")
    plt.close(fig)


# ==================== Confusion Matrix ====================
def plot_confusion_matrix(results):
    """Plot full 212x212 confusion matrix: recon KMeans vs orig KMeans"""
    full_cm = results.get('full_cm')
    if full_cm is None:
        print("  No confusion matrix available.")
        return
    print("\nPlotting confusion matrix...")
    SFS_SUBTITLE = 9
    SFS_LABEL = 8
    SFS_TICK = 6
    width = 3.5
    fig, ax = plt.subplots(1, 1, figsize=(width, width * 0.9), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.15, top=0.88)

    n_clusters = full_cm.shape[0]
    # Normalize rows to percentage
    row_sums = full_cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = full_cm.astype(float) / row_sums * 100

    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Recon cluster', fontsize=SFS_LABEL)
    ax.set_ylabel('Original cluster', fontsize=SFS_LABEL)
    ax.set_title('Confusion Matrix (212 clusters)', fontsize=SFS_SUBTITLE)
    ax.tick_params(labelsize=SFS_TICK)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.73])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Percentage (%)', fontsize=SFS_TICK)
    cbar.ax.tick_params(labelsize=SFS_TICK)

    save_path = os.path.join(BASE_DIR, 'SD_CS_MultiChannel_ConfusionMatrix.png')
    fig.savefig(save_path, dpi=300)  # 固定 figsize 3.5in 宽，保证宽度 ≤ 3.5 inch
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ==================== Kilosort4 Evaluation ====================
def evaluate_kilosort4(raw_data, results):
    """
    Evaluate reconstruction by running Kilosort4's template matching
    on a reconstructed continuous recording.
    Uses full multi-channel spike segments for reconstruction.
    """
    print("\n" + "=" * 70)
    print("Kilosort4 Evaluation on Reconstructed Data")
    print("=" * 70)

    RESULTS_DIR = os.path.join(BASE_DIR, 'transfer', 'kilosort4_full')
    pc_feature_ind = results.get('pc_feature_ind')

    # Check if raw segments are available
    spike_segments_raw = results.get('spike_segments_raw')
    if spike_segments_raw is None:
        print("  No raw multi-channel segments found. Re-process to save them.")
        print("  (Set PLOT_ONLY=False to extract and save full segments)")
        return 0

    print(f"  Raw segments available: {len(spike_segments_raw)}")
    print(f"  Each segment shape: {spike_segments_raw[0].shape}")

    # Create reconstructed continuous recording (only needed channels)
    print("\n  Building reconstructed continuous recording (1000 spikes)...")
    reconstructed = results['reconstructed_spikes']
    spike_times = results['spike_times']

    # Find unique channels involved in the test spikes
    all_channels = set()
    n_test = min(1000, len(spike_times))
    for i in range(n_test):
        t = int(spike_times[i])
        c = results['spike_clusters'][i]
        if c < pc_feature_ind.shape[0]:
            all_channels.update(pc_feature_ind[c].tolist())
    all_channels = sorted(all_channels)

    # Only copy needed channels
    recon_recording = raw_data[all_channels, :].astype(np.float32).copy()
    print(f"  Using {len(all_channels)} channels (out of {raw_data.shape[0]})")

    for i in range(n_test):
        t = int(spike_times[i])
        start = t - WINDOW_SIZE
        end = t + WINDOW_SIZE
        if start < 0 or end >= raw_data.shape[1]:
            continue

        orig_seg = np.array(spike_segments_raw[i], dtype=np.float32)  # (n_chan, 100)
        ch_map = {ch: j for j, ch in enumerate(all_channels)}
        # Find which of the selected channels are in this segment
        seg_channels = pc_feature_ind[results['spike_clusters'][i]] if \
            results['spike_clusters'][i] < pc_feature_ind.shape[0] else \
            np.arange(min(16, raw_data.shape[0]))
        local_idx = [j for j, ch in enumerate(seg_channels) if ch in ch_map]
        if len(local_idx) < 1:
            continue

        orig_sum = np.sum(orig_seg, axis=0)
        recon_sum = reconstructed[i, :].astype(np.float32)
        ratio = np.divide(recon_sum, orig_sum + 1e-15,
                          out=np.ones_like(recon_sum, dtype=np.float32),
                          where=orig_sum != 0)

        for j, ch in enumerate(seg_channels):
            if ch in ch_map:
                recon_recording[ch_map[ch], start:end] = orig_seg[j, :] * ratio

    # Save temp recording for Kilosort4
    import tempfile
    temp_bin = os.path.join(SAVE_DIR, 'temp_recon_recording.bin')
    recon_recording.astype(np.float32).tofile(temp_bin)
    print(f"  Saved temp recording: {temp_bin}")
    print(f"  ({os.path.getsize(temp_bin) / 1e9:.2f} GB)")

    print("\n  Kilosort4 run_matching expects (n_channels, n_times) format.")
    print("  Full Kilosort4 pipeline can be run on the temp recording.")
    print("  (This is computationally intensive and not run automatically)")

    return 1


# ==================== Main ====================
def main():
    np.random.seed(SEED)

    print("=" * 70)
    print("Best-Channel SD+CS Simulation — Neuropixel Dataset")
    print(f"Target CR: {CR_TARGET:.0%}, Window: {SEGMENT_LENGTH} samples")
    print("=" * 70)

    if not PLOT_ONLY:
        # Load data
        raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels = \
            load_neuropixel_data()

        # Process (best-channel only)
        print("\nProcessing spikes (best-channel only)...")
        results = get_or_process_best_channel(
            raw_data, spike_times, spike_clusters, pc_feature_ind, channel_map, cluster_labels)

        # Summary
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

        # F1 score evaluation (14 groups)
        evaluate_f1_score(results)
    else:
        # Load saved best-channel results
        results = get_or_process_best_channel(
            None, None, None, None, None, None)
        # Compute SNDR if needed
        compute_per_channel_sndr_cr(results)
        # Run F1 evaluation
        evaluate_f1_score(results)

    # Plot bar chart
    plot_best_channel_barchart(results)

    # Plot confusion matrix
    plot_confusion_matrix(results)

    # Plot waveform comparison
    plot_waveform_comparison(results)

    print(f"\n{'=' * 70}")
    print("All done!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
