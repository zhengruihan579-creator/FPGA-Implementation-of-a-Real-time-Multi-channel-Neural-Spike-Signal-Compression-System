#!/usr/bin/env python
# coding: utf-8
"""
Single-channel SD+CS Simulation
- Quiroga datasets (20 datasets, 4 groups)
- Spike extraction by spike_times (same as FE_Try/common.py)
- MDC matrix compression (CR=0.84)
- IRLS reconstruction
- PCA + K-Means clustering evaluation with confusion matrix heatmap
"""

import math
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
CR_TARGET = 0.84
WINDOW_SIZE = 50
SEGMENT_LENGTH = 2 * WINDOW_SIZE
P_SIGMA = 0.4  # sigma parameter
N_CLUSTERS = 3
PCA_COMPONENTS = 2
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))

# ====== Plot-Only Mode ======
# Set PLOT_ONLY = True to skip re-processing and just plot from saved results
PLOT_ONLY = True
# =============================

# Font: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Font sizes
FS_TITLE = 11
FS_SUBTITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 8
FS_ANNOT = 9

# ==================== All 20 datasets in 4 groups ====================
DATASET_GROUPS = [
    {
        'name': 'C_Easy1',
        'title': 'Easy1',
        'label': 'E1',
        'datasets': [
            'C_Easy1_noise005', 'C_Easy1_noise01', 'C_Easy1_noise015',
            'C_Easy1_noise02', 'C_Easy1_noise025', 'C_Easy1_noise03',
            'C_Easy1_noise035', 'C_Easy1_noise04'
        ]
    },
    {
        'name': 'C_Easy2',
        'title': 'Easy2',
        'label': 'E2',
        'datasets': [
            'C_Easy2_noise005', 'C_Easy2_noise01',
            'C_Easy2_noise015', 'C_Easy2_noise02'
        ]
    },
    {
        'name': 'C_Difficult1',
        'title': 'Difficult1',
        'label': 'D1',
        'datasets': [
            'C_Difficult1_noise005', 'C_Difficult1_noise01',
            'C_Difficult1_noise015', 'C_Difficult1_noise02'
        ]
    },
    {
        'name': 'C_Difficult2',
        'title': 'Difficult2',
        'label': 'D2',
        'datasets': [
            'C_Difficult2_noise005', 'C_Difficult2_noise01',
            'C_Difficult2_noise015', 'C_Difficult2_noise02'
        ]
    }
]


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
    while (epsilong > 10e-9) and (times < len(y) / 4):
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


def load_quiroga_dataset(mat_path):
    """Load Quiroga .mat dataset"""
    data = scio.loadmat(mat_path)
    raw_data = data['data'].flatten()
    spike_times = data['spike_times'][0, 0].flatten()
    spike_labels = data['spike_class'][0, 0].flatten()
    return raw_data, spike_times, spike_labels


def extract_spike_segments(signal_data, spike_positions, spike_labels, window_size=WINDOW_SIZE):
    """Extract spike segments by spike positions (same as FE_Try/common.py)"""
    segment_length = 2 * window_size
    spike_segments = []
    valid_labels = []
    for i, spike_pos in enumerate(spike_positions):
        start_idx = int(spike_pos) - window_size
        end_idx = int(spike_pos) + window_size
        if start_idx >= 0 and end_idx < len(signal_data):
            segment = signal_data[start_idx:end_idx]
            if len(segment) == segment_length:
                spike_segments.append(segment)
                valid_labels.append(spike_labels[i])
    return np.array(spike_segments), np.array(valid_labels)


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


# ==================== Process one dataset ====================
def process_dataset(dataset_name, return_waveform=False):
    """Load, compress with MDC, reconstruct with IRLS, cluster, evaluate"""
    np.random.seed(SEED)
    mat_path = os.path.join(DATABASE_DIR, f'{dataset_name}.mat')

    # Load and extract spikes
    raw_data, spike_times, spike_labels = load_quiroga_dataset(mat_path)
    spike_segments, gt_labels = extract_spike_segments(raw_data, spike_times, spike_labels)
    gt_labels = gt_labels - 1  # 1-based -> 0-based

    n_spikes = spike_segments.shape[0]
    seg_length = spike_segments.shape[1]

    # SD+CS compression & IRLS reconstruction
    sndr_list = []
    actual_cr_list = []
    reconstructed_spikes = np.empty((0, seg_length))

    for i in range(n_spikes):
        input_signal = spike_segments[i, :]
        _sigma = Distance(input_signal, CR_TARGET, P_SIGMA)
        _MDC = MDC_UMDC_Gen(input_signal, _sigma)
        com_signal = np.dot(_MDC, input_signal)
        m = len(com_signal)
        actual_cr_list.append(1 - m / seg_length)
        re_signal = CS_IRLS(com_signal.reshape(com_signal.shape[0], 1),
                            _MDC, len(input_signal))
        sndr = Signal_to_noise_Distortion_Ratio(
            re_signal, input_signal.reshape(input_signal.shape[0], 1))
        sndr_list.append(sndr)
        reconstructed_spikes = np.append(
            reconstructed_spikes, re_signal.reshape(1, seg_length), axis=0)
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{n_spikes} spikes processed")

    sndr_array = np.array(sndr_list)
    actual_cr_array = np.array(actual_cr_list)

    # PCA + K-Means on reconstructed signal
    pca_recon = PCA(n_components=PCA_COMPONENTS)
    recon_features = pca_recon.fit_transform(reconstructed_spikes)
    kmeans_recon = KMeans(n_clusters=N_CLUSTERS, random_state=5, n_init=10)
    recon_labels = kmeans_recon.fit_predict(recon_features)

    aligned_recon = align_labels_hungarian(recon_labels, gt_labels)
    cm_recon = confusion_matrix(gt_labels, aligned_recon)
    recon_f1 = f1_score(gt_labels, aligned_recon, average='weighted')

    # PCA + K-Means on original signal
    pca_orig = PCA(n_components=PCA_COMPONENTS)
    orig_features = pca_orig.fit_transform(spike_segments)
    kmeans_orig = KMeans(n_clusters=N_CLUSTERS, random_state=5, n_init=10)
    orig_labels = kmeans_orig.fit_predict(orig_features)
    aligned_orig = align_labels_hungarian(orig_labels, gt_labels)
    orig_f1 = f1_score(gt_labels, aligned_orig, average='weighted')

    # Original vs Reconstructed clustering confusion matrix
    aligned_recon_to_orig = align_labels_hungarian(recon_labels, orig_labels)
    cm_orig_vs_recon = confusion_matrix(orig_labels, aligned_recon_to_orig)

    result = {
        'name': dataset_name,
        'n_spikes': n_spikes,
        'sndr': sndr_array,
        'sndr_mean': np.mean(sndr_array),
        'sndr_std': np.std(sndr_array),
        'actual_cr': actual_cr_array,
        'actual_cr_mean': np.mean(actual_cr_array),
        'orig_f1': orig_f1,
        'recon_f1': recon_f1,
        'cm_recon': cm_recon,
        'cm_orig_vs_recon': cm_orig_vs_recon,
        'gt_labels': gt_labels,
        'spike_segments': spike_segments,
        'reconstructed_spikes': reconstructed_spikes,
    }

    if return_waveform:
        result['spike_segments_full'] = spike_segments
        result['reconstructed_full'] = reconstructed_spikes
        result['sndr_array'] = sndr_array

    return result


# ==================== Save/Load results ====================
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
os.makedirs(SAVE_DIR, exist_ok=True)


def save_dataset_results(result):
    """Save processed dataset results to .npz file"""
    save_path = os.path.join(SAVE_DIR, f'{result["name"]}_results.npz')
    np.savez_compressed(save_path,
                        name=result['name'],
                        n_spikes=result['n_spikes'],
                        sndr_mean=result['sndr_mean'],
                        sndr_std=result['sndr_std'],
                        actual_cr_mean=result['actual_cr_mean'],
                        orig_f1=result['orig_f1'],
                        recon_f1=result['recon_f1'],
                        cm_recon=result['cm_recon'],
                        cm_orig_vs_recon=result['cm_orig_vs_recon'],
                        gt_labels=result['gt_labels'],
                        spike_segments=result['spike_segments'],
                        reconstructed_spikes=result['reconstructed_spikes'],
                        sndr=result['sndr'],
                        actual_cr=result['actual_cr'])
    return save_path


def load_dataset_results(dataset_name):
    """Load saved dataset results if available"""
    save_path = os.path.join(SAVE_DIR, f'{dataset_name}_results.npz')
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        # 兼容旧文件：若缺 cm_orig_vs_recon，用已保存波形即时重算（无需重新压缩）
        if 'cm_orig_vs_recon' not in data.files:
            segs = data['spike_segments']
            recon = data['reconstructed_spikes']
            orig_labels = KMeans(n_clusters=N_CLUSTERS, random_state=5, n_init=10).fit_predict(
                PCA(n_components=2).fit_transform(segs))
            recon_labels = KMeans(n_clusters=N_CLUSTERS, random_state=5, n_init=10).fit_predict(
                PCA(n_components=2).fit_transform(recon))
            aligned = align_labels_hungarian(recon_labels, orig_labels)
            cm_orig_vs_recon = confusion_matrix(orig_labels, aligned)
        else:
            cm_orig_vs_recon = data['cm_orig_vs_recon']
        return {
            'name': str(data['name']),
            'n_spikes': int(data['n_spikes']),
            'sndr_mean': float(data['sndr_mean']),
            'sndr_std': float(data['sndr_std']),
            'actual_cr_mean': float(data['actual_cr_mean']),
            'orig_f1': float(data['orig_f1']),
            'recon_f1': float(data['recon_f1']),
            'cm_recon': data['cm_recon'],
            'cm_orig_vs_recon': cm_orig_vs_recon,
            'gt_labels': data['gt_labels'],
            'spike_segments': data['spike_segments'],
            'reconstructed_spikes': data['reconstructed_spikes'],
            'sndr': data['sndr'],
            'actual_cr': data['actual_cr'],
        }
    return None


def get_or_process_dataset(dataset_name, return_waveform=False):
    """Load saved result if PLOT_ONLY, otherwise process (and save)"""
    if PLOT_ONLY:
        result = load_dataset_results(dataset_name)
        if result is not None and not return_waveform:
            print(f"  Loaded saved result for {dataset_name}")
            return result
        elif result is not None and return_waveform:
            print(f"  {dataset_name} needs waveform data, re-processing...")
    result = process_dataset(dataset_name, return_waveform=return_waveform)
    save_dataset_results(result)
    return result


def plot_group(group_results, group_info):
    """Create figure for one group: (a)(b)(c) + confusion heatmaps"""
    n_datasets = len(group_results)
    group_title = group_info['title']

    # Layout: row0=(a)(b)(c), then heatmaps in up to 4 columns
    heatmap_cols = min(4, n_datasets)
    n_heatmap_rows = (n_datasets + heatmap_cols - 1) // heatmap_cols
    total_rows = 1 + n_heatmap_rows
    total_cols = max(heatmap_cols, 3)

    fig = plt.figure(figsize=(4.5 * total_cols, 3.2 * total_rows), dpi=200)
    fig.suptitle(f'SD+CS Simulation — {group_title} Group (CR={CR_TARGET:.0%})',
                 fontsize=FS_TITLE, fontweight='bold')

    gs = GridSpec(total_rows, total_cols, figure=fig,
                  hspace=0.40, wspace=0.35,
                  left=0.07, right=0.88, top=0.90, bottom=0.08)

    d0 = group_results[0]

    # --- (a) SNDR distribution ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(d0['sndr'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax_a.axvline(d0['sndr_mean'], color='red', linestyle='--',
                 label=f'Mean={d0["sndr_mean"]:.1f}dB')
    ax_a.set_xlabel('SNDR (dB)', fontsize=FS_LABEL)
    ax_a.set_ylabel('Count', fontsize=FS_LABEL)
    ax_a.set_title('(a) SNDR Distribution', fontsize=FS_SUBTITLE)
    ax_a.legend(fontsize=FS_LEGEND, loc='upper right')
    ax_a.tick_params(labelsize=FS_TICK)

    # --- (b) Actual CR distribution ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(d0['actual_cr'], bins=50, color='seagreen', edgecolor='white', alpha=0.8)
    ax_b.axvline(d0['actual_cr_mean'], color='red', linestyle='--',
                 label=f'Mean={d0["actual_cr_mean"]:.4f}')
    ax_b.axvline(CR_TARGET, color='orange', linestyle=':',
                 label=f'Target={CR_TARGET}')
    ax_b.set_xlabel('Actual CR', fontsize=FS_LABEL)
    ax_b.set_ylabel('Count', fontsize=FS_LABEL)
    ax_b.set_title('(b) CR Distribution', fontsize=FS_SUBTITLE)
    ax_b.legend(fontsize=FS_LEGEND, loc='upper left')
    ax_b.tick_params(labelsize=FS_TICK)

    # --- (c) Waveform comparison ---
    ax_c = fig.add_subplot(gs[0, 2])
    time_axis = np.arange(-WINDOW_SIZE, WINDOW_SIZE) / 50  # ms
    ax_c.plot(time_axis, d0['spike_segments'][0, :], 'b-', label='Original', lw=1.2)
    ax_c.plot(time_axis, d0['reconstructed_spikes'][0, :], 'r--',
              label=f'Recon ({d0["sndr"][0]:.1f}dB)', lw=1.2)
    ax_c.set_xlabel('Time (ms)', fontsize=FS_LABEL)
    ax_c.set_ylabel('Amplitude', fontsize=FS_LABEL)
    ax_c.set_title('(c) Waveform Comparison', fontsize=FS_SUBTITLE)
    ax_c.legend(fontsize=FS_LEGEND)
    ax_c.tick_params(labelsize=FS_TICK)

    # Hide unused cell in first row if total_cols > 3
    for col in range(3, total_cols):
        ax_empty = fig.add_subplot(gs[0, col])
        ax_empty.axis('off')

    # --- (i) Confusion matrix heatmaps (reconstructed vs GT) ---
    for idx, ds in enumerate(group_results):
        row = 1 + idx // heatmap_cols
        col = idx % heatmap_cols
        ax_h = fig.add_subplot(gs[row, col])

        cm = ds['cm_recon']
        cm_perc = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax_h.imshow(cm_perc, cmap='Blues', vmin=0, vmax=100)

        # Short dataset label
        lbl = ds['name'].replace('C_Easy1_noise', 'E1n') \
                        .replace('C_Easy2_noise', 'E2n') \
                        .replace('C_Difficult1_noise', 'D1n') \
                        .replace('C_Difficult2_noise', 'D2n')

        ax_h.set_title(f'({chr(ord("d") + idx)}) {lbl}  F1={ds["recon_f1"]:.3f}',
                       fontsize=FS_SUBTITLE)

        # Cell annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f'{cm[i, j]}\n({cm_perc[i, j]:.0f}%)'
                clr = 'white' if cm_perc[i, j] > 50 else 'black'
                ax_h.text(j, i, txt, ha='center', va='center',
                          fontsize=FS_ANNOT, color=clr)

        ax_h.set_xticks(range(3))
        ax_h.set_yticks(range(3))
        ax_h.set_xticklabels(['1', '2', '3'], fontsize=FS_TICK)
        ax_h.set_yticklabels(['1', '2', '3'], fontsize=FS_TICK)
        ax_h.set_xlabel('Predicted', fontsize=FS_LABEL)
        ax_h.set_ylabel('True', fontsize=FS_LABEL)

    # Hide unused heatmap subplots
    for idx in range(n_datasets, n_heatmap_rows * heatmap_cols):
        row = 1 + idx // heatmap_cols
        col = idx % heatmap_cols
        if row < total_rows:
            ax_h = fig.add_subplot(gs[row, col])
            ax_h.axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Percentage (%)', fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    # Save
    output_png = os.path.join(
        BASE_DIR, f'SD_CS_SingleChannel_{group_info["name"]}_CR{CR_TARGET:.0%}.png')
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_png}")
    plt.close(fig)

    # Summary
    print(f"\n  === {group_title} Summary ===")
    print(f"  {'Dataset':30s} {'SNDR':>8s} {'CR':>8s} {'F1_orig':>8s} {'F1_recon':>8s}")
    for ds in group_results:
        print(f"  {ds['name']:30s} {ds['sndr_mean']:>6.1f}dB "
              f"{ds['actual_cr_mean']:>7.4f} "
              f"{ds['orig_f1']:>7.3f} {ds['recon_f1']:>7.3f}")


# ==================== Main ====================
def main():
    np.random.seed(SEED)

    print("=" * 70)
    print("Single-channel SD+CS Simulation — All 20 Quiroga Datasets")
    print(f"Target CR: {CR_TARGET:.0%}")
    print("=" * 70)

    for group_info in DATASET_GROUPS:
        group_name = group_info['name']
        datasets = group_info['datasets']
        print(f"\n{'=' * 70}")
        print(f"Processing Group: {group_info['title']} ({len(datasets)} datasets)")
        print(f"{'=' * 70}")

        group_results = []
        for i, ds_name in enumerate(datasets):
            print(f"\n  [{i + 1}/{len(datasets)}] {ds_name} ...")
            try:
                result = get_or_process_dataset(ds_name)
                group_results.append(result)
                print(f"    -> SNDR: {result['sndr_mean']:.1f} dB, "
                      f"F1_recon: {result['recon_f1']:.3f}, "
                      f"Spikes: {result['n_spikes']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        if group_results:
            print(f"\n  Plotting group {group_info['title']}...")
            plot_group(group_results, group_info)

    print(f"\n{'=' * 70}")
    print("All groups completed!")
    print(f"{'=' * 70}")

    # Also generate summary figures
    plot_summary_figures()


# ==================== Summary Figures (bar chart + waveform) ====================

WAVEFORM_DS_NAMES = ['C_Easy1_noise02', 'C_Easy2_noise02',
                     'C_Difficult1_noise02', 'C_Difficult2_noise02']
WAVEFORM_LABELS = ['E1_N02', 'E2_N02', 'D1_N02', 'D2_N02']

# Font sizes for summary figures
SFS_TITLE = 10
SFS_SUBTITLE = 9
SFS_LABEL = 8
SFS_LEGEND = 8
SFS_TICK = 6
SFS_ANNOT = 9


def plot_summary_figures():
    """Generate summary bar chart and waveform comparison figures"""
    group_labels = []
    group_means_sndr = []
    group_means_cr = []
    group_means_f1 = []
    group_waveforms = []

    print("\n" + "=" * 60)
    print("Generating summary figures...")
    print("=" * 60)

    for g_idx, group_info in enumerate(DATASET_GROUPS):
        datasets = group_info['datasets']
        label = group_info['label']
        print(f"\nGroup {label} ({len(datasets)} datasets)...")

        sndr_means = []
        cr_means = []
        f1_vals = []

        for ds_name in datasets:
            get_wf = (ds_name == WAVEFORM_DS_NAMES[g_idx])
            result = get_or_process_dataset(ds_name, return_waveform=get_wf)
            sndr_means.append(result['sndr_mean'])
            cr_means.append(result['actual_cr_mean'])
            f1_vals.append(result['recon_f1'])
            print(f"  {ds_name}: SNDR={result['sndr_mean']:.1f}, "
                  f"CR={result['actual_cr_mean']:.4f}, F1={result['recon_f1']:.3f}")

            if get_wf:
                time_axis = np.arange(-WINDOW_SIZE, WINDOW_SIZE) / 50
                sndr_arr = result['sndr_array']
                segments = result['spike_segments_full']
                mean_sndr = np.mean(sndr_arr)
                pkpk = np.max(segments, axis=1) - np.min(segments, axis=1)
                sndr_dist = np.abs(sndr_arr - mean_sndr)
                score = sndr_dist / (sndr_dist.max() + 1e-10) - pkpk / (pkpk.max() + 1e-10)
                best_idx = np.argmin(score)
                diff = np.abs(segments[best_idx, :] - result['reconstructed_full'][best_idx, :])
                group_waveforms.append({
                    'time': time_axis,
                    'orig': segments[best_idx, :],
                    'recon': result['reconstructed_full'][best_idx, :],
                    'diff': diff,
                    'sndr': sndr_arr[best_idx],
                    'actual_cr': result['actual_cr'][best_idx],
                    'label': WAVEFORM_LABELS[g_idx],
                })

        group_labels.append(label)
        group_means_sndr.append(np.mean(sndr_means))
        group_means_cr.append(np.mean(cr_means))
        group_means_f1.append(np.mean(f1_vals))

    # =========================================
    # FIGURE 1: Summary box plot (3.5" wide, vertical)
    # =========================================
    print(f"\n{'=' * 60}")
    print("Plotting Figure 1: Summary box plots...")

    width = 3.5

    # Per-metric consistent colors
    C_SNDR = '#348ABD'  # blue
    C_CR   = '#E24A33'  # red
    C_F1   = '#988ED5'  # purple

    fig1, axes = plt.subplots(4, 1, figsize=(width, width * 1.5), dpi=300,
                               sharex=True)
    fig1.subplots_adjust(left=0.14, right=0.97, bottom=0.08, top=0.92, hspace=0.5)

    x = np.arange(len(group_labels))

    # Collect per-dataset values grouped by label for each metric
    def collect_per_dataset(metric_key):
        """Return list of lists: per-dataset values for each group"""
        grouped = {info['label']: [] for info in DATASET_GROUPS}
        for g_idx, group_info in enumerate(DATASET_GROUPS):
            for ds_name in group_info['datasets']:
                result = get_or_process_dataset(ds_name, return_waveform=False)
                grouped[group_info['label']].append(result[metric_key])
        return [grouped[lab] for lab in group_labels]

    def add_boxplot(ax, data, positions, color, ylabel, title, ylim=None, target_line=None,
                    legend_loc='upper right', legend_ncol=3):
        """Add box plot with mean marker and global max/min dashed lines"""
        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5),
                        medianprops=dict(color='white', linewidth=1.5),
                        whiskerprops=dict(color=color, linewidth=1.0),
                        capprops=dict(color=color, linewidth=1.0))
        # Mean markers
        means = [np.mean(d) for d in data]
        ax.scatter(positions, means, marker='D', s=20, color='white',
                   edgecolors='black', linewidths=0.5, zorder=5)
        # Target line (for CR)
        h_target = None
        if target_line is not None:
            h_target = ax.axhline(target_line, color='black', linestyle='--', linewidth=0.8,
                                  label=f'Target={target_line:.0f}%')
        # Global max/min from whisker ends (caps, across all groups)
        n_g = len(data)
        whisk_max = max(bp['caps'][i*2+1].get_ydata()[0] for i in range(n_g))
        whisk_min = min(bp['caps'][i*2].get_ydata()[0] for i in range(n_g))
        h_max = ax.axhline(whisk_max, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        h_min = ax.axhline(whisk_min, color=color, linestyle='-.', linewidth=1.2, alpha=0.8)
        # Mean text: x right by 0.2, same y as diamond
        for i, mu in enumerate(means):
            ax.text(positions[i] + 0.2, mu, f'{mu:.1f}', ha='left', va='center',
                    fontsize=5.5, color='black', fontweight='bold')
        # Legend
        from matplotlib.lines import Line2D
        handles = [h_max, h_min,
                   plt.scatter([], [], marker='D', s=15, color='white',
                              edgecolors='black', linewidths=0.5)]
        labels_list = [f'MAX={whisk_max:.1f}', f'MIN={whisk_min:.1f}', 'Mean']
        if target_line is not None and h_target is not None:
            handles.insert(0, h_target)
            labels_list.insert(0, f'Target={target_line:.0f}%')
        ax.legend(handles, labels_list, fontsize=5.5, loc=legend_loc,
                  framealpha=0.8, ncol=legend_ncol)
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=SFS_TICK)
        ax.set_ylabel(ylabel, fontsize=SFS_LABEL)
        # title already includes (a)(b)(c) prefix from caller
        ax.set_title(title, fontsize=SFS_SUBTITLE)
        ax.tick_params(labelsize=SFS_TICK)
        if ylim:
            ax.set_ylim(ylim)

    # (a) SNDR box plot — legend top-left, 1 row
    sndr_data = collect_per_dataset('sndr_mean')
    add_boxplot(axes[0], sndr_data, x, C_SNDR, 'SNDR (dB)', '(a) SNDR',
                legend_loc='upper left', legend_ncol=3)

    # (b) CR box plot — legend lower-right, 2 rows (ncol=2 with 4 items)
    cr_data = [[v * 100 for v in lst] for lst in collect_per_dataset('actual_cr_mean')]
    target_cr_pct = CR_TARGET * 100
    add_boxplot(axes[1], cr_data, x, C_CR, 'Actual CR (%)', '(b) Actual CR',
                ylim=(80, 90), target_line=target_cr_pct,
                legend_loc='lower right', legend_ncol=4)

    # (c) F1 (Recon) box plot — legend lower-right, 1 row
    f1_data = [[v * 100 for v in lst] for lst in collect_per_dataset('recon_f1')]
    add_boxplot(axes[2], f1_data, x, C_F1, 'F1 Score (%)', '(c) F1 Score (Recon)',
                legend_loc='lower right', legend_ncol=3, ylim=(30, 100))

    # (d) F1 (Original) box plot — legend lower-right, 1 row
    f1_orig_data = [[v * 100 for v in lst] for lst in collect_per_dataset('orig_f1')]
    add_boxplot(axes[3], f1_orig_data, x, C_F1, 'F1 Score (%)', '(d) F1 Score (Original)',
                legend_loc='lower right', legend_ncol=3, ylim=(30, 100))

    fig1.savefig(os.path.join(BASE_DIR, 'SD_CS_Summary_BarChart.png'),
                 dpi=300, bbox_inches='tight')
    print(f"  Saved: SD_CS_Summary_BarChart.png")

    # =========================================
    # FIGURE 1.5: Confusion matrices (3.5" wide)
    # =========================================
    print("Plotting Figure 1.5: Confusion matrices...")
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(width, width * 0.9), dpi=300)
    fig_cm.subplots_adjust(left=0.12, right=0.86, bottom=0.10, top=0.90,
                           wspace=0.35, hspace=0.40)

    for g_idx, group_info in enumerate(DATASET_GROUPS):
        row, col = g_idx // 2, g_idx % 2
        ax = axes_cm[row, col]
        # Use the same waveform dataset for this group
        ds_name = WAVEFORM_DS_NAMES[g_idx]
        result = get_or_process_dataset(ds_name, return_waveform=False)
        cm = result['cm_orig_vs_recon']  # 原始 vs 重建聚类
        n_clusters = cm.shape[0]
        # Normalize to percentages
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto',
                       origin='lower')
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        # Only bottom row has x-axis label
        if row == 0:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Recon cluster', fontsize=SFS_TICK)
        ax.set_ylabel('Original cluster', fontsize=SFS_TICK)
        ax.set_title(f'({chr(97 + g_idx)}) {group_info["label"]} (Orig vs Recon)',
                     fontsize=SFS_SUBTITLE)
        # Add text annotations (i,j now correspond to origin='lower')
        for i in range(n_clusters):
            for j in range(n_clusters):
                val = cm[i, j]
                pct = cm_pct[i, j]
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                        fontsize=5.5, color=color)
        ax.tick_params(labelsize=SFS_TICK)
    # Colorbar (cax 手动放置，确保在 figure 内)
    cbar_ax = fig_cm.add_axes([0.88, 0.10, 0.02, 0.80])
    cbar = fig_cm.colorbar(im, cax=cbar_ax)
    cbar.set_label('Percentage (%)', fontsize=SFS_TICK)
    cbar.ax.tick_params(labelsize=SFS_TICK)
    fig_cm.savefig(os.path.join(BASE_DIR, 'SD_CS_ConfusionMatrices.png'),
                   dpi=300)  # 固定 figsize 3.5in 宽，保证宽度 ≤ 3.5 inch
    print(f"  Saved: SD_CS_ConfusionMatrices.png")

    # =========================================
    # FIGURE 2: Waveform + error bars for N02 (3.5" wide)
    # =========================================
    print("Plotting Figure 2: Waveform comparisons with error bars...")

    fig2, axes2 = plt.subplots(2, 2, figsize=(width, width * 0.85))
    fig2.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.88,
                         wspace=0.30, hspace=0.40)

    all_handles = []
    all_labels = []

    for idx, wf in enumerate(group_waveforms):
        row, col = idx // 2, idx % 2
        ax = axes2[row, col]

        t = wf['time']
        orig = wf['orig']
        recon = wf['recon']
        diff = wf['diff']

        line1 = ax.plot(t, orig, 'b-', label='Original', lw=1.0)
        line2 = ax.plot(t, recon, 'r--', label='Reconstructed', lw=1.0)
        if idx == 0:
            all_handles.extend([line1[0], line2[0]])
            all_labels.extend(['Original', 'Reconstructed'])

        bar_w = (t[1] - t[0]) * 0.6
        y_range = max(orig.max(), recon.max()) - min(orig.min(), recon.min())
        bar_baseline = min(orig.min(), recon.min()) - 0.4 * y_range
        bar_max_height = 0.25 * y_range
        bar_heights = (diff / (diff.max() + 1e-10)) * bar_max_height if diff.max() > 0 else diff * 0

        bars = ax.bar(t, bar_heights, bottom=bar_baseline, width=bar_w,
                      color='gray', alpha=0.5, edgecolor='gray', linewidth=0.1)
        if idx == 0:
            all_handles.append(bars)
            all_labels.append('|Error|')

        ax.text(0.04, 0.96, f'SNDR={wf["sndr"]:.1f} dB\nActual CR={wf["actual_cr"]*100:.1f}%',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=SFS_ANNOT, color='green')

        y_bottom = bar_baseline - 0.05 * y_range
        y_top = max(orig.max(), recon.max()) * 1.25
        ax.set_ylim(y_bottom, y_top)

        ax.set_title(f'({chr(97 + idx)}) {wf["label"]}', fontsize=SFS_SUBTITLE)

        if row == 0:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (ms)', fontsize=SFS_LABEL)

        if col > 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Amplitude', fontsize=SFS_LABEL)

        ax.tick_params(labelsize=SFS_TICK)

    fig2.legend(all_handles, all_labels, loc='lower center',
                ncol=3, fontsize=SFS_LEGEND,
                bbox_to_anchor=(0.5, -0.06),
                frameon=True, edgecolor='gray')

    fig2.savefig(os.path.join(BASE_DIR, 'SD_CS_Waveform_Comparison.png'),
                 dpi=300, bbox_inches='tight')
    print(f"  Saved: SD_CS_Waveform_Comparison.png")

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary Table")
    print(f"{'=' * 60}")
    print(f"{'Group':<8} {'Mean SNDR':>10} {'Mean CR':>10} {'Mean F1':>10}")
    print(f"{'-' * 40}")
    for i, label in enumerate(group_labels):
        print(f"{label:<8} {group_means_sndr[i]:>8.2f}dB "
              f"{group_means_cr[i]:>9.4f} {group_means_f1[i]:>9.3f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
