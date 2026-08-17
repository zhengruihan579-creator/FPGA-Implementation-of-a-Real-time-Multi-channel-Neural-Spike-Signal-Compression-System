#!/usr/bin/env python
# coding: utf-8
"""
Analyze the effect of parameter p on actual compression ratio.
Tests CR_target from 70% to 90% (step 2%) and p from 0.35 to 0.45 (step 0.01).
Uses real spike segments from both single-channel and multi-channel datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
DATABASE_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))

WINDOW_SIZE = 50
SEGMENT_LENGTH = 2 * WINDOW_SIZE

# ====== Plot-Only Mode ======
PLOT_ONLY = False  # Set True after first run to just re-plot
# =============================
SAVE_RESULTS_FILE = os.path.join(SAVE_DIR, 'parameter_analysis_results.npz')

# ==================== MDC functions ====================
def Distance(input_signal, CR, p):
    _max = np.max(input_signal)
    _min = np.min(input_signal)
    N = len(input_signal)
    M = N * (1 - CR)
    ave_width = p * (_max - _min) / M
    sigma = (_max - _min - (M - 1) * ave_width) / M
    return sigma


def MDC_UMDC_Gen(input_signal, sigma):
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


# ==================== Load sample signals ====================
def load_sample_signals(n_spikes=5000):
    """Load multiple spike segments from single-channel and multi-channel"""
    import scipy.io as sio

    # Single-channel
    sc_path = os.path.join(DATABASE_DIR, 'C_Easy1_noise02.mat')
    mat = sio.loadmat(sc_path)
    raw_data = mat['data'].flatten()
    spike_times_raw = mat['spike_times'][0, 0].flatten().astype(int) - 1
    segs_sc = []
    for t in spike_times_raw:
        if t - WINDOW_SIZE >= 0 and t + WINDOW_SIZE < len(raw_data):
            segs_sc.append(raw_data[t - WINDOW_SIZE:t + WINDOW_SIZE].astype(np.float64))
            if len(segs_sc) >= n_spikes:
                break
    segs_sc = np.array(segs_sc)
    print(f"  SC: {len(segs_sc)} spikes, shape={segs_sc.shape}")

    # Multi-channel
    mc_path = os.path.join(SAVE_DIR, 'best_channel_data.npz')
    data = np.load(mc_path, allow_pickle=True)
    segs_mc = data['orig_best'][:n_spikes].astype(np.float64)
    print(f"  MC: {len(segs_mc)} spikes, shape={segs_mc.shape}")

    return segs_sc, segs_mc


# ==================== Main analysis ====================
def main():
    print("=" * 70)
    print("Parameter p Analysis: Effect on Actual Compression Ratio")
    print("=" * 70)

    cr_targets = np.arange(70, 92, 2)  # 70, 72, ..., 90
    p_values = np.arange(0.35, 0.46, 0.01)  # 0.35, 0.36, ..., 0.45

    if PLOT_ONLY and os.path.exists(SAVE_RESULTS_FILE):
        print("\nLoading saved results...")
        data = np.load(SAVE_RESULTS_FILE, allow_pickle=True)
        results = data['results'].item()
        cr_targets = data['cr_targets']
        p_values = data['p_values']
        print("  Loaded.")
    else:
        # Load signals
        print("\nLoading sample spike segments...")
        segs_sc, segs_mc = load_sample_signals(n_spikes=5000)
        N = segs_sc.shape[1]
        print(f"  Segment length: {N}")

        # Compute for both signal types
        results = {'SC': {}, 'MC': {}}
        for label, segs in [('SC', segs_sc), ('MC', segs_mc)]:
            print(f"\nProcessing {label} ({len(segs)} spikes)...")
            actual_cr_matrix = np.zeros((len(cr_targets), len(p_values)))
            n_seg = len(segs)
            for ci, cr in enumerate(cr_targets):
                cr_frac = cr / 100.0
                for pi, p in enumerate(p_values):
                    cr_sum = 0.0
                    for si in range(n_seg):
                        sigma = Distance(segs[si], cr_frac, p)
                        MDC = MDC_UMDC_Gen(segs[si], sigma)
                        M = MDC.shape[0]
                        cr_sum += (1 - M / N)
                    actual_cr_matrix[ci, pi] = cr_sum / n_seg * 100
                print(f"  CR_target={cr}% done", flush=True)
            results[label] = {
                'actual_cr': actual_cr_matrix,
            }
        # Save results
        np.savez_compressed(SAVE_RESULTS_FILE,
                            results=results, cr_targets=cr_targets,
                            p_values=p_values)
        print(f"\n  Saved results: {SAVE_RESULTS_FILE}")

    # ===== Plot =====
    print("\nPlotting results...")
    SFS_SUBTITLE = 9
    SFS_LABEL = 8
    SFS_TICK = 6
    width = 3.5
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 1.0), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.32, top=0.94,
                        wspace=0.28)

    # Distinct colors and markers for 11 lines
    line_colors = ['#E24A33', '#348ABD', '#988ED5', '#2E8B57', '#FF8C00',
                   '#8B4513', '#FF69B4', '#00CED1', '#A0522D', '#6A5ACD', '#DC143C']
    line_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']

    # Find CR_target=84% index (idx 7: 70,72,74,76,78,80,82,84,86,88,90)
    cr84_idx = list(cr_targets).index(84)
    p40_idx = list(np.round(p_values, 2)).index(0.40)

    for idx, (label, title) in enumerate([('SC', 'Single-Channel'),
                                           ('MC', 'Multi-Channel')]):
        ax = axes[idx]
        res = results[label]
        for ci, cr in enumerate(cr_targets):
            ax.plot(p_values, res['actual_cr'][ci, :],
                    marker=line_markers[ci % len(line_markers)],
                    markersize=3, linewidth=0.8,
                    color=line_colors[ci % len(line_colors)],
                    label=f'{cr}%')
        # Highlight CR_target=84% line
        cr84_vals = res['actual_cr'][cr84_idx, :]
        ax.plot(p_values, cr84_vals, linewidth=1.8,
                color=line_colors[cr84_idx], alpha=0.6)

        # Mark p=0.40 on 84% line
        val_at_p40 = cr84_vals[p40_idx]
        ax.plot(0.40, val_at_p40, 'o', markersize=6,
                color=line_colors[cr84_idx], markeredgecolor='black',
                markeredgewidth=0.5, zorder=5)
        ax.annotate(f'p=0.4\n→{val_at_p40:.1f}%',
                    xy=(0.40, val_at_p40), xytext=(0.41, val_at_p40 - 1),
                    fontsize=5.5, color='black', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=0.5))

        # Find where actual CR=84% on the 84% target line
        above = np.where(cr84_vals >= 84.0)[0]
        if len(above) > 0:
            p_at_84 = p_values[above[-1]]  # last p where actual CR >= 84
            ax.plot(p_at_84, 84.0, 's', markersize=5,
                    color=line_colors[cr84_idx], markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)
            ax.annotate(f'CR=84%\n→p={p_at_84:.2f}',
                        xy=(p_at_84, 84.0), xytext=(p_at_84 - 0.04, 83.0),
                        fontsize=5.5, color='black', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', lw=0.5))

        ax.set_xlabel('p', fontsize=SFS_LABEL)
        ax.set_ylabel('Actual CR (%)', fontsize=SFS_LABEL)
        ax.set_title(title, fontsize=SFS_SUBTITLE)
        ax.set_ylim(70, 95)
        ax.set_xlim(0.34, 0.47)
        ax.tick_params(labelsize=SFS_TICK)
        ax.grid(True, alpha=0.3)

    # Unified legend at bottom of figure, two rows
    handles = []
    labels_list = []
    for ci, cr in enumerate(cr_targets):
        handles.append(plt.Line2D([0], [0],
                       marker=line_markers[ci % len(line_markers)],
                       color=line_colors[ci % len(line_colors)],
                       linestyle='-', linewidth=0.8, markersize=4))
        labels_list.append(f'{cr}%')
    fig.legend(handles, labels_list, title='Target CR',
               fontsize=5.5, title_fontsize=6.5,
               loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.savefig(os.path.join(BASE_DIR, 'SD_CS_Parameter_Analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: SD_CS_Parameter_Analysis.png")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("Summary: Actual CR for each (target_CR, p) combination")
    print(f"{'=' * 80}")
    for label in ['SC', 'MC']:
        print(f"\n--- {label} ---")
        header = f"{'CR/p':>6s}"
        for p in p_values:
            header += f"  {p:.2f}"
        print(header)
        for ci, cr in enumerate(cr_targets):
            row = f"{cr:>4d}%  "
            for pi in range(len(p_values)):
                row += f"  {results[label]['actual_cr'][ci, pi]:5.1f}"
            print(row)

    print(f"\n{'=' * 80}")
    print("All done!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
