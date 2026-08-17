#!/usr/bin/env python
# coding: utf-8
"""
Build a continuous .bin recording from per-channel reconstructed spike segments,
suitable for loading in Kilosort GUI for clustering.
"""
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
TRANSFER_DIR = os.path.join(BASE_DIR, 'transfer')
WINDOW_SIZE = 50
N_SPIKES_MAX = None  # limit for testing; set to None for all

def main():
    print("=" * 70)
    print("Build reconstructed continuous recording (.bin)")
    print("=" * 70)

    # 1. Load original raw data (int16)
    print("\nLoading original raw data...")
    raw_path = os.path.join(TRANSFER_DIR, 'dataSample_BPF_300_5000.npy')
    raw_data = np.load(raw_path)  # (374, 1800000), int16
    n_chan_total, n_samples = raw_data.shape
    print(f"  Original: {n_chan_total} channels, {n_samples} samples, dtype={raw_data.dtype}")

    # 2. Load saved results
    print("\nLoading saved results...")
    res_path = os.path.join(SAVE_DIR, 'neuropixel_results.npz')
    rec_path = os.path.join(SAVE_DIR, 'neuropixel_recon_segments.npz')

    if not os.path.exists(res_path) or not os.path.exists(rec_path):
        print("  ERROR: Saved results not found. Run SD_CS_MultiChannel_Sim.py first.")
        sys.exit(1)

    data = np.load(res_path, allow_pickle=True)
    spike_times = data['spike_times']
    spike_clusters = data['spike_clusters']

    rec_data = np.load(rec_path, allow_pickle=True)
    reconstructed_spikes_raw = rec_data['reconstructed_spikes_raw']

    pc_feature_ind = data['pc_feature_ind'] if 'pc_feature_ind' in data else \
        np.load(os.path.join(TRANSFER_DIR, 'kilosort4_full', 'pc_feature_ind.npy'))

    n_spikes = len(reconstructed_spikes_raw)
    if N_SPIKES_MAX and N_SPIKES_MAX < n_spikes:
        n_spikes = N_SPIKES_MAX
    print(f"  Spikes to reconstruct: {n_spikes} / {len(reconstructed_spikes_raw)}")

    # 3. Build reconstructed recording (copy of original, then overwrite spike regions)
    print("\nBuilding reconstructed continuous recording...")
    recon_recording = raw_data.copy()  # int16

    n_placed = 0
    for i in range(n_spikes):
        t = int(spike_times[i])
        c = int(spike_clusters[i])
        start = t - WINDOW_SIZE
        end = t + WINDOW_SIZE
        if start < 0 or end >= n_samples:
            continue

        # Get the reconstructed segment for this spike
        recon_seg = np.array(reconstructed_spikes_raw[i], dtype=np.float32)  # (n_chan, 100)

        # Get the channels for this spike
        if c < pc_feature_ind.shape[0]:
            channels = pc_feature_ind[c]
        else:
            channels = np.arange(min(16, n_chan_total))

        n_chan_seg = recon_seg.shape[0]
        use_chan = min(n_chan_seg, len(channels))

        # Place reconstructed segment onto the recording
        for ch_idx in range(use_chan):
            chan_id = channels[ch_idx]
            recon_ch = recon_seg[ch_idx, :]

            # Clip to int16 range and cast
            recon_ch_int16 = np.clip(np.round(recon_ch), -32768, 32767).astype(np.int16)
            recon_recording[chan_id, start:end] = recon_ch_int16

        n_placed += 1
        if (i + 1) % 1000 == 0:
            print(f"    ... {i + 1}/{n_spikes} spikes placed")

    print(f"\n  Placed {n_placed} spikes into recording")

    # 4. Save as .bin file
    bin_path = os.path.join(SAVE_DIR, 'recon_recording.bin')
    recon_recording.tofile(bin_path)
    file_size_gb = os.path.getsize(bin_path) / 1e9
    print(f"\n  Saved: {bin_path}")
    print(f"  Size: {file_size_gb:.2f} GB")
    print(f"  Shape: {recon_recording.shape} (channels × samples)")
    print(f"  dtype: {recon_recording.dtype}")
    print(f"  Range: [{recon_recording.min()}, {recon_recording.max()}]")

    # 5. Also save channel map for Kilosort GUI
    chan_map_path = os.path.join(SAVE_DIR, 'channel_map_for_gui.npy')
    channel_map = np.load(os.path.join(TRANSFER_DIR, 'kilosort4_full', 'channel_map.npy'))
    np.save(chan_map_path, channel_map)
    print(f"\n  Channel map saved: {chan_map_path}")
    print(f"  (Load in Kilosort GUI with this channel map)")

    print(f"\n{'=' * 70}")
    print("Done! Use Kilosort GUI to open:")
    print(f"  {bin_path}")
    print(f"  Channel map: {chan_map_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
