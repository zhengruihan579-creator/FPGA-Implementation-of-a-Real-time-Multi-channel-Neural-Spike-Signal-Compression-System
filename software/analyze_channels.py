#!/usr/bin/env python
# coding: utf-8
"""
通道分析：判断哪些通道主要承载噪声、哪些有有效spike
以及每个spike的16个关联通道中选择信号最明显的通道。

输出：
1. per_spike_channel_analysis.npz — 每个spike的最佳通道信息
2. 控制台统计信息
"""
import numpy as np
import os
import time
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
TRANSFER_DIR = os.path.join(BASE_DIR, 'transfer')

# 加载结果摘要
results_path = os.path.join(SAVE_DIR, 'neuropixel_results.npz')
print("Loading results...")
data = np.load(results_path, allow_pickle=True)
pc_feature_ind = data['pc_feature_ind']  # (364, 16)
spike_clusters = data['spike_clusters']  # (169666,)
gt_labels = data['gt_labels']
print(f"  {len(spike_clusters)} spikes, {len(np.unique(gt_labels))} GT clusters")
print(f"  pc_feature_ind: {pc_feature_ind.shape}")

# 加载原始多通道段（大文件，需要时间）
raw_path = os.path.join(SAVE_DIR, 'neuropixel_raw_segments.npz')
print(f"\nLoading raw segments ({os.path.getsize(raw_path)/1e9:.2f}GB)...")
t0 = time.time()
raw_data = np.load(raw_path, allow_pickle=True)
spike_segments_raw = raw_data['spike_segments_raw']
print(f"  Loaded {len(spike_segments_raw)} segments in {time.time()-t0:.0f}s")

# ============================================================
# 分析1: 每个spike找出最佳通道（信号最强的）
# ============================================================
print("\n=== 分析1: 每个spike的最佳通道 ===")
NOISE_THRESH = 1.0  # 信号范数阈值

best_ch_info = []  # (spike_idx, ch_idx_in_group, physical_ch, signal_norm, sndr_approx)
noise_ch_counts = Counter()  # 每个物理通道被跳过的次数
signal_ch_counts = Counter()  # 每个物理通道有有效信号的次数

n_spikes = len(spike_segments_raw)
BATCH = 10000

for i in range(n_spikes):
    raw_seg = np.array(spike_segments_raw[i], dtype=np.float64)  # (n_chan, 100)
    n_chan = raw_seg.shape[0]
    
    # 获取该spike所属cluster对应的物理通道
    cluster_id = spike_clusters[i]
    if cluster_id < pc_feature_ind.shape[0]:
        phys_channels = pc_feature_ind[cluster_id][:n_chan]
    else:
        phys_channels = np.arange(n_chan)
    
    # 计算每个通道的信号范数
    signal_norms = np.array([np.linalg.norm(raw_seg[ch, :]) for ch in range(n_chan)])
    
    # 找到信号最强的通道
    best_local_idx = np.argmax(signal_norms)
    best_norm = signal_norms[best_local_idx]
    best_phys_ch = phys_channels[best_local_idx]
    
    best_ch_info.append({
        'spike_idx': i,
        'local_idx': best_local_idx,      # 在16个通道组中的位置(0-15)
        'physical_ch': best_phys_ch,      # 物理通道号(0-372)
        'signal_norm': best_norm,
        'has_signal': best_norm >= NOISE_THRESH,
    })
    
    # 统计每个物理通道的信号情况
    for ch in range(n_chan):
        pch = phys_channels[ch]
        if signal_norms[ch] >= NOISE_THRESH:
            signal_ch_counts[pch] += 1
        else:
            noise_ch_counts[pch] += 1
    
    if (i + 1) % BATCH == 0:
        print(f"  ... {i+1}/{n_spikes} ({time.time()-t0:.0f}s)")

print(f"  Done in {time.time()-t0:.0f}s")

# ============================================================
# 分析2: 全局通道使用统计
# ============================================================
print("\n=== 分析2: 全局通道使用统计 ===")
print(f"总通道数: 374 (0-373), 活跃通道: {len(signal_ch_counts)}")

# 每个通道被分配到多少spike
total_per_ch = Counter()
for i in range(n_spikes):
    cluster_id = spike_clusters[i]
    if cluster_id < pc_feature_ind.shape[0]:
        chs = pc_feature_ind[cluster_id]
    else:
        chs = np.arange(min(16, 374))
    for ch in chs:
        total_per_ch[ch] += 1

# 计算每个通道的"信号率" = 有信号次数 / 总分配次数
ch_signal_ratio = {}
for ch in range(374):
    total = total_per_ch.get(ch, 0)
    if total > 0:
        signal = signal_ch_counts.get(ch, 0)
        ch_signal_ratio[ch] = signal / total
    else:
        ch_signal_ratio[ch] = 0.0

# 按信号率排序
sorted_ch = sorted(ch_signal_ratio.items(), key=lambda x: x[1])

print("\n--- 信号率最低的20个通道（主要噪声） ---")
for ch, ratio in sorted_ch[:20]:
    total = total_per_ch.get(ch, 0)
    signal = signal_ch_counts.get(ch, 0)
    noise = noise_ch_counts.get(ch, 0)
    print(f"  Ch {ch:3d}: signal_ratio={ratio:.3f} (signal={signal}, noise={noise}, total={total})")

print("\n--- 信号率最高的20个通道（主要信号） ---")
for ch, ratio in sorted_ch[-20:]:
    total = total_per_ch.get(ch, 0)
    signal = signal_ch_counts.get(ch, 0)
    noise = noise_ch_counts.get(ch, 0)
    print(f"  Ch {ch:3d}: signal_ratio={ratio:.3f} (signal={signal}, noise={noise}, total={total})")

# ============================================================
# 分析3: 最佳通道的分布
# ============================================================
print("\n=== 分析3: 最佳通道（信号最强）的分布 ===")

# 最佳通道在16通道组中的位置分布
local_positions = [info['local_idx'] for info in best_ch_info]
pos_counts = Counter(local_positions)
print("\n最佳通道在16通道组中的位置分布:")
for pos in range(16):
    cnt = pos_counts.get(pos, 0)
    bar = '#' * (cnt // 2000)
    print(f"  Position {pos:2d}: {cnt:6d} spikes ({cnt/n_spikes*100:.1f}%) {bar}")

# 最佳通道的物理通道分布
best_phys = [info['physical_ch'] for info in best_ch_info]
best_phys_counts = Counter(best_phys)
print("\n最佳通道（物理通道号）Top 20:")
for ch, cnt in best_phys_counts.most_common(20):
    bar = '#' * (cnt // 2000)
    print(f"  Ch {ch:3d}: {cnt:6d} spikes ({cnt/n_spikes*100:.1f}%) {bar}")

# 无信号spike统计
no_signal = sum(1 for info in best_ch_info if not info['has_signal'])
print(f"\n无有效信号的spike: {no_signal} / {n_spikes} ({no_signal/n_spikes*100:.1f}%)")

# ============================================================
# 保存分析结果
# ============================================================
print("\n保存分析结果...")
save_analysis = {
    'best_physical_ch': np.array([info['physical_ch'] for info in best_ch_info], dtype=np.uint16),
    'best_local_idx': np.array([info['local_idx'] for info in best_ch_info], dtype=np.uint8),
    'best_signal_norm': np.array([info['signal_norm'] for info in best_ch_info], dtype=np.float32),
    'has_signal': np.array([info['has_signal'] for info in best_ch_info], dtype=bool),
    'ch_signal_ratio': np.array([ch_signal_ratio[ch] for ch in range(374)], dtype=np.float32),
    'ch_signal_count': np.array([signal_ch_counts.get(ch, 0) for ch in range(374)], dtype=np.int32),
    'ch_noise_count': np.array([noise_ch_counts.get(ch, 0) for ch in range(374)], dtype=np.int32),
    'ch_total_count': np.array([total_per_ch.get(ch, 0) for ch in range(374)], dtype=np.int32),
}
np.savez_compressed(os.path.join(SAVE_DIR, 'channel_analysis.npz'), **save_analysis)
print("  Saved to channel_analysis.npz")
print("\n=== 分析完成 ===")
