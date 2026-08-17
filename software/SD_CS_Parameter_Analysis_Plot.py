#!/usr/bin/env python
"""Plot parameter analysis from pre-computed data (no MDC recomputation)."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cr_targets = np.arange(70, 92, 2)
p_values = np.arange(0.35, 0.46, 0.01)

# Data from terminal output (5000 spike average)
sc_data = [
    [77.4, 77.2, 77.0, 76.8, 76.6, 76.4, 76.1, 75.9, 75.6, 75.4, 75.1, 74.9],
    [78.4, 78.3, 78.0, 77.8, 77.6, 77.4, 77.1, 76.9, 76.7, 76.5, 76.2, 75.9],
    [79.5, 79.3, 79.1, 78.9, 78.7, 78.4, 78.2, 78.0, 77.8, 77.5, 77.3, 77.0],
    [80.5, 80.3, 80.2, 80.0, 79.7, 79.5, 79.3, 79.1, 78.9, 78.7, 78.5, 78.2],
    [81.6, 81.4, 81.3, 81.1, 80.9, 80.7, 80.5, 80.3, 80.1, 79.9, 79.7, 79.4],
    [82.8, 82.6, 82.5, 82.3, 82.1, 81.9, 81.7, 81.5, 81.3, 81.2, 80.9, 80.7],
    [84.0, 83.9, 83.7, 83.6, 83.4, 83.2, 83.1, 82.9, 82.7, 82.5, 82.3, 82.1],
    [85.3, 85.2, 85.0, 84.9, 84.7, 84.6, 84.4, 84.2, 84.1, 83.9, 83.7, 83.5],
    [86.7, 86.6, 86.5, 86.3, 86.2, 86.0, 85.9, 85.7, 85.6, 85.4, 85.2, 85.1],
    [88.1, 88.0, 87.9, 87.8, 87.6, 87.5, 87.4, 87.3, 87.1, 87.0, 86.9, 86.7],
    [89.7, 89.6, 89.5, 89.4, 89.3, 89.2, 89.1, 88.9, 88.8, 88.7, 88.6, 88.4],
]
mc_data = [
    [77.2, 76.9, 76.8, 76.5, 76.2, 76.1, 75.9, 75.7, 75.5, 75.2, 74.9, 74.6],
    [78.2, 78.0, 77.8, 77.5, 77.4, 77.1, 76.9, 76.6, 76.4, 76.2, 76.0, 75.8],
    [79.3, 79.1, 78.9, 78.8, 78.5, 78.2, 78.0, 77.8, 77.5, 77.2, 77.1, 76.8],
    [80.5, 80.3, 80.1, 80.0, 79.7, 79.5, 79.2, 78.9, 78.8, 78.5, 78.2, 78.0],
    [81.6, 81.5, 81.2, 81.1, 80.9, 80.7, 80.5, 80.3, 80.1, 79.9, 79.5, 79.3],
    [82.9, 82.6, 82.5, 82.3, 82.1, 81.9, 81.8, 81.6, 81.4, 81.2, 81.0, 80.7],
    [84.1, 83.9, 83.7, 83.5, 83.4, 83.3, 83.0, 82.9, 82.7, 82.5, 82.3, 82.1],
    [85.3, 85.2, 85.1, 84.9, 84.8, 84.6, 84.5, 84.3, 84.1, 83.9, 83.7, 83.5],
    [86.7, 86.5, 86.5, 86.3, 86.2, 86.0, 85.9, 85.8, 85.6, 85.4, 85.2, 85.1],
    [88.2, 88.1, 88.0, 87.9, 87.7, 87.6, 87.4, 87.3, 87.2, 87.0, 86.9, 86.7],
    [89.8, 89.7, 89.7, 89.5, 89.4, 89.3, 89.2, 89.0, 88.9, 88.8, 88.7, 88.5],
]

sc_arr = np.array(sc_data)
mc_arr = np.array(mc_data)

width = 3.5
fig, axes = plt.subplots(1, 2, figsize=(width, width * 1.0), dpi=300)
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.28, top=0.94, wspace=0.28)

line_colors = ['#E24A33','#348ABD','#988ED5','#2E8B57','#FF8C00',
               '#8B4513','#FF69B4','#00CED1','#A0522D','#6A5ACD','#DC143C']
line_markers = ['o','s','D','^','v','<','>','p','*','h','x']

cr84_idx = list(cr_targets).index(84)
p40_idx = list(np.round(p_values,2)).index(0.40)

for idx, (data_arr, title) in enumerate([(sc_arr, 'Single-Channel'), (mc_arr, 'Multi-Channel')]):
    ax = axes[idx]
    for ci in range(len(cr_targets)):
        ax.plot(p_values, data_arr[ci], marker=line_markers[ci], markersize=3,
                linewidth=0.8, color=line_colors[ci], label=f'{cr_targets[ci]}%')

    # Highlight CR_target=84% line
    cr84_vals = data_arr[cr84_idx]
    ax.plot(p_values, cr84_vals, linewidth=1.8, color=line_colors[cr84_idx], alpha=0.6)

    # Horizontal red dashed line at CR=84% (thick)
    ax.axhline(84.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Find where actual CR=84% on 84% target line
    above = np.where(cr84_vals >= 84.0)[0]
    if len(above) > 0:
        p_at_84 = p_values[above[-1]]
        # Vertical red dashed line from bottom to intersection (stops at curve)
        ax.plot([p_at_84, p_at_84], [70, 84.0], 'r--', linewidth=1.5, alpha=0.8)
        # Hollow circle marker at intersection
        ax.plot(p_at_84, 84.0, 'o', markersize=7, markerfacecolor='none',
                markeredgecolor='red', markeredgewidth=1.2, zorder=5)
        # Label at y=0.1 (axes), right of line by 0.02, centered
        label_y = 70 + 0.1 * (95 - 70)
        ax.text(p_at_84 + 0.02, label_y,
                f'Actual CR = 84.0%\np = {p_at_84:.2f}',
                ha='center', va='center', fontsize=5.5, fontweight='bold', color='red')

    # Vertical red dashed line at p=0.4 (from bottom to intersection, stops at curve)
    val_p40 = cr84_vals[p40_idx]
    ax.plot([0.40, 0.40], [70, val_p40], 'r--', linewidth=1.5, alpha=0.8)
    # Hollow circle marker at intersection
    ax.plot(0.40, val_p40, 'o', markersize=7, markerfacecolor='none',
            markeredgecolor='red', markeredgewidth=1.2, zorder=5)
    # Label at y=0.1 (axes), left of line by 0.03, centered
    ax.text(0.40 - 0.03, label_y,
            f'Actual CR = {val_p40:.1f}%\np = 0.40',
            ha='center', va='center', fontsize=5.5, fontweight='bold', color='red')

    # Arrow from horizontal line at 45° to label at y=0.8, x=0.34
    label_y = 70 + 0.8 * (95 - 70)
    ax.annotate('Target CR = 84%', xy=(0.37, 84.0), xytext=(0.40, label_y),
                ha='center', va='bottom', fontsize=5.5, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0))

    ax.set_xlabel('p', fontsize=8)
    ax.set_ylabel('Actual CR (%)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(70, 95)
    ax.set_xlim(0.34, 0.47)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

# Legend
handles = [plt.Line2D([0],[0], marker=line_markers[ci], color=line_colors[ci],
                      linestyle='-', linewidth=0.8, markersize=4)
           for ci in range(len(cr_targets))]
fig.legend(handles, [f'{cr}%' for cr in cr_targets], title='Target CR',
           fontsize=5.5, title_fontsize=6.5, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.06), frameon=True)

fig.savefig(os.path.join(BASE_DIR, 'SD_CS_Parameter_Analysis.png'),
            dpi=300, bbox_inches='tight')
print("Saved: SD_CS_Parameter_Analysis.png")
