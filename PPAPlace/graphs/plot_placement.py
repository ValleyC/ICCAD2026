#!/usr/bin/env python3
"""Visualize chip placements from DEF files — publication quality.

Renders macros as labeled rectangles, std cells as a density heatmap.
Produces a side-by-side comparison figure suitable for top-venue papers.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import re
import json
import os

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   6.5,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.5,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# Macro type → size in microns (from LEF)
MACRO_SIZES = {
    'fakeram45_512x64': (152.570, 113.400),
    'fakeram45_64x15':  (11.210,  58.800),
    'fakeram45_64x96':  (54.530,  89.400),
}

# Colors
C_MACRO_FILL  = '#EE6677'   # Tol red
C_MACRO_EDGE  = '#882255'   # dark magenta
C_BG          = '#FFFFFF'
C_DIE_EDGE    = '#333333'


def parse_def(def_path):
    """Parse a DEF file and return die area, macros, and std cell positions."""
    die_area = None
    macros = []       # [(name, cell_type, x_um, y_um, orient)]
    stdcells = []     # [(x_um, y_um)]
    units = 1000      # DEF distance units per micron

    with open(def_path, 'r') as f:
        in_components = False
        prev_line = ''
        for line in f:
            line = line.strip()

            # Units
            m = re.match(r'UNITS DISTANCE MICRONS (\d+)', line)
            if m:
                units = int(m.group(1))
                continue

            # Die area
            m = re.match(r'DIEAREA \( (\d+) (\d+) \) \( (\d+) (\d+) \)', line)
            if m:
                die_area = (
                    int(m.group(1)) / units,
                    int(m.group(2)) / units,
                    int(m.group(3)) / units,
                    int(m.group(4)) / units,
                )
                continue

            if line.startswith('COMPONENTS'):
                in_components = True
                continue
            if line.startswith('END COMPONENTS'):
                in_components = False
                continue

            if not in_components:
                prev_line = line
                continue

            # Component definition line: "- inst_name cell_type"
            if line.startswith('- '):
                parts = line.split()
                if len(parts) >= 3:
                    inst_name = parts[1]
                    cell_type = parts[2]
                    prev_line = f'{inst_name}|{cell_type}'
                continue

            # Placement line: "+ FIXED ( x y ) orient ;" or "+ PLACED ( x y ) orient ;"
            m_placed = re.match(r'\+ (FIXED|PLACED) \( (\d+) (\d+) \) (\w+)', line)
            if m_placed and '|' in prev_line:
                status = m_placed.group(1)
                x = int(m_placed.group(2)) / units
                y = int(m_placed.group(3)) / units
                orient = m_placed.group(4)
                inst_name, cell_type = prev_line.split('|', 1)

                if status == 'FIXED':
                    macros.append((inst_name, cell_type, x, y, orient))
                else:
                    stdcells.append((x, y))

    return die_area, macros, np.array(stdcells), units


def plot_placement(ax, def_path, title=None, subtitle=None,
                   density_bins=128, show_macro_labels=False):
    """Plot a single placement on the given axes."""
    die_area, macros, stdcells, units = parse_def(def_path)
    xl, yl, xh, yh = die_area
    w_die = xh - xl
    h_die = yh - yl

    # White background
    ax.set_facecolor(C_BG)

    # Std cell density heatmap
    if len(stdcells) > 0:
        heatmap, xedges, yedges = np.histogram2d(
            stdcells[:, 0], stdcells[:, 1],
            bins=density_bins,
            range=[[xl, xh], [yl, yh]]
        )
        # Normalize and apply colormap
        heatmap = heatmap.T  # imshow expects (row, col)
        ax.imshow(heatmap, origin='lower', extent=[xl, xh, yl, yh],
                  cmap='Blues', alpha=0.6, aspect='equal',
                  interpolation='bilinear', zorder=1)

    # Draw macros
    patches = []
    for inst_name, cell_type, mx, my, orient in macros:
        if cell_type in MACRO_SIZES:
            mw, mh = MACRO_SIZES[cell_type]
        else:
            mw, mh = 50, 50  # fallback
        rect = mpatches.FancyBboxPatch(
            (mx, my), mw, mh,
            boxstyle='round,pad=0.5',
            facecolor=C_MACRO_FILL, edgecolor=C_MACRO_EDGE,
            linewidth=0.6, alpha=0.85, zorder=3
        )
        ax.add_patch(rect)

        if show_macro_labels:
            # Short label: last component of hierarchical name
            short = inst_name.split('/')[-1]
            if len(short) > 10:
                short = short[:8] + '..'
            ax.text(mx + mw/2, my + mh/2, short,
                    ha='center', va='center', fontsize=3,
                    color='white', fontweight='bold', zorder=4)

    # Die outline
    die_rect = mpatches.Rectangle(
        (xl, yl), w_die, h_die,
        fill=False, edgecolor=C_DIE_EDGE, linewidth=0.8, zorder=5
    )
    ax.add_patch(die_rect)

    ax.set_xlim(xl - w_die*0.02, xh + w_die*0.02)
    ax.set_ylim(yl - h_die*0.02, yh + h_die*0.02)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=8, pad=4)
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=6, color='#555555')


def main():
    """Generate placement comparison figure for bp_be."""
    data_dir = 'E:/ChipSAT/dreamplace_data/bp_be'
    grt_dir = os.path.join(data_dir, 'grt_jsons')

    # Find best and worst by WNS
    import glob
    results = []
    for f in glob.glob(os.path.join(grt_dir, '*.json')):
        with open(f) as fh:
            d = json.load(fh)
        wns = d['globalroute__timing__setup__ws']
        tns = d['globalroute__timing__setup__tns']
        power = d['globalroute__power__total']
        name = os.path.basename(f).replace('dp_', '').replace('_grt.json', '')
        results.append((name, wns, tns, power))

    results.sort(key=lambda x: x[1])  # ascending WNS (most negative = worst)

    worst = results[0]    # worst WNS
    median_idx = len(results) // 2
    median = results[median_idx]
    best = results[-1]    # best WNS

    picks = [
        (worst, '(a) Poor config.'),
        (median, '(b) Median config.'),
        (best, '(c) Best config. (PPAPlace)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    for ax, ((name, wns, tns, power), title) in zip(axes, picks):
        def_path = os.path.join(data_dir, f'{name}.def')
        if not os.path.exists(def_path):
            print(f'  [skip] {def_path} not found')
            continue
        subtitle = f'WNS={wns:.0f}ps  TNS={tns:.0f}ns  Pwr={power:.3f}W'
        plot_placement(ax, def_path, title=title, subtitle=subtitle,
                       show_macro_labels=True)
        print(f'  Plotted {name}')

    # Legend
    macro_patch = mpatches.Patch(facecolor=C_MACRO_FILL, edgecolor=C_MACRO_EDGE,
                                 linewidth=0.6, label='Macros')
    from matplotlib.lines import Line2D
    cell_patch = Line2D([0], [0], marker='s', color='w',
                        markerfacecolor='#4477AA', markersize=6,
                        label='Std cell density')
    fig.legend(handles=[macro_patch, cell_patch],
               loc='lower center', ncol=2, frameon=True,
               framealpha=0.95, edgecolor='#cccccc',
               fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig('placement_comparison.pdf')
    plt.close(fig)
    print('  placement_comparison.pdf')


if __name__ == '__main__':
    print('Generating placement visualization ...')
    main()
    print('Done.')
