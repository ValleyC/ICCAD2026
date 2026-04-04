#!/usr/bin/env python3
"""Visualize chip placements from DEF files — publication quality.

Renders macros as colored rectangles, std cells as a density heatmap.
Produces a side-by-side comparison of placement methods.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
import os

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8,
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
    # ariane133
    'fakeram45_256x16':  (77.710, 40.600),
    # bp_fe / bp_be
    'fakeram45_512x64': (152.570, 113.400),
    'fakeram45_64x15':  (11.210,  58.800),
    'fakeram45_64x96':  (54.530,  89.400),
    # swerv_wrapper
    'fakeram45_2048x39': (206.910, 219.800),
    'fakeram45_256x34':  (98.420,  65.800),
    'fakeram45_64x21':   (15.770,  60.200),
    # black_parrot
    'fakeram45_256x95':  (77.710, 40.600),   # placeholder
    'fakeram45_64x7':    (11.210, 30.800),    # placeholder
}

# Cell types that are macros (prefixes)
MACRO_PREFIXES = ('fakeram45_', 'RAM16X1D', 'memMod_', 'spram_', 'memory_block_')

# Per-type colors — Nature Reviews / ggsci palette (fill, edge)
_NBLUE   = ('#3C5488', '#2A3C66')   # dark slate blue
_NRED    = ('#E64B35', '#B33A29')   # vermillion
_NTEAL   = ('#00A087', '#007A66')   # emerald teal
_NPURPLE = ('#7E6148', '#5E4836')   # warm brown
MACRO_TYPE_COLORS = {
    'fakeram45_2048x39': _NBLUE,
    'fakeram45_256x34':  _NRED,
    'fakeram45_64x21':   _NTEAL,
    'fakeram45_512x64':  _NBLUE,
    'fakeram45_256x16':  _NBLUE,
    'fakeram45_64x15':   _NTEAL,
    'fakeram45_64x96':   _NRED,
    'fakeram45_256x95':  _NPURPLE,
    'fakeram45_64x7':    _NTEAL,
}
MACRO_DEFAULT_COLORS = ('#8C8C8C', '#5A5A5A')

C_BG          = '#FAFAFA'
C_DIE_EDGE    = '#1A1A1A'


def is_macro_type(cell_type):
    """Check if a cell type is a macro (not a standard cell)."""
    return any(cell_type.startswith(p) for p in MACRO_PREFIXES)


def parse_def(def_path):
    """Parse a DEF file and return die area, macros, and std cell positions.

    Handles both single-line and multi-line component formats:
      Single: - inst_name cell_type + FIXED ( x y ) orient ;
      Multi:  - inst_name cell_type
              + PLACED ( x y ) orient ;
    """
    die_area = None
    macros = []       # [(name, cell_type, x_um, y_um, orient)]
    stdcells = []     # [(x_um, y_um)]
    units = 1000      # DEF distance units per micron

    # Regex for placement info anywhere in a line
    place_re = re.compile(r'\+\s+(FIXED|PLACED)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)')

    with open(def_path, 'r') as f:
        in_components = False
        pending_inst = None   # (inst_name, cell_type) waiting for placement
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
                continue

            # Component definition: "- inst_name cell_type [+ FIXED/PLACED ...]"
            if line.startswith('- '):
                parts = line.split()
                if len(parts) >= 3:
                    inst_name = parts[1]
                    cell_type = parts[2]
                    # Check if placement info is on the same line
                    m_place = place_re.search(line)
                    if m_place:
                        x = int(m_place.group(2)) / units
                        y = int(m_place.group(3)) / units
                        orient = m_place.group(4)
                        if is_macro_type(cell_type):
                            macros.append((inst_name, cell_type, x, y, orient))
                        else:
                            stdcells.append((x, y))
                        pending_inst = None
                    else:
                        pending_inst = (inst_name, cell_type)
                continue

            # Continuation line with placement info
            if pending_inst is not None:
                m_place = place_re.search(line)
                if m_place:
                    x = int(m_place.group(2)) / units
                    y = int(m_place.group(3)) / units
                    orient = m_place.group(4)
                    inst_name, cell_type = pending_inst
                    if is_macro_type(cell_type):
                        macros.append((inst_name, cell_type, x, y, orient))
                    else:
                        stdcells.append((x, y))
                    pending_inst = None

    return die_area, macros, np.array(stdcells) if stdcells else np.empty((0, 2)), units


def plot_placement(ax, def_path, title=None, subtitle=None,
                   density_bins=128, show_macro_labels=False):
    """Plot a single placement on the given axes."""
    die_area, macros, stdcells, units = parse_def(def_path)
    xl, yl, xh, yh = die_area
    w_die = xh - xl
    h_die = yh - yl

    # White background
    ax.set_facecolor(C_BG)

    # Std cell density heatmap (subtle blue-grey)
    if len(stdcells) > 0:
        heatmap, xedges, yedges = np.histogram2d(
            stdcells[:, 0], stdcells[:, 1],
            bins=density_bins,
            range=[[xl, xh], [yl, yh]]
        )
        heatmap = heatmap.T
        # Clip top 1% to avoid hotspot saturation
        vmax = np.percentile(heatmap[heatmap > 0], 99) if np.any(heatmap > 0) else 1
        ax.imshow(heatmap, origin='lower', extent=[xl, xh, yl, yh],
                  cmap='GnBu', alpha=0.55, aspect='equal', vmin=0, vmax=vmax,
                  interpolation='bilinear', zorder=1)

    # Draw macros (color by type)
    for inst_name, cell_type, mx, my, orient in macros:
        if cell_type in MACRO_SIZES:
            mw, mh = MACRO_SIZES[cell_type]
        else:
            mw, mh = 50, 50
        if orient in ('E', 'W', 'FE', 'FW'):
            mw, mh = mh, mw
        fill_c, edge_c = MACRO_TYPE_COLORS.get(cell_type, MACRO_DEFAULT_COLORS)
        rect = mpatches.Rectangle(
            (mx, my), mw, mh,
            facecolor=fill_c, edgecolor=edge_c,
            linewidth=0.5, alpha=0.92, zorder=3
        )
        ax.add_patch(rect)

    # Die outline
    die_rect = mpatches.Rectangle(
        (xl, yl), w_die, h_die,
        fill=False, edgecolor=C_DIE_EDGE, linewidth=1.0, zorder=5
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
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=7.5, color='#444444',
                linespacing=1.4)

    n_macros = len(macros)
    n_std = len(stdcells)
    print(f'  {title}: {n_macros} macros, {n_std:,} std cells')


def _make_legend(ax):
    """Add shared macro-type legend to an axes."""
    from matplotlib.lines import Line2D
    legend_items = [
        mpatches.Patch(facecolor=_NBLUE[0], edgecolor=_NBLUE[1],
                       linewidth=0.6, label='SRAM 2048\u00d739'),
        mpatches.Patch(facecolor=_NRED[0], edgecolor=_NRED[1],
                       linewidth=0.6, label='SRAM 256\u00d734'),
        mpatches.Patch(facecolor=_NTEAL[0], edgecolor=_NTEAL[1],
                       linewidth=0.6, label='SRAM 64\u00d721'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='#6BAED6', markersize=5,
               label='Std cell density'),
    ]
    return legend_items


def main():
    """Generate placement comparison figure for swerv_wrapper."""
    data_dir = 'e:/ChipSAT/dreamplace_data/swerv_wrapper'
    out_dir = os.path.dirname(os.path.abspath(__file__))

    defs = [
        # (path, title, subtitle with metrics)
        (os.path.join(data_dir, 'swerv_rtlmp_placed.def'),
         '(a) Hier-RTLMP',
         'HPWL = 3.66\u00d710\u2076   mHPWL = 0.45\u00d710\u2076\nTNS = \u2212490   WNS = \u22120.57'),
        (os.path.join(data_dir, 'cfg_default_final.def'),
         '(b) DREAMPlace (default)',
         'HPWL = 2.35\u00d710\u2076   mHPWL = 0.28\u00d710\u2076\nTNS = \u22121,067   WNS = \u22123.02'),
        (os.path.join(data_dir, 'cfg_2109_final.def'),
         '(c) PPASurrogate (best)',
         'HPWL = 3.08\u00d710\u2076   mHPWL = 0.67\u00d710\u2076\nTNS = \u2212288   WNS = \u22120.48'),
    ]

    fig, axes = plt.subplots(1, len(defs), figsize=(7.0, 3.2))
    if len(defs) == 1:
        axes = [axes]

    for ax, (def_path, title, subtitle) in zip(axes, defs):
        if not os.path.exists(def_path):
            print(f'  [SKIP] {def_path} not found')
            continue
        plot_placement(ax, def_path, title=title, subtitle=subtitle,
                       density_bins=100)

    fig.tight_layout(w_pad=0.5)
    fig.subplots_adjust(bottom=0.13)
    out_path = os.path.join(out_dir, 'placement_comparison.png')
    fig.savefig(out_path, dpi=200)
    fig.savefig(os.path.join(out_dir, 'placement_comparison.pdf'))
    plt.close(fig)
    print(f'  Saved: {out_path}')


if __name__ == '__main__':
    print('Generating placement visualization ...')
    main()
    print('Done.')
