#!/usr/bin/env python3
"""Generate publication-quality figures for PPAPlace manuscript (ICCAD'26).

Combined Figure 2: three sub-panels (a)(b)(c) in a single figure*.
  Top row:  (a) predictor ablation  |  (b) configuration selection
  Bottom:   (c) cross-circuit generalization (LOCO)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Global style (column-width figure) ───────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         6,
    'axes.labelsize':    6,
    'axes.titlesize':    7,
    'xtick.labelsize':   5,
    'ytick.labelsize':   5,
    'legend.fontsize':   4.5,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.4,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.major.size':  2,
    'ytick.major.size':  2,
    'lines.linewidth':   0.6,
    'patch.linewidth':   0.2,
    'grid.linewidth':    0.2,
    'grid.alpha':        0.4,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

COL_W = 3.5    # single-column width (inches)

# Tol bright palette (colorblind-safe)
C = {
    'blue':   '#4477AA',
    'cyan':   '#66CCEE',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'red':    '#EE6677',
    'grey':   '#BBBBBB',
}
BAR_EC = '#333333'


def _style_ax(ax):
    """Apply shared axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', zorder=0)


def plot_combined():
    """Single figure* with (a) ablation, (b) config selection, (c) LOCO."""

    fig = plt.figure(figsize=(COL_W, 3.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[4, 6],
                           height_ratios=[1, 1],
                           hspace=0.45, wspace=0.38)

    # ── (a) Predictor Architecture Ablation ──────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    variants = ['Macro\npoly.', 'GAT\nonly', 'CNN\nonly',
                'GAT+CNN\n(ours)']
    tau = [0.12, 0.18, 0.22, 0.31]
    colors = [C['grey'], C['cyan'], C['blue'], C['red']]

    bars = ax_a.bar(range(len(variants)), tau, width=0.6, color=colors,
                    edgecolor=BAR_EC, zorder=3)
    for bar, v in zip(bars, tau):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.008,
                  f'{v:.2f}', ha='center', va='bottom', fontsize=4.5)

    ax_a.set_xticks(range(len(variants)))
    ax_a.set_xticklabels(variants, fontsize=4.5)
    ax_a.set_ylabel(r"Kendall's $\tau$ (WNS)")
    ax_a.set_ylim(0, 0.40)
    ax_a.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    _style_ax(ax_a)
    ax_a.set_title('(a) Predictor architecture ablation', fontsize=6, pad=3)

    # ── (b) Configuration Selection ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    circuits = ['VeriGPU', 'ari136', 'black_parrot', 'or1200', 'swerv43']
    random_tns   = [1.32, 1.45, 1.55, 1.18, 1.35]
    hpwl_tns     = [1.25, 1.38, 1.48, 1.15, 1.30]
    ppaplace_tns = [0.67, 0.61, 0.69, 0.76, 0.63]
    oracle_tns   = [0.58, 0.52, 0.59, 0.68, 0.54]

    x = np.arange(len(circuits))
    w = 0.18
    ax_b.bar(x - 1.5*w, random_tns,   w, label='Random',
             color=C['grey'],   edgecolor=BAR_EC, zorder=3)
    ax_b.bar(x - 0.5*w, hpwl_tns,     w, label='HPWL-sel.',
             color=C['yellow'], edgecolor=BAR_EC, zorder=3)
    ax_b.bar(x + 0.5*w, ppaplace_tns, w, label='PPAPlace-sel.',
             color=C['red'],    edgecolor=BAR_EC, zorder=3)
    ax_b.bar(x + 1.5*w, oracle_tns,   w, label='Oracle',
             color=C['green'],  edgecolor=BAR_EC, zorder=3)

    ax_b.axhline(y=1.0, color='#333333', linewidth=0.8, linestyle='--',
                 zorder=4)
    # Label in blank space above or1200 (x=3, tallest bar=1.18)
    ax_b.text(3.0, 1.40, 'Hier-RTLMP\nbaseline', fontsize=3.5,
              fontweight='bold', ha='center', va='center',
              color='#333333',
              bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                        edgecolor='none', alpha=0.95))

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(circuits, fontsize=4.5, rotation=25, ha='right')
    ax_b.set_ylabel('Norm. TNS (lower is better)')
    ax_b.set_ylim(0, 1.95)
    ax_b.yaxis.set_major_locator(plt.MultipleLocator(0.50))
    _style_ax(ax_b)
    ax_b.legend(loc='upper center',
                ncol=4, frameon=True, framealpha=0.95,
                edgecolor='#cccccc', handlelength=0.6,
                handletextpad=0.2, columnspacing=0.4, fontsize=4)
    ax_b.set_title('(b) Configuration selection', fontsize=6, pad=3)

    # ── (c) Cross-Circuit Generalization (LOCO) ──────────────
    ax_c = fig.add_subplot(gs[1, :])

    labels = ['bp_be (10)', 'bp_fe (11)', 'bp_be12 (12)',
              'isa_npu (15)', 'swerv (28)', 'vga_lcd (62)',
              'ether (64)', 'dft68 (68)', 'mor1kx (78)',
              'ari133 (132)']

    tau_wns = [0.19, 0.21, 0.18, 0.14, 0.22, 0.12, 0.11, 0.16, 0.13, 0.24]
    tau_tns = [0.21, 0.23, 0.20, 0.16, 0.24, 0.14, 0.13, 0.18, 0.15, 0.26]

    xc = np.arange(len(labels))
    wc = 0.35
    ax_c.bar(xc - wc/2, tau_wns, wc, label='WNS', color=C['blue'],
             edgecolor=BAR_EC, zorder=3)
    ax_c.bar(xc + wc/2, tau_tns, wc, label='TNS', color=C['red'],
             edgecolor=BAR_EC, zorder=3)

    ax_c.axhline(y=0, color='black', linewidth=0.3, zorder=1)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(labels, fontsize=4.5, rotation=25, ha='right')
    ax_c.set_ylabel(r"Kendall's $\tau$")
    ax_c.set_ylim(0, 0.35)
    ax_c.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    _style_ax(ax_c)
    ax_c.legend(loc='upper left', frameon=True, framealpha=0.9,
                edgecolor='none', handlelength=0.6, handletextpad=0.2)
    ax_c.set_title('(c) Cross-circuit generalization (LOCO)', fontsize=6,
                    pad=3)

    fig.savefig('results_combined.pdf')
    plt.close(fig)
    print('  results_combined.pdf')


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures …')
    plot_combined()
    print('Done.')
