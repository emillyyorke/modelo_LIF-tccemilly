#!/usr/bin/env python3
"""
plot_summary_tcc.py
===================
Layout para o TCC:

  Coluna esquerda — label A (unico, no topo):
    A sup. esq.: Raster de spikes
    A inf. esq.: Atividade da rede

  Coluna direita:
    C sup. dir.: Correlacao PRECEDENTE
    D inf. dir.: Correlacao SEGUINTE

Uso manual:
  python plot_summary_tcc.py --dir ./results/DEPRESSION_STDP_OFF_...
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def detect_episodes(activity, threshold_frac=0.15, min_episode_ms=10,
                    min_gap_ms=50, dt_ms=1.0):
    a = np.asarray(activity, dtype=float)
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    thr = lo + threshold_frac * (hi - lo)
    above = a > thr
    diff  = np.diff(above.astype(int))
    onsets  = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    if above[0]:  onsets  = np.concatenate([[0], onsets])
    if above[-1]: offsets = np.concatenate([offsets, [len(a) - 1]])
    n_ep = min(len(onsets), len(offsets))
    episodes = []
    for i in range(n_ep):
        dur = (offsets[i] - onsets[i]) * dt_ms
        if dur >= min_episode_ms:
            episodes.append({'onset_ms':  onsets[i] * dt_ms,
                             'offset_ms': offsets[i] * dt_ms,
                             'duration_ms': dur})
    merged = []
    for ep in episodes:
        if merged and (ep['onset_ms'] - merged[-1]['offset_ms']) < min_gap_ms:
            merged[-1]['offset_ms']   = ep['offset_ms']
            merged[-1]['duration_ms'] = ep['offset_ms'] - merged[-1]['onset_ms']
        else:
            merged.append(ep)
    return merged


def compute_intervals(episodes):
    n = len(episodes)
    if n < 3:
        return None
    durations = np.array([ep['duration_ms'] for ep in episodes])
    preceding = np.array([episodes[i]['onset_ms'] - episodes[i-1]['offset_ms']
                          for i in range(1, n)])
    following = np.array([episodes[i+1]['onset_ms'] - episodes[i]['offset_ms']
                          for i in range(n - 1)])
    dur_prec = durations[1:]
    dur_foll = durations[:-1]
    np_ = min(len(dur_prec), len(preceding))
    nf  = min(len(dur_foll), len(following))
    return {'preceding': preceding[:np_], 'dur_prec': dur_prec[:np_],
            'following': following[:nf],  'dur_foll': dur_foll[:nf]}


def _run(args):
    rdir       = args.dir
    thr        = getattr(args, 'thr', 0.15)
    skip_first = getattr(args, 'skip_first', 1)

    assert os.path.isdir(rdir), f"Diretorio nao encontrado: {rdir}"

    rate_t  = np.load(os.path.join(rdir, "rate_t.npy"))
    rate_hz = np.load(os.path.join(rdir, "rate_hz.npy"))
    spike_i = np.load(os.path.join(rdir, "spike_i.npy"))
    spike_t = np.load(os.path.join(rdir, "spike_t.npy"))

    a_min, a_max = rate_hz.min(), rate_hz.max()
    a_norm = (rate_hz - a_min) / (a_max - a_min) if a_max > a_min else np.zeros_like(rate_hz)

    dt_rate  = float(np.median(np.diff(rate_t))) if len(rate_t) > 1 else 1.0
    episodes = detect_episodes(a_norm, threshold_frac=thr, dt_ms=dt_rate)
    skip_n   = max(0, int(skip_first))
    if skip_n >= len(episodes):
        skip_n = 0
    episodes_stats = episodes[skip_n:]
    corr_data      = compute_intervals(episodes_stats)
    n_ep_total     = len(episodes)
    print(f"[INFO] Episodios: {n_ep_total} | para estatisticas: {len(episodes_stats)}")

    VIEW_MAX_S  = 40.0
    view_end_s  = min(rate_t[-1] / 1000.0, VIEW_MAX_S)
    view_end_ms = view_end_s * 1000.0

    # ============================================================
    # Estilo
    # ============================================================
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    11,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
    })

    # ============================================================
    # Layout 2x2:
    #   col 0: raster (cima) + atividade (baixo)  → label A unico no topo
    #   col 1: precedente (cima) + seguinte (baixo) → labels C e D
    # ============================================================
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(
        2, 2,
        hspace=0.28,
        wspace=0.18,
        left=0.08, right=0.98,
        top=0.94,  bottom=0.09,
    )

    ax_rast  = fig.add_subplot(gs[0, 0])  # col 0, linha 0 — Raster
    ax_act   = fig.add_subplot(gs[1, 0])  # col 0, linha 1 — Atividade
    ax_prec  = fig.add_subplot(gs[0, 1])  # col 1, linha 0 — Precedente
    ax_foll  = fig.add_subplot(gs[1, 1])  # col 1, linha 1 — Seguinte

    # ── Label A — unico, fora do ax_rast, posicionado para cobrir os dois ──
    ax_rast.text(-0.13, 1.05, "A",
                 transform=ax_rast.transAxes,
                 fontsize=15, fontweight='bold',
                 va='bottom', ha='left')

    # ── Raster ─────────────────────────────────────────────────
    spk_mask  = spike_t <= view_end_ms
    n_neurons = int(spike_i.max()) + 1 if len(spike_i) > 0 else 100
    ax_rast.scatter(spike_t[spk_mask] / 1000.0, spike_i[spk_mask],
                    s=0.8, marker='.', c='#CC2222', alpha=0.6,
                    linewidths=0, rasterized=True)
    ax_rast.set_xlim([0, view_end_s])
    ax_rast.set_ylim([-1, n_neurons])
    ax_rast.invert_yaxis()
    ax_rast.set_xlabel("Tempo (s)")
    ax_rast.set_ylabel("Neurônio")
    ax_rast.set_title("Raster de Spikes", pad=5)
    ax_rast.tick_params(length=4)

    # ── Atividade ───────────────────────────────────────────────
    rate_mask = rate_t <= view_end_ms
    ax_act.plot(rate_t[rate_mask] / 1000.0, a_norm[rate_mask],
                lw=0.7, color='black', zorder=3)
    for ep in episodes:
        if ep['onset_ms'] > view_end_ms:
            break
        x0 = ep['onset_ms'] / 1000.0
        x1 = min(ep['offset_ms'], view_end_ms) / 1000.0
        ax_act.axvspan(x0, x1, alpha=0.09, color='steelblue', zorder=1)
    ax_act.set_xlim([0, view_end_s])
    ax_act.set_ylim([-0.03, 1.10])
    ax_act.set_xlabel("Tempo (s)")
    ax_act.set_ylabel(r"$\langle a \rangle$ (normalizada)")
    ax_act.set_title(f"Atividade da Rede  ({n_ep_total} episódios)", pad=5)
    ax_act.tick_params(length=4)

    # ── C — Correlação PRECEDENTE ───────────────────────────────
    ax_prec.text(-0.13, 1.05, "B",
                 transform=ax_prec.transAxes,
                 fontsize=15, fontweight='bold',
                 va='bottom', ha='left')
    if corr_data is not None and len(corr_data['dur_prec']) > 2:
        x, y = corr_data['preceding'], corr_data['dur_prec']
        r_val, p_val = stats.pearsonr(x, y)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax_prec.scatter(x, y, s=35, c='black', alpha=0.75,
                        edgecolors='none', marker='o', zorder=3)
        ax_prec.set_title(f"PRECEDENTE  —  R = {r_val:.2f}  ({p_str},  n = {len(x)})", pad=5)
    else:
        ax_prec.set_title("episodios insuficientes", pad=5)
    ax_prec.set_xlabel("Intervalo interepisódico precedente (ms)")
    ax_prec.set_ylabel("Duração do episódio (ms)")
    ax_prec.tick_params(length=4)

    # ── D — Correlação SEGUINTE ─────────────────────────────────
    ax_foll.text(-0.13, 1.05, "C",
                 transform=ax_foll.transAxes,
                 fontsize=15, fontweight='bold',
                 va='bottom', ha='left')
    if corr_data is not None and len(corr_data['dur_foll']) > 2:
        x, y = corr_data['following'], corr_data['dur_foll']
        r_val, p_val = stats.pearsonr(x, y)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax_foll.scatter(x, y, s=35, c='black', alpha=0.75,
                        edgecolors='none', marker='o', zorder=3)
        ax_foll.set_title(f"SEGUINTE  —  R = {r_val:.2f}  ({p_str},  n = {len(x)})", pad=5)
    else:
        ax_foll.set_title("episodios insuficientes", pad=5)
    ax_foll.set_xlabel("Intervalo interepisódico seguinte (ms)")
    ax_foll.set_ylabel("Duração do episódio (ms)")
    ax_foll.tick_params(length=4)

    # Eixos Y iguais em C e D
    if corr_data is not None:
        all_y = np.concatenate([corr_data['dur_prec'], corr_data['dur_foll']])
        ypad  = (all_y.max() - all_y.min()) * 0.12 + 1
        ax_prec.set_ylim([all_y.min() - ypad, all_y.max() + ypad])
        ax_foll.set_ylim([all_y.min() - ypad, all_y.max() + ypad])

    # ── Salvar ──────────────────────────────────────────────────
    out_png = os.path.join(rdir, "summary_tcc.png")
    out_pdf = os.path.join(rdir, "summary_tcc.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[OK] PNG salvo em: {out_png}")
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"[OK] PDF vetorial salvo em: {out_pdf}")
    plt.close(fig)


def main_from_args(args):
    """Chamado automaticamente pelo LIF_EMILLY.py ao fim da simulacao."""
    import matplotlib
    matplotlib.use('Agg')
    _run(args)


def main():
    parser = argparse.ArgumentParser(description="Figura TCC layout Tabak")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--thr", type=float, default=0.15)
    parser.add_argument("--skip-first", type=int, default=1, dest='skip_first')
    args = parser.parse_args()
    _run(args)
    plt.show()


if __name__ == "__main__":
    main()
