#!/usr/bin/env python3
"""
plot_summary.py
===============
Painel resumo da simulacao em layout 2x2 otimizado para inclusao em TCC/tese:
  Painel A (sup. esq.):  Raster plot de spikes
  Painel B (sup. dir.):  Atividade da rede (taxa de disparo suavizada)
  Painel C (inf. esq.):  Correlacao duracao vs intervalo PRECEDENTE
  Painel D (inf. dir.):  Correlacao duracao vs intervalo SEGUINTE

Le diretamente dos .npy (nao precisa de export_for_plot).

A figura segue o padrao da Fig. 2 de Tabak et al. (2010), com rotulos A/B/C/D
em cada painel para facilitar a descricao no texto.

Uso:
  python plot_summary.py --dir ./results/DEPRESSION_STDP_OFF_...
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Detecção de episódios (mesma lógica do plot_tabak_analysis)
# ============================================================
def detect_episodes(activity, threshold_frac=0.15, min_episode_ms=10,
                    min_gap_ms=50, dt_ms=1.0):
    a = np.asarray(activity, dtype=float)
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    thr = lo + threshold_frac * (hi - lo)

    above = a > thr
    diff = np.diff(above.astype(int))
    onsets  = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1

    if above[0]:
        onsets = np.concatenate([[0], onsets])
    if above[-1]:
        offsets = np.concatenate([offsets, [len(a) - 1]])

    n_ep = min(len(onsets), len(offsets))
    episodes = []
    for i in range(n_ep):
        on_i  = onsets[i]
        off_i = offsets[i]
        dur = (off_i - on_i) * dt_ms
        if dur >= min_episode_ms:
            episodes.append({
                'onset_idx':  on_i,
                'offset_idx': off_i,
                'onset_ms':   on_i * dt_ms,
                'offset_ms':  off_i * dt_ms,
                'duration_ms': dur,
            })

    merged = []
    for ep in episodes:
        if merged and (ep['onset_ms'] - merged[-1]['offset_ms']) < min_gap_ms:
            merged[-1]['offset_idx'] = ep['offset_idx']
            merged[-1]['offset_ms']  = ep['offset_ms']
            merged[-1]['duration_ms'] = ep['offset_ms'] - merged[-1]['onset_ms']
        else:
            merged.append(ep)

    return merged


def compute_intervals(episodes):
    n = len(episodes)
    if n < 3:
        return None

    durations = np.array([ep['duration_ms'] for ep in episodes])

    preceding = np.array([
        episodes[i]['onset_ms'] - episodes[i-1]['offset_ms']
        for i in range(1, n)
    ])
    following = np.array([
        episodes[i+1]['onset_ms'] - episodes[i]['offset_ms']
        for i in range(n - 1)
    ])

    dur_for_prec = durations[1:]
    dur_for_foll = durations[:-1]

    n_prec = min(len(dur_for_prec), len(preceding))
    n_foll = min(len(dur_for_foll), len(following))

    return {
        'preceding': preceding[:n_prec],
        'dur_prec':  dur_for_prec[:n_prec],
        'following': following[:n_foll],
        'dur_foll':  dur_for_foll[:n_foll],
    }


# ============================================================
# Ler parâmetros da simulação
# ============================================================
def get_params(rdir):
    params = {}
    params_path = os.path.join(rdir, "params.txt")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.split("=", 1)
                    params[key.strip()] = value.strip()
    return params


# ============================================================
# Detectar picos de atividade (para linhas verticais)
# ============================================================
def find_peaks_simple(y, dt_ms, prominence=0.10, min_dist_ms=400):
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    cand = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if len(cand) == 0:
        return np.array([], dtype=int)
    ymin, ymax = np.min(y), np.max(y)
    span = max(1e-9, ymax - ymin)
    good = cand[y[cand] >= (ymin + prominence * span)]
    if len(good) <= 1:
        return good
    min_dist = max(1, int(min_dist_ms / dt_ms))
    keep = [int(good[0])]
    last = good[0]
    for idx in good[1:]:
        if idx - last >= min_dist:
            keep.append(int(idx))
            last = idx
    return np.asarray(keep, dtype=int)


# ============================================================
# Adicionar rótulo A/B/C/D no canto superior esquerdo do painel
# ============================================================
def add_panel_label(ax, label, dx=-0.12, dy=1.08):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Painel resumo 2x2: raster + atividade + correlacoes")
    parser.add_argument("--dir", required=True,
                        help="Pasta de resultados da simulacao")
    parser.add_argument("--thr", type=float, default=0.15,
                        help="Limiar fracional para deteccao de episodios")
    parser.add_argument("--skip-first", type=int, default=1,
                        help="Quantos episodios iniciais descartar das estatisticas "
                             "(transientes de inicializacao). Padrao: 1. Use 0 "
                             "para incluir todos.")
    args = parser.parse_args()

    rdir = args.dir
    assert os.path.isdir(rdir), f"Diretorio nao encontrado: {rdir}"

    # ---- Carregar dados ----
    spike_i = np.load(os.path.join(rdir, "spike_i.npy"))
    spike_t = np.load(os.path.join(rdir, "spike_t.npy"))   # ms
    rate_t  = np.load(os.path.join(rdir, "rate_t.npy"))     # ms
    rate_hz = np.load(os.path.join(rdir, "rate_hz.npy"))    # Hz

    params = get_params(rdir)

    # ---- Normalizar atividade para [0, 1] ----
    a_min, a_max = rate_hz.min(), rate_hz.max()
    if a_max > a_min:
        a_norm = (rate_hz - a_min) / (a_max - a_min)
    else:
        a_norm = np.zeros_like(rate_hz)

    # ---- Detectar episódios ----
    dt_rate = np.median(np.diff(rate_t)) if len(rate_t) > 1 else 1.0
    episodes = detect_episodes(a_norm, threshold_frac=args.thr, dt_ms=dt_rate)

    # ---- Separar: visualização usa TODOS, estatísticas descartam os primeiros N ----
    skip_n = max(0, int(args.skip_first))
    if skip_n >= len(episodes):
        print(f"[AVISO] skip_first={skip_n} >= episodios detectados ({len(episodes)}). "
              f"Mantendo todos para estatisticas.")
        skip_n = 0
    episodes_stats = episodes[skip_n:]
    corr_data = compute_intervals(episodes_stats)

    print(f"[INFO] Episodios detectados: {len(episodes)} "
          f"(usando {len(episodes_stats)} para estatisticas, "
          f"descartando {skip_n} inicial(is))")

    # ---- Picos para linhas verticais ----
    peak_idx = find_peaks_simple(a_norm, dt_rate)
    peak_t_ms = rate_t[peak_idx] if peak_idx.size else np.array([])

    # ---- Tempo máximo ----
    t_max_ms = rate_t[-1] if len(rate_t) > 0 else 1.0
    t_max_s  = t_max_ms / 1000.0

    # ---- Limite visual da janela temporal (raster + atividade) ----
    # Nao afeta as estatisticas: TODOS os episodios continuam contribuindo para R.
    VIEW_MAX_S = 40.0
    view_end_s = min(t_max_s, VIEW_MAX_S)
    view_end_ms = view_end_s * 1000.0

    # ============================================================
    # FIGURA — Layout 2x2 otimizado para TCC/tese
    # ============================================================
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Layout 2x2: largura maior que altura, adequado para paginas retrato
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.45,
        wspace=0.28,
        left=0.08, right=0.97,
        top=0.93, bottom=0.08,
    )

    # ──────────────────────────────────────
    # PAINEL A (superior esquerdo): Raster plot
    # ──────────────────────────────────────
    ax_raster = fig.add_subplot(gs[0, 0])
    spk_mask = spike_t <= view_end_ms
    ax_raster.scatter(spike_t[spk_mask] / 1000.0, spike_i[spk_mask],
                      s=1.5, marker='.', c='red', alpha=0.7,
                      linewidths=0, rasterized=True)
    for tt in peak_t_ms:
        if tt <= view_end_ms:
            ax_raster.axvline(tt / 1000.0, color='gray', lw=0.6, ls='--', alpha=0.4)
    n_neurons = int(spike_i.max()) + 1 if len(spike_i) > 0 else 100
    ax_raster.set_xlim([0, view_end_s])
    ax_raster.set_ylim([-1, n_neurons])
    ax_raster.invert_yaxis()
    ax_raster.set_ylabel("Neuronio")
    ax_raster.set_xlabel("Tempo (s)")
    if t_max_s > VIEW_MAX_S:
        ax_raster.set_title(f"Raster de Spikes (primeiros {VIEW_MAX_S:.0f} s)")
    else:
        ax_raster.set_title("Raster de Spikes")
    ax_raster.grid(alpha=0.15, linestyle=":")
    add_panel_label(ax_raster, "A")

    # ──────────────────────────────────────
    # PAINEL B (superior direito): Atividade da rede
    # ──────────────────────────────────────
    ax_act = fig.add_subplot(gs[0, 1])
    rate_mask = rate_t <= view_end_ms
    ax_act.plot(rate_t[rate_mask] / 1000.0, a_norm[rate_mask],
                lw=0.9, color='black')
    for tt in peak_t_ms:
        if tt <= view_end_ms:
            ax_act.axvline(tt / 1000.0, color='gray', lw=0.6, ls='--', alpha=0.4)

    # Marcar episodios visiveis
    for i, ep in enumerate(episodes):
        if ep['onset_ms'] > view_end_ms:
            break
        if i < skip_n:
            color, alpha = 'gray', 0.15
        else:
            color, alpha = 'blue', 0.08
        x0 = ep['onset_ms'] / 1000.0
        x1 = min(ep['offset_ms'], view_end_ms) / 1000.0
        ax_act.axvspan(x0, x1, alpha=alpha, color=color)

    ax_act.set_xlim([0, view_end_s])
    ax_act.set_ylim([-0.02, 1.05])
    ax_act.set_ylabel("Atividade (normalizada)")
    ax_act.set_xlabel("Tempo (s)")
    ax_act.set_title(f"Atividade da rede  ({len(episodes)} episodios)")
    ax_act.grid(alpha=0.15, linestyle=":")
    add_panel_label(ax_act, "B")

    # ──────────────────────────────────────
    # PAINEL C (inferior esquerdo): Correlacao PRECEDENTE
    # ──────────────────────────────────────
    ax_prec = fig.add_subplot(gs[1, 0])
    if corr_data is not None and len(corr_data['dur_prec']) > 2:
        x = corr_data['preceding']
        y = corr_data['dur_prec']
        r_val, p_val = stats.pearsonr(x, y)
        ax_prec.scatter(x, y, s=30, c='black', alpha=0.7, edgecolors='none')
        z = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 50)
        ax_prec.plot(xfit, np.polyval(z, xfit), 'r--', lw=1.5)
        ax_prec.set_title(
            f'Precedente  —  R = {r_val:.2f}  (n = {len(x)})'
        )
    else:
        ax_prec.set_title('Precedente  —  dados insuficientes')
        ax_prec.text(0.5, 0.5, f'{len(episodes_stats)} episodios usados\n'
                                f'(min. 3 necessarios)',
                     transform=ax_prec.transAxes, ha='center', va='center',
                     fontsize=11, color='gray')
    ax_prec.set_xlabel('Intervalo interepisodico precedente (ms)')
    ax_prec.set_ylabel('Duracao do episodio (ms)')
    ax_prec.grid(alpha=0.2)
    add_panel_label(ax_prec, "C")

    # ──────────────────────────────────────
    # PAINEL D (inferior direito): Correlacao SEGUINTE
    # ──────────────────────────────────────
    ax_foll = fig.add_subplot(gs[1, 1])
    if corr_data is not None and len(corr_data['dur_foll']) > 2:
        x = corr_data['following']
        y = corr_data['dur_foll']
        r_val, p_val = stats.pearsonr(x, y)
        ax_foll.scatter(x, y, s=30, c='black', alpha=0.7, edgecolors='none')
        z = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 50)
        ax_foll.plot(xfit, np.polyval(z, xfit), 'r--', lw=1.5)
        ax_foll.set_title(
            f'Seguinte  —  R = {r_val:.2f}  (n = {len(x)})'
        )
    else:
        ax_foll.set_title('Seguinte  —  dados insuficientes')
        ax_foll.text(0.5, 0.5, f'{len(episodes_stats)} episodios usados\n'
                                f'(min. 3 necessarios)',
                     transform=ax_foll.transAxes, ha='center', va='center',
                     fontsize=11, color='gray')
    ax_foll.set_xlabel('Intervalo interepisodico seguinte (ms)')
    ax_foll.set_ylabel('Duracao do episodio (ms)')
    ax_foll.grid(alpha=0.2)
    add_panel_label(ax_foll, "D")

    # ---- Salvar (300 DPI, sem bbox_inches='tight' para manter dimensoes consistentes) ----
    out_path = os.path.join(rdir, "summary.png")
    fig.savefig(out_path, dpi=300)
    print(f"[OK] Painel resumo salvo em: {out_path}")

    # Salvar tambem em PDF vetorial (melhor para TCC)
    out_path_pdf = os.path.join(rdir, "summary.pdf")
    fig.savefig(out_path_pdf)
    print(f"[OK] Versao PDF vetorial salva em: {out_path_pdf}")

    plt.show()


if __name__ == "__main__":
    main()

