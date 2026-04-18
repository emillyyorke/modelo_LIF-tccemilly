#!/usr/bin/env python3
"""
plot_tabak_analysis.py
======================
Reproduz os painéis da Figura 2 (depressão) ou Figura 5 (adaptação)
de Tabak et al. (2010):
  A) Séries temporais de <a> e <s> (ou <θ>)
  B) Plano de fase <a> vs <s> (ou <θ>)
  C) Correlação: duração do episódio vs intervalo PRECEDENTE
  D) Correlação: duração do episódio vs intervalo SEGUINTE
  E) Distribuição da variável lenta no início dos episódios
  F) Distribuição da variável lenta no término dos episódios

Detecta automaticamente o modo (depression/adaptation) a partir dos
arquivos presentes na pasta de resultados.

Uso:
  python plot_tabak_analysis.py --dir ./results/DEPRESSION_STDP_OFF_...
  python plot_tabak_analysis.py --dir ./results/ADAPTATION_STDP_OFF_...
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Detecção de episódios
# ============================================================
def detect_episodes(activity, threshold_frac=0.15, min_episode_ms=10,
                    min_gap_ms=50, dt_ms=1.0):
    """
    Detecta episódios de atividade alta.
    threshold_frac: fração entre min e max da atividade para considerar 'ativo'.
    Retorna lista de dicts com {onset_idx, offset_idx, onset_ms, offset_ms, duration_ms}.
    """
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

    # Mesclar episódios muito próximos
    merged = []
    for ep in episodes:
        if merged and (ep['onset_ms'] - merged[-1]['offset_ms']) < min_gap_ms:
            merged[-1]['offset_idx'] = ep['offset_idx']
            merged[-1]['offset_ms']  = ep['offset_ms']
            merged[-1]['duration_ms'] = ep['offset_ms'] - merged[-1]['onset_ms']
        else:
            merged.append(ep)

    return merged


def compute_intervals_and_correlations(episodes):
    """
    Calcula intervalos interepisódicos e correlações com duração do episódio.
    """
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
        'durations': durations,
        'preceding': preceding[:n_prec],
        'dur_prec':  dur_for_prec[:n_prec],
        'following': following[:n_foll],
        'dur_foll':  dur_for_foll[:n_foll],
    }


# ============================================================
# Detectar modo a partir dos arquivos disponíveis
# ============================================================
def detect_mode(rdir):
    """
    Detecta o modo de feedback a partir dos arquivos .npy na pasta.
    Retorna 'depression', 'adaptation' ou None.
    """
    has_s     = os.path.exists(os.path.join(rdir, "s_mean.npy"))
    has_theta = os.path.exists(os.path.join(rdir, "theta_mean.npy"))

    if has_s:
        return 'depression'
    elif has_theta:
        return 'adaptation'

    # Tentar ler params.txt
    params_path = os.path.join(rdir, "params.txt")
    if os.path.exists(params_path):
        with open(params_path) as f:
            for line in f:
                if 'FEEDBACK_MODE' in line and 'adaptation' in line:
                    return 'adaptation'
                if 'FEEDBACK_MODE' in line and 'depression' in line:
                    return 'depression'

    return None


# ============================================================
# Carregar variável lenta conforme o modo
# ============================================================
def load_slow_variable(rdir, mode):
    """
    Carrega a variável lenta e seus timestamps.
    Retorna (slow_mean, slow_t_ms, label_symbol, label_name, increases_during_episode).
    """
    if mode == 'depression':
        slow_mean = np.load(os.path.join(rdir, "s_mean.npy"))
        slow_t    = np.load(os.path.join(rdir, "s_t.npy"))
        return slow_mean, slow_t, r'$\langle s \rangle$', 'recuperação', False

    elif mode == 'adaptation':
        slow_mean = np.load(os.path.join(rdir, "theta_mean.npy"))
        slow_t    = np.load(os.path.join(rdir, "theta_t.npy"))
        return slow_mean, slow_t, r'$\langle \theta \rangle$', 'adaptação', True

    else:
        raise ValueError(f"Modo inválido: {mode}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Análise estilo Tabak Fig. 2 (depressão) / Fig. 5 (adaptação)")
    parser.add_argument("--dir", required=True,
                        help="Pasta de resultados da simulação")
    parser.add_argument("--thr", type=float, default=0.15,
                        help="Limiar fracional para detecção de episódios")
    parser.add_argument("--skip-first", type=int, default=1,
                        help="Quantos episódios iniciais descartar das estatísticas "
                             "(transientes de inicialização). Padrão: 1.")
    args = parser.parse_args()

    rdir = args.dir
    assert os.path.isdir(rdir), f"Diretório não encontrado: {rdir}"

    # ---- Detectar modo ----
    mode = detect_mode(rdir)
    if mode is None:
        print("[ERRO] Não foi possível detectar o modo (depression/adaptation).")
        print("       Certifique-se de que s_mean.npy ou theta_mean.npy está presente.")
        return

    print(f"[INFO] Modo detectado: {mode}")

    mode_titles = {
        'depression': 'Depressão Sináptica',
        'adaptation': 'Adaptação Celular',
    }
    mode_fig_ref = {
        'depression': 'Fig. 2',
        'adaptation': 'Fig. 5',
    }

    # ---- Carregar dados ----
    rate_t  = np.load(os.path.join(rdir, "rate_t.npy"))
    rate_hz = np.load(os.path.join(rdir, "rate_hz.npy"))

    slow_mean, slow_t, sym_label, var_name, increases = load_slow_variable(rdir, mode)

    # ---- Normalizar atividade ----
    a_raw = rate_hz.copy()
    a_min, a_max = a_raw.min(), a_raw.max()
    if a_max > a_min:
        a_norm = (a_raw - a_min) / (a_max - a_min) * 0.5
    else:
        a_norm = np.zeros_like(a_raw)

    # ---- Interpolar variável lenta no grid de rate_t ----
    slow_interp = np.interp(rate_t, slow_t, slow_mean)

    # ---- Detectar episódios ----
    dt_rate = np.median(np.diff(rate_t)) if len(rate_t) > 1 else 1.0
    episodes = detect_episodes(a_norm, threshold_frac=args.thr, dt_ms=dt_rate)

    # ---- Separar: visualização (A, B) usa TODOS; estatísticas (C-F) descartam os primeiros N ----
    skip_n = max(0, int(args.skip_first))
    if skip_n >= len(episodes):
        print(f"[AVISO] skip_first={skip_n} >= episódios detectados ({len(episodes)}). "
              f"Mantendo todos para estatísticas.")
        skip_n = 0
    episodes_stats = episodes[skip_n:]

    print(f"[INFO] Episódios detectados: {len(episodes)} "
          f"(usando {len(episodes_stats)} para estatísticas, "
          f"descartando {skip_n} inicial(is))")

    if len(episodes_stats) < 3:
        print("[AVISO] Poucos episódios para estatísticas. Aumente SIM_TIME.")

    # Valores da variável lenta no onset/offset — TODOS os episódios (para painel B)
    slow_at_onset_all  = []
    slow_at_offset_all = []
    for ep in episodes:
        idx_on  = int(ep['onset_idx'])
        idx_off = int(ep['offset_idx'])
        if idx_on < len(slow_interp):
            slow_at_onset_all.append(slow_interp[idx_on])
        if idx_off < len(slow_interp):
            slow_at_offset_all.append(slow_interp[idx_off])
    slow_at_onset_all  = np.array(slow_at_onset_all)
    slow_at_offset_all = np.array(slow_at_offset_all)

    # Valores filtrados (sem transiente) — para histogramas E/F e estatísticas
    slow_at_onset  = slow_at_onset_all[skip_n:]
    slow_at_offset = slow_at_offset_all[skip_n:]

    corr_data = compute_intervals_and_correlations(episodes_stats)

    # ============================================================
    # PLOTAR — 6 painéis estilo Tabak
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Análise Tabak ({mode_fig_ref[mode]}) — {mode_titles[mode]}",
                 fontsize=14, y=0.98)

    # ---- A) Séries temporais ----
    ax = axes[0, 0]
    t_sec = rate_t / 1000.0
    ax.plot(t_sec, a_norm, 'k', lw=0.8,
            label=r'$\langle a \rangle$ (atividade)')
    ax.plot(slow_t / 1000.0, slow_mean, color='gray', lw=2.0, alpha=0.7,
            label=f'{sym_label} ({var_name})')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel(f'$\\langle a \\rangle$, {sym_label}')
    ax.set_title('A) Séries temporais')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim([0, rate_t[-1] / 1000.0])
    ax.grid(alpha=0.2)

    # ---- B) Plano de fase ---- (mostra TODOS os episódios, incluindo transiente)
    ax = axes[0, 1]
    ax.plot(slow_interp, a_norm, 'k', lw=0.3, alpha=0.5)
    if len(slow_at_onset_all) > 0:
        a_at_onset  = [a_norm[int(ep['onset_idx'])]
                       for ep in episodes if int(ep['onset_idx']) < len(a_norm)]
        a_at_offset = [a_norm[int(ep['offset_idx'])]
                       for ep in episodes if int(ep['offset_idx']) < len(a_norm)]
        ax.scatter(slow_at_onset_all[:len(a_at_onset)], a_at_onset,
                   c='blue', s=20, zorder=5, label='Onset')
        ax.scatter(slow_at_offset_all[:len(a_at_offset)], a_at_offset,
                   c='red', s=20, zorder=5, label='Offset')
    ax.set_xlabel(sym_label)
    ax.set_ylabel(r'$\langle a \rangle$')
    ax.set_title('B) Plano de fase')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ---- C) Correlação: duração vs intervalo PRECEDENTE ----
    ax = axes[1, 0]
    if corr_data is not None and len(corr_data['dur_prec']) > 2:
        x = corr_data['preceding']
        y = corr_data['dur_prec']
        r_val, p_val = stats.pearsonr(x, y)
        ax.scatter(x, y, s=25, c='black', alpha=0.7)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xfit, np.polyval(z, xfit), 'r--', lw=1)
        ax.set_title(f'C) PRECEDENTE — R = {r_val:.2f}  (n = {len(x)})')
    else:
        ax.set_title('C) PRECEDENTE — dados insuficientes')
    ax.set_xlabel('Intervalo interepisódico (ms)')
    ax.set_ylabel('Duração do episódio (ms)')
    ax.grid(alpha=0.2)

    # ---- D) Correlação: duração vs intervalo SEGUINTE ----
    ax = axes[1, 1]
    if corr_data is not None and len(corr_data['dur_foll']) > 2:
        x = corr_data['following']
        y = corr_data['dur_foll']
        r_val, p_val = stats.pearsonr(x, y)
        ax.scatter(x, y, s=25, c='black', alpha=0.7)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xfit, np.polyval(z, xfit), 'r--', lw=1)
        ax.set_title(f'D) SEGUINTE — R = {r_val:.2f}  (n = {len(x)})')
    else:
        ax.set_title('D) SEGUINTE — dados insuficientes')
    ax.set_xlabel('Intervalo interepisódico (ms)')
    ax.set_ylabel('Duração do episódio (ms)')
    ax.grid(alpha=0.2)

    # ---- E) Distribuição da variável lenta no onset ----
    ax = axes[2, 0]
    if len(slow_at_onset) > 1:
        ax.hist(slow_at_onset, bins=15, edgecolor='black', alpha=0.75,
                color='steelblue')
        sd_on = np.std(slow_at_onset)
        ax.set_title(f'E) {sym_label} no onset — sd = {sd_on:.4f}')
    else:
        ax.set_title(f'E) {sym_label} no onset — dados insuficientes')
    ax.set_xlabel(f'{sym_label} no início do episódio')
    ax.set_ylabel('Frequência')
    ax.grid(alpha=0.2)

    # ---- F) Distribuição da variável lenta no offset ----
    ax = axes[2, 1]
    if len(slow_at_offset) > 1:
        ax.hist(slow_at_offset, bins=15, edgecolor='black', alpha=0.75,
                color='salmon')
        sd_off = np.std(slow_at_offset)
        ax.set_title(f'F) {sym_label} no offset — sd = {sd_off:.4f}')
    else:
        ax.set_title(f'F) {sym_label} no offset — dados insuficientes')
    ax.set_xlabel(f'{sym_label} no término do episódio')
    ax.set_ylabel('Frequência')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(rdir, "tabak_analysis.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Figura completa salva em: {out_path}")

    # ============================================================
    # FIGURA SEPARADA — apenas C e D (correlações)
    # ============================================================
    fig2, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Correlação: Duração do Episódio vs Intervalo Interepisódico",
                  fontsize=13)

    if corr_data is not None and len(corr_data['dur_prec']) > 2:
        x = corr_data['preceding']
        y = corr_data['dur_prec']
        r_val, p_val = stats.pearsonr(x, y)
        ax_c.scatter(x, y, s=30, c='black', alpha=0.7)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax_c.plot(xfit, np.polyval(z, xfit), 'r--', lw=1.5)
        ax_c.set_title(f'PRECEDENTE — R = {r_val:.2f}  (n = {len(x)})', fontsize=12)
    else:
        ax_c.set_title('PRECEDENTE — dados insuficientes', fontsize=12)
    ax_c.set_xlabel('Intervalo interepisódico precedente (ms)', fontsize=10)
    ax_c.set_ylabel('Duração do episódio (ms)', fontsize=10)
    ax_c.grid(alpha=0.2)

    if corr_data is not None and len(corr_data['dur_foll']) > 2:
        x = corr_data['following']
        y = corr_data['dur_foll']
        r_val, p_val = stats.pearsonr(x, y)
        ax_d.scatter(x, y, s=30, c='black', alpha=0.7)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax_d.plot(xfit, np.polyval(z, xfit), 'r--', lw=1.5)
        ax_d.set_title(f'SEGUINTE — R = {r_val:.2f}  (n = {len(x)})', fontsize=12)
    else:
        ax_d.set_title('SEGUINTE — dados insuficientes', fontsize=12)
    ax_d.set_xlabel('Intervalo interepisódico seguinte (ms)', fontsize=10)
    ax_d.set_ylabel('Duração do episódio (ms)', fontsize=10)
    ax_d.grid(alpha=0.2)

    plt.tight_layout()
    out_cd = os.path.join(rdir, "tabak_correlations_CD.png")
    fig2.savefig(out_cd, dpi=200, bbox_inches='tight')
    print(f"[OK] Figura C/D salva em: {out_cd}")

    # ============================================================
    # EXPORTAR DADOS — CSV e TXT
    # ============================================================
    # Rótulo da variável lenta para os cabeçalhos
    slow_col_name = "s" if mode == 'depression' else "theta"
    n_ep = len(episodes)

    if n_ep > 0:
        header_cols = [
            "Episodio", "Onset_ms", "Offset_ms", "Duracao_ms",
            "IEI_precedente_ms", "IEI_seguinte_ms",
            f"{slow_col_name}_no_onset", f"{slow_col_name}_no_offset",
        ]

        rows = []
        for i, ep in enumerate(episodes):
            iei_prec = (ep['onset_ms'] - episodes[i-1]['offset_ms']
                        if i > 0 else np.nan)
            iei_foll = (episodes[i+1]['onset_ms'] - ep['offset_ms']
                        if i < n_ep - 1 else np.nan)
            v_on  = slow_at_onset_all[i]  if i < len(slow_at_onset_all)  else np.nan
            v_off = slow_at_offset_all[i] if i < len(slow_at_offset_all) else np.nan

            # Marca asterisco em episódios descartados das estatísticas
            ep_label = f"{i+1}*" if i < skip_n else f"{i+1}"

            rows.append([
                ep_label,
                f"{ep['onset_ms']:.2f}",
                f"{ep['offset_ms']:.2f}",
                f"{ep['duration_ms']:.2f}",
                f"{iei_prec:.2f}" if np.isfinite(iei_prec) else "—",
                f"{iei_foll:.2f}" if np.isfinite(iei_foll) else "—",
                f"{v_on:.6f}"  if np.isfinite(v_on)  else "—",
                f"{v_off:.6f}" if np.isfinite(v_off) else "—",
            ])

        # ---- CSV ----
        csv_path = os.path.join(rdir, "episodios_tabela.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(",".join(header_cols) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        print(f"[OK] CSV salvo em: {csv_path}")

        # ---- TXT formatado ----
        txt_path = os.path.join(rdir, "episodios_tabela.txt")
        col_widths = [10, 12, 12, 12, 18, 18, 14, 14]

        def fmt_row(vals, widths):
            return "│".join(str(v).center(w) for v, w in zip(vals, widths))

        sep_line = "┼".join("─" * w for w in col_widths)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * sum(col_widths) + "\n")
            f.write(f"  TABELA DE EPISÓDIOS — {mode_titles[mode]} (Tabak et al. 2010)\n")
            f.write("=" * sum(col_widths) + "\n\n")

            header_short = [
                "Ep.", "Onset(ms)", "Offset(ms)", "ED(ms)",
                "IEI_prec(ms)", "IEI_seg(ms)",
                f"{slow_col_name}_onset", f"{slow_col_name}_offset",
            ]
            f.write(fmt_row(header_short, col_widths) + "\n")
            f.write(sep_line + "\n")

            for row in rows:
                f.write(fmt_row(row, col_widths) + "\n")

            f.write(sep_line + "\n\n")

            # Resumo estatístico
            durations = np.array([ep['duration_ms'] for ep in episodes])
            f.write(f"  Total de episódios: {n_ep}\n")
            f.write(f"  Duração média: {np.mean(durations):.2f} ms "
                    f"(sd = {np.std(durations):.2f})\n")
            if len(slow_at_onset) > 1:
                f.write(f"  <{slow_col_name}> onset:  média = "
                        f"{np.mean(slow_at_onset):.4f}  "
                        f"sd = {np.std(slow_at_onset):.4f}\n")
            if len(slow_at_offset) > 1:
                f.write(f"  <{slow_col_name}> offset: média = "
                        f"{np.mean(slow_at_offset):.4f}  "
                        f"sd = {np.std(slow_at_offset):.4f}\n")
            if corr_data is not None and len(corr_data['dur_prec']) > 2:
                r_p, _ = stats.pearsonr(corr_data['preceding'],
                                        corr_data['dur_prec'])
                f.write(f"  R (precedente): {r_p:.4f}\n")
            if corr_data is not None and len(corr_data['dur_foll']) > 2:
                r_f, _ = stats.pearsonr(corr_data['following'],
                                        corr_data['dur_foll'])
                f.write(f"  R (seguinte):   {r_f:.4f}\n")

        print(f"[OK] TXT salvo em: {txt_path}")

    plt.show()


if __name__ == "__main__":
    main()