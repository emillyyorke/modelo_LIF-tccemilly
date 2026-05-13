#!/usr/bin/env python3
"""
relatorio_metricas.py
=====================
Gera um relatório de métricas no mesmo formato adotado pelo modelo
Hodgkin-Huxley do PIBIC (Vieira & Blanco, 2025), para permitir
comparação direta entre os dois modelos.

Correções:
  - Calcula dt_ms automaticamente a partir de rate_t.npy.
  - Evita o erro de T_LEPIS 10x maior quando DT = 0.1 ms.
  - Usa detecção de episódios menos sensível a microeventos residuais.
  - Salva também T_LEPIS no arquivo tlepis.txt.

Uso:
  python relatorio_metricas.py --dir ./results/DEPRESSION_STDP_BATCH_...
"""
import argparse
import os
import numpy as np


def _calcular_dt_ms(rate_t):
    """
    Calcula o passo temporal real do vetor rate_t, em ms.
    """
    rate_t = np.asarray(rate_t, dtype=float)

    if len(rate_t) > 1:
        return float(np.median(np.diff(rate_t)))

    return 1.0


def _detectar_episodios(
    activity,
    threshold_frac=0.35,
    min_episode_ms=80,
    min_gap_ms=300,
    dt_ms=1.0,
    min_peak_frac=0.25
):
    """
    Detecta episódios de atividade populacional.

    Parâmetros principais:
    - threshold_frac:
        controla o limiar baseado nos percentis da atividade.
        Valores maiores tornam a detecção menos sensível.

    - min_episode_ms:
        duração mínima para considerar um evento como episódio.

    - min_gap_ms:
        intervalo mínimo entre episódios.
        Eventos muito próximos são mesclados.

    - min_peak_frac:
        filtra episódios cujo pico é muito pequeno em relação ao pico global.
        Isso evita contar pequenas oscilações residuais no final como episódios.
    """
    a = np.asarray(activity, dtype=float)

    if len(a) == 0:
        return []

    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    thr = lo + threshold_frac * (hi - lo)

    peak_global = float(np.max(a))
    min_peak = min_peak_frac * peak_global

    above = a > thr
    diff = np.diff(above.astype(int))

    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1

    if len(a) > 0 and above[0]:
        onsets = np.concatenate([[0], onsets])

    if len(a) > 0 and above[-1]:
        offsets = np.concatenate([offsets, [len(a) - 1]])

    n_ep = min(len(onsets), len(offsets))

    episodes = []

    for i in range(n_ep):
        on_i = int(onsets[i])
        off_i = int(offsets[i])

        if off_i <= on_i:
            continue

        dur_ms = (off_i - on_i) * dt_ms
        peak = float(np.max(a[on_i:off_i + 1]))

        if dur_ms >= min_episode_ms and peak >= min_peak:
            episodes.append({
                'onset_ms': on_i * dt_ms,
                'offset_ms': off_i * dt_ms,
                'duration_ms': dur_ms,
                'peak_hz': peak,
            })

    # Mesclar episódios muito próximos
    merged = []

    for ep in episodes:
        if merged and (ep['onset_ms'] - merged[-1]['offset_ms']) < min_gap_ms:
            merged[-1]['offset_ms'] = ep['offset_ms']
            merged[-1]['duration_ms'] = ep['offset_ms'] - merged[-1]['onset_ms']
            merged[-1]['peak_hz'] = max(merged[-1]['peak_hz'], ep['peak_hz'])
        else:
            merged.append(ep)

    return merged


def _salvar_tlepis(results_dir, episodes, sim_time_s, dt_ms):
    """
    Salva o tempo do último episódio em tlepis.txt.
    """
    if episodes:
        last_ep = episodes[-1]
        tlepis_ms = float(last_ep['onset_ms'])
        tlepis_s = tlepis_ms / 1000.0
        n_ep = len(episodes)
    else:
        last_ep = None
        tlepis_ms = 0.0
        tlepis_s = 0.0
        n_ep = 0

    linhas = [
        "# Tempo do último episódio detectado na simulação.",
        "# T_LEPIS = onset do último episódio (ms / s).",
        "# Se N_EPISODES = 0, nenhum episódio foi detectado.",
        f"DT_USADO_MS      = {dt_ms:.6f}",
        f"T_LEPIS_MS       = {tlepis_ms:.3f}",
        f"T_LEPIS_S        = {tlepis_s:.6f}",
        f"N_EPISODES       = {n_ep}",
        f"SIM_TIME_S       = {sim_time_s:.3f}",
    ]

    if last_ep is not None:
        linhas += [
            f"LAST_EP_ONSET_MS  = {last_ep['onset_ms']:.3f}",
            f"LAST_EP_OFFSET_MS = {last_ep['offset_ms']:.3f}",
            f"LAST_EP_DUR_MS    = {last_ep['duration_ms']:.3f}",
            f"LAST_EP_PEAK_HZ   = {last_ep['peak_hz']:.6f}",
        ]

    out_path = os.path.join(results_dir, "tlepis.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    return {
        'tlepis_ms': tlepis_ms,
        'tlepis_s': tlepis_s,
        'n_episodes': n_ep,
        'path': out_path,
    }


def gerar_relatorio(results_dir, verbose=True):
    """
    Gera o relatório de métricas a partir dos arquivos .npy salvos
    em results_dir e escreve em relatorio_metricas.txt.
    Também atualiza tlepis.txt.
    """
    spike_i_path = os.path.join(results_dir, "spike_i.npy")
    rate_t_path = os.path.join(results_dir, "rate_t.npy")
    rate_hz_path = os.path.join(results_dir, "rate_hz.npy")

    if not all(os.path.exists(p) for p in [spike_i_path, rate_t_path, rate_hz_path]):
        if verbose:
            print(
                f"[AVISO] Arquivos .npy não encontrados em {results_dir}. "
                f"Relatório de métricas não gerado."
            )
        return None

    spike_i = np.load(spike_i_path)
    rate_t = np.load(rate_t_path)      # ms
    rate_hz = np.load(rate_hz_path)    # Hz

    dt_ms = _calcular_dt_ms(rate_t)

    params_path = os.path.join(results_dir, "params.txt")
    params = {}

    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    params[k.strip()] = v.strip()

    N = int(params.get("N", 100))

    sim_time_ms = float(rate_t[-1] - rate_t[0]) if len(rate_t) > 1 else 0.0
    sim_time_s = sim_time_ms / 1000.0

    n_spikes = len(spike_i)
    taxa_media = n_spikes / (N * sim_time_s) if sim_time_s > 0 else 0.0

    episodes = _detectar_episodios(
        rate_hz,
        dt_ms=dt_ms,
        threshold_frac=0.35,
        min_episode_ms=80,
        min_gap_ms=300,
        min_peak_frac=0.25
    )

    n_episodes = len(episodes)

    if n_episodes > 0:
        duracoes_s = np.array([ep['duration_ms'] / 1000.0 for ep in episodes])
        duracao_media = float(np.mean(duracoes_s))
        duracao_std = float(np.std(duracoes_s))
    else:
        duracao_media = 0.0
        duracao_std = 0.0

    freq_episodios = n_episodes / sim_time_s if sim_time_s > 0 else 0.0

    tlepis_info = _salvar_tlepis(
        results_dir=results_dir,
        episodes=episodes,
        sim_time_s=sim_time_s,
        dt_ms=dt_ms
    )

    linhas = []
    linhas.append("=" * 50)
    linhas.append("RELATÓRIO DE MÉTRICAS DA SIMULAÇÃO LIF")
    linhas.append("=" * 50)
    linhas.append("")
    linhas.append("--- Configuração da Simulação ---")
    linhas.append(f"Número de Neurônios: {N}")
    linhas.append(f"Tipo de Neurônio: Leaky Integrate-and-Fire (LIF)")
    linhas.append(f"Modo de feedback lento: {params.get('FEEDBACK_MODE', 'N/A')}")
    linhas.append(f"STDP habilitada: {params.get('STDP_ENABLED', 'N/A')}")

    if params.get('STDP_ENABLED', '').lower() == 'true':
        linhas.append(f"  Modo STDP: {params.get('STDP_MODE', 'N/A')}")
        linhas.append(f"  A_LTP = {params.get('A_LTP', 'N/A')}")
        linhas.append(f"  A_LTD = {params.get('A_LTD', 'N/A')}")

    linhas.append(f"Tempo Total de Simulação: {sim_time_s:.2f} s ({sim_time_ms:.0f} ms)")
    linhas.append(f"dt usado na detecção: {dt_ms:.6f} ms")
    linhas.append("")
    linhas.append("--- Métricas de Atividade da Rede ---")
    linhas.append(f"Taxa de Disparo Média da Rede: {taxa_media:.4f} Hz")
    linhas.append(f"Número Total de Episódios (Sincronização): {n_episodes}")
    linhas.append(
        f"Duração Média dos Episódios: {duracao_media:.4f} s "
        f"(desvio padrão: {duracao_std:.4f} s)"
    )
    linhas.append(f"Frequência de Episódios: {freq_episodios:.4f} Hz")
    linhas.append(f"T_LEPIS: {tlepis_info['tlepis_s']:.4f} s")
    linhas.append("")
    linhas.append("--- Parâmetros da Detecção de Episódios ---")
    linhas.append("threshold_frac = 0.35")
    linhas.append("min_episode_ms = 80")
    linhas.append("min_gap_ms = 300")
    linhas.append("min_peak_frac = 0.25")
    linhas.append("")
    linhas.append("--- Fim do Relatório ---")

    texto = "\n".join(linhas)

    out_path = os.path.join(results_dir, "relatorio_metricas.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(texto)

    if verbose:
        print(texto)
        print(f"\n[OK] Relatório salvo em: {out_path}")
        print(f"[OK] T_LEPIS salvo em: {tlepis_info['path']}")

    return {
        'N': N,
        'sim_time_s': sim_time_s,
        'dt_ms': dt_ms,
        'taxa_media_hz': taxa_media,
        'n_episodes': n_episodes,
        'duracao_media_s': duracao_media,
        'duracao_std_s': duracao_std,
        'freq_episodios_hz': freq_episodios,
        'tlepis_s': tlepis_info['tlepis_s'],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera relatório de métricas de uma simulação LIF."
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Pasta de resultados"
    )

    args = parser.parse_args()

    gerar_relatorio(args.dir)