#!/usr/bin/env python3
"""
relatorio_metricas.py
=====================
Gera um relatório de métricas no mesmo formato adotado pelo modelo
Hodgkin-Huxley do PIBIC (Vieira & Blanco, 2025), para permitir
comparação direta entre os dois modelos.

Métricas calculadas:
  - Taxa de Disparo Média da Rede (Hz)
  - Número Total de Episódios (Sincronização)
  - Duração Média dos Episódios (s)
  - Frequência de Episódios (Hz)

Pode ser usado de duas formas:

  1) Manualmente, via linha de comando:
     python relatorio_metricas.py --dir ./results/DEPRESSION_STDP_OFF_...

  2) Automaticamente, importado pelo LIF_EMILLY.py:
     from relatorio_metricas import gerar_relatorio
     gerar_relatorio(results_dir)
"""
import argparse
import os
import numpy as np


def _detectar_episodios(activity, threshold_frac=0.15, min_episode_ms=10,
                        min_gap_ms=50, dt_ms=1.0):
    """
    Versão local da detecção de episódios, idêntica à de plot_tabak_analysis.py.
    Mantida aqui para evitar dependência circular durante a importação.
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
                'onset_ms':   on_i * dt_ms,
                'offset_ms':  off_i * dt_ms,
                'duration_ms': dur,
            })

    # Mesclar episódios muito próximos
    merged = []
    for ep in episodes:
        if merged and (ep['onset_ms'] - merged[-1]['offset_ms']) < min_gap_ms:
            merged[-1]['offset_ms']  = ep['offset_ms']
            merged[-1]['duration_ms'] = ep['offset_ms'] - merged[-1]['onset_ms']
        else:
            merged.append(ep)

    return merged


def gerar_relatorio(results_dir, verbose=True):
    """
    Gera o relatório de métricas a partir dos arquivos .npy salvos
    em results_dir e o escreve em relatorio_metricas.txt na mesma pasta.

    Parâmetros
    ----------
    results_dir : str
        Caminho para a pasta de resultados de uma simulação.
    verbose : bool
        Se True, imprime o relatório no terminal além de salvá-lo.

    Retorna
    -------
    dict com as quatro métricas calculadas, ou None se não houver dados.
    """
    # ---- Carregar dados ----
    spike_i_path = os.path.join(results_dir, "spike_i.npy")
    rate_t_path  = os.path.join(results_dir, "rate_t.npy")
    rate_hz_path = os.path.join(results_dir, "rate_hz.npy")

    if not all(os.path.exists(p) for p in [spike_i_path, rate_t_path, rate_hz_path]):
        if verbose:
            print(f"[AVISO] Arquivos .npy nao encontrados em {results_dir}. "
                  f"Relatorio de metricas nao gerado.")
        return None

    spike_i = np.load(spike_i_path)
    rate_t  = np.load(rate_t_path)    # ms
    rate_hz = np.load(rate_hz_path)   # Hz

    # ---- Ler params.txt para pegar N e configuração ----
    params_path = os.path.join(results_dir, "params.txt")
    params = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    params[k.strip()] = v.strip()

    N = int(params.get("N", 100))
    sim_time_ms = float(rate_t[-1] - rate_t[0])
    sim_time_s = sim_time_ms / 1000.0

    # ---- Métrica 1: Taxa de Disparo Média ----
    n_spikes = len(spike_i)
    taxa_media = n_spikes / (N * sim_time_s) if sim_time_s > 0 else 0.0

    # ---- Métricas 2, 3, 4: Episódios ----
    episodes = _detectar_episodios(rate_hz, dt_ms=1.0)
    n_episodes = len(episodes)

    if n_episodes > 0:
        duracoes_s = np.array([ep['duration_ms'] / 1000.0 for ep in episodes])
        duracao_media = float(np.mean(duracoes_s))
        duracao_std = float(np.std(duracoes_s))
    else:
        duracao_media = 0.0
        duracao_std = 0.0

    freq_episodios = n_episodes / sim_time_s if sim_time_s > 0 else 0.0

    # ---- Montar relatório ----
    linhas = []
    linhas.append("=" * 50)
    linhas.append("RELATORIO DE METRICAS DA SIMULACAO LIF")
    linhas.append("=" * 50)
    linhas.append("")
    linhas.append("--- Configuracao da Simulacao ---")
    linhas.append(f"Numero de Neuronios: {N}")
    linhas.append(f"Tipo de Neuronio: Leaky Integrate-and-Fire (LIF)")
    linhas.append(f"Modo de feedback lento: {params.get('FEEDBACK_MODE', 'N/A')}")
    linhas.append(f"STDP habilitada: {params.get('STDP_ENABLED', 'N/A')}")
    if params.get('STDP_ENABLED', '').lower() == 'true':
        linhas.append(f"  Modo STDP: {params.get('STDP_MODE', 'N/A')}")
        linhas.append(f"  A_LTP = {params.get('A_LTP', 'N/A')}")
        linhas.append(f"  A_LTD = {params.get('A_LTD', 'N/A')}")
    linhas.append(f"Tempo Total de Simulacao: {sim_time_s:.2f} s ({sim_time_ms:.0f} ms)")
    linhas.append("")
    linhas.append("--- Metricas de Atividade da Rede ---")
    linhas.append(f"Taxa de Disparo Media da Rede: {taxa_media:.4f} Hz")
    linhas.append(f"Numero Total de Episodios (Sincronizacao): {n_episodes}")
    linhas.append(f"Duracao Media dos Episodios: {duracao_media:.4f} s "
                  f"(desvio padrao: {duracao_std:.4f} s)")
    linhas.append(f"Frequencia de Episodios: {freq_episodios:.4f} Hz")
    linhas.append("")
    linhas.append("--- Fim do Relatorio ---")

    texto = "\n".join(linhas)

    # Salvar
    out_path = os.path.join(results_dir, "relatorio_metricas.txt")
    with open(out_path, "w") as f:
        f.write(texto)

    if verbose:
        print(texto)
        print(f"\n[OK] Relatorio salvo em: {out_path}")

    return {
        'N': N,
        'sim_time_s': sim_time_s,
        'taxa_media_hz': taxa_media,
        'n_episodes': n_episodes,
        'duracao_media_s': duracao_media,
        'duracao_std_s': duracao_std,
        'freq_episodios_hz': freq_episodios,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera relatorio de metricas de uma simulacao LIF."
    )
    parser.add_argument("--dir", required=True, help="Pasta de resultados")
    args = parser.parse_args()
    gerar_relatorio(args.dir)
    