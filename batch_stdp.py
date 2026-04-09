# batch_stdp.py — STDP em lote (batch) para redes episódicas
#
# Motivação: O STDP event-driven do Brian2 cria um viés sistemático de LTD
# em redes episódicas, porque os últimos spikes de cada episódio acumulam
# traces pós-sinápticos negativos sem compensação LTP correspondente.
# Isso faz os pesos caírem até W_MIN e a rede morre.
#
# Solução: Coletar spikes em janelas temporais, calcular todos os pares
# causais/anti-causais de uma vez, e aplicar as atualizações em lote
# fora do loop de integração do Brian2.
#
# Baseado na abordagem do modelo de referência HH (STDP.cpp).

import numpy as np
from SimulationParameters import (
    A_LTP, A_LTD, eta,
    tau_pre, tau_post,
    W_MIN, W_MAX,
    N
)

# Converter constantes de tempo para ms (float)
_TAU_PRE_MS  = float(tau_pre / 1e-3)   # ex: 20.0
_TAU_POST_MS = float(tau_post / 1e-3)   # ex: 20.0
_MAX_DT_MS   = 5.0 * max(_TAU_PRE_MS, _TAU_POST_MS)  # janela máxima (~100 ms)


def apply_batch_stdp(S, spike_i, spike_t_ms, t_start_ms, t_end_ms):
    """
    Calcula e aplica atualizações STDP em lote para todos os spikes
    no intervalo [t_start_ms, t_end_ms].

    Parâmetros
    ----------
    S : brian2.Synapses
        Objeto de sinapses (deve ter S.w, S.i, S.j).
    spike_i : array-like
        Índices dos neurônios que dispararam (do SpikeMonitor).
    spike_t_ms : array-like
        Tempos dos spikes em milissegundos.
    t_start_ms : float
        Início do intervalo a processar.
    t_end_ms : float
        Fim do intervalo a processar.

    Retorna
    -------
    dict com estatísticas: mean_dw, std_dw, mean_w, min_w, max_w,
                           n_ltp_pairs, n_ltd_pairs
    """
    spike_i = np.asarray(spike_i)
    spike_t_ms = np.asarray(spike_t_ms)

    # --- Filtrar spikes no intervalo ---
    mask = (spike_t_ms >= t_start_ms) & (spike_t_ms < t_end_ms)
    si = spike_i[mask]
    st = spike_t_ms[mask]

    if len(si) == 0:
        w_arr = np.asarray(S.w)
        return {
            'mean_dw': 0.0, 'std_dw': 0.0,
            'mean_w': float(np.mean(w_arr)),
            'min_w': float(np.min(w_arr)),
            'max_w': float(np.max(w_arr)),
            'n_ltp_pairs': 0, 'n_ltd_pairs': 0
        }

    # --- Agrupar spikes por neurônio ---
    spike_trains = {}
    for neuron_id, t in zip(si, st):
        nid = int(neuron_id)
        if nid not in spike_trains:
            spike_trains[nid] = []
        spike_trains[nid].append(t)

    # Converter listas para arrays numpy (ordenados)
    for nid in spike_trains:
        spike_trains[nid] = np.sort(np.array(spike_trains[nid]))

    # --- Obter índices das sinapses ---
    syn_pre  = np.asarray(S.i)   # índice do neurônio pré para cada sinapse
    syn_post = np.asarray(S.j)   # índice do neurônio pós para cada sinapse
    n_syn = len(syn_pre)

    # --- Calcular Δw para cada sinapse ---
    dw = np.zeros(n_syn, dtype=np.float64)
    n_ltp_total = 0
    n_ltd_total = 0

    # Identificar quais neurônios dispararam (para pular sinapses sem atividade)
    active_neurons = set(spike_trains.keys())

    for syn_idx in range(n_syn):
        pre_id  = int(syn_pre[syn_idx])
        post_id = int(syn_post[syn_idx])

        # Pular se pré OU pós não dispararam neste intervalo
        if pre_id not in active_neurons or post_id not in active_neurons:
            continue

        t_pre  = spike_trains[pre_id]    # array de tempos
        t_post = spike_trains[post_id]   # array de tempos

        # Calcular todas as diferenças dt = t_post - t_pre
        # Usando broadcasting: dt_matrix[i,j] = t_post[i] - t_pre[j]
        dt_matrix = t_post[:, None] - t_pre[None, :]  # (n_post, n_pre)

        # --- Pares causais: dt > 0 (pré antes de pós) → LTP ---
        causal_mask = (dt_matrix > 0) & (dt_matrix < _MAX_DT_MS)
        if np.any(causal_mask):
            dt_causal = dt_matrix[causal_mask]
            ltp_contrib = A_LTP * np.sum(np.exp(-dt_causal / _TAU_PRE_MS))
            dw[syn_idx] += ltp_contrib
            n_ltp_total += int(np.sum(causal_mask))

        # --- Pares anti-causais: dt < 0 (pós antes de pré) → LTD ---
        anti_mask = (dt_matrix < 0) & (dt_matrix > -_MAX_DT_MS)
        if np.any(anti_mask):
            dt_anti = dt_matrix[anti_mask]
            # A_LTD é negativo, dt_anti é negativo → exp(dt/tau) = exp(negativo) < 1
            ltd_contrib = A_LTD * np.sum(np.exp(dt_anti / _TAU_POST_MS))
            dw[syn_idx] += ltd_contrib
            n_ltd_total += int(np.sum(anti_mask))

    # --- Aplicar fator global e atualizar pesos ---
    dw *= eta
    w_arr = np.asarray(S.w)
    w_new = np.clip(w_arr + dw, W_MIN, W_MAX)
    S.w = w_new

    return {
        'mean_dw': float(np.mean(dw)),
        'std_dw': float(np.std(dw)),
        'mean_w': float(np.mean(w_new)),
        'min_w': float(np.min(w_new)),
        'max_w': float(np.max(w_new)),
        'n_ltp_pairs': n_ltp_total,
        'n_ltd_pairs': n_ltd_total
    }