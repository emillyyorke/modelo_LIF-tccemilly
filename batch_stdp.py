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
# Implementação: replica a lógica de applySTDP_OverlappedWindow() do
# modelo de referência HH (STDP.cpp), incluindo o tratamento de fronteiras
# de janela por zona de sobreposição (overlap-ignore).
#
# --- Correspondência com o HH ---
# HH:  windowSize=100ms, step=50ms, overlapIgnore=50ms
#       → chamado a cada 50ms; pares onde AMBOS os spikes estão na
#         zona de sobreposição [windowStart, windowStart+50ms) são
#         ignorados (já foram contados na janela anterior).
#
# LIF: STDP_BATCH_INTERVAL_MS=500ms (stride), _MAX_DT_MS=100ms (corte).
#       → chamado a cada 500ms; estende a coleta de spikes 100ms para
#         trás da fronteira do batch atual e aplica o mesmo overlap-ignore:
#         pares onde AMBOS os spikes estão em [t_start−100ms, t_start)
#         são ignorados (já contados no batch anterior).
#
# Diferenças intencionais mantidas:
#   • Amplitude: A_LTP/A_LTD = 0.0005 (HH: 0.005) — reescalonamento
#     para topologia all-to-all com N=100 neurônios.
#   • W_MIN: 0.5 (HH: 0.0) — piso elevado para evitar morte da rede LIF.
#   • Stride: 500ms (HH: 50ms) — lote maior compatível com loop do Brian2;
#     o overlap-ignore garante que os pares de fronteira não sejam
#     perdidos nem contados duas vezes.

import numpy as np
from SimulationParameters import (
    A_LTP, A_LTD, eta,
    tau_pre, tau_post,
    W_MIN, W_MAX,
    N
)

# Converter constantes de tempo para ms (float)
_TAU_PRE_MS  = float(tau_pre / 1e-3)   # ex: 20.0
_TAU_POST_MS = float(tau_post / 1e-3)  # ex: 20.0
_MAX_DT_MS   = 5.0 * max(_TAU_PRE_MS, _TAU_POST_MS)  # janela máxima = 100 ms


def apply_batch_stdp(S, spike_i, spike_t_ms, t_start_ms, t_end_ms):
    """
    Calcula e aplica atualizações STDP em lote para o intervalo
    [t_start_ms, t_end_ms], com tratamento de fronteira equivalente
    ao applySTDP_OverlappedWindow() do modelo HH de referência.

    Zona de sobreposição: os spikes de [t_start_ms − _MAX_DT_MS, t_start_ms)
    são incluídos na busca de pares, mas pares onde AMBOS os spikes estão
    nessa zona são ignorados (já foram contados no batch anterior).
    Isso garante que pares causais/anti-causais que cruzam a fronteira de
    500ms não sejam perdidos.

    Parâmetros
    ----------
    S : brian2.Synapses
        Objeto de sinapses (deve ter S.w, S.i, S.j).
    spike_i : array-like
        Índices dos neurônios que dispararam (de SpikeMonitor.i).
    spike_t_ms : array-like
        Tempos dos spikes em milissegundos (de SpikeMonitor.t/ms).
    t_start_ms : float
        Início do batch atual (ms).
    t_end_ms : float
        Fim do batch atual (ms).

    Retorna
    -------
    dict com estatísticas: mean_dw, std_dw, mean_w, min_w, max_w,
                           n_ltp_pairs, n_ltd_pairs
    """
    spike_i    = np.asarray(spike_i)
    spike_t_ms = np.asarray(spike_t_ms)

    # --- Zona de sobreposição (overlap-ignore) ---
    # Inclui spikes de até _MAX_DT_MS antes de t_start_ms para capturar
    # pares que cruzam a fronteira do batch.
    overlap_start_ms = t_start_ms - _MAX_DT_MS  # = t_start - 100ms
    is_first_batch   = (t_start_ms == 0.0)

    # Coletar spikes na janela estendida: [overlap_start, t_end)
    mask_ext = (spike_t_ms >= overlap_start_ms) & (spike_t_ms < t_end_ms)
    si_ext   = spike_i[mask_ext]
    st_ext   = spike_t_ms[mask_ext]

    if len(si_ext) == 0:
        w_arr = np.asarray(S.w)
        return {
            'mean_dw': 0.0, 'std_dw': 0.0,
            'mean_w':  float(np.mean(w_arr)),
            'min_w':   float(np.min(w_arr)),
            'max_w':   float(np.max(w_arr)),
            'n_ltp_pairs': 0, 'n_ltd_pairs': 0
        }

    # --- Agrupar spikes por neurônio ---
    spike_trains = {}
    for neuron_id, t in zip(si_ext, st_ext):
        nid = int(neuron_id)
        if nid not in spike_trains:
            spike_trains[nid] = []
        spike_trains[nid].append(t)

    for nid in spike_trains:
        spike_trains[nid] = np.sort(np.array(spike_trains[nid]))

    # --- Índices das sinapses ---
    syn_pre  = np.asarray(S.i)
    syn_post = np.asarray(S.j)
    n_syn    = len(syn_pre)

    dw          = np.zeros(n_syn, dtype=np.float64)
    n_ltp_total = 0
    n_ltd_total = 0

    active_neurons = set(spike_trains.keys())

    for syn_idx in range(n_syn):
        pre_id  = int(syn_pre[syn_idx])
        post_id = int(syn_post[syn_idx])

        if pre_id not in active_neurons or post_id not in active_neurons:
            continue

        t_pre  = spike_trains[pre_id]
        t_post = spike_trains[post_id]

        # dt_matrix[i, j] = t_post[i] − t_pre[j]
        dt_matrix = t_post[:, None] - t_pre[None, :]  # shape (n_post, n_pre)

        # Vetores de tempo para as máscaras de overlap-ignore
        t_post_mat = np.broadcast_to(t_post[:, None], dt_matrix.shape)
        t_pre_mat  = np.broadcast_to(t_pre[None, :],  dt_matrix.shape)

        # Máscara de overlap-ignore: ambos os spikes estão na zona de
        # sobreposição [overlap_start, t_start). Se for o primeiro batch,
        # essa condição nunca é verdadeira (não há batch anterior).
        if is_first_batch:
            in_overlap = np.zeros(dt_matrix.shape, dtype=bool)
        else:
            in_overlap = (t_post_mat < t_start_ms) & (t_pre_mat < t_start_ms)

        # --- Pares causais: dt > 0 (pré antes de pós) → LTP ---
        causal_mask = (dt_matrix > 0) & (dt_matrix < _MAX_DT_MS) & (~in_overlap)
        if np.any(causal_mask):
            dt_causal   = dt_matrix[causal_mask]
            ltp_contrib = A_LTP * np.sum(np.exp(-dt_causal / _TAU_PRE_MS))
            dw[syn_idx] += ltp_contrib
            n_ltp_total += int(np.sum(causal_mask))

        # --- Pares anti-causais: dt < 0 (pós antes de pré) → LTD ---
        # A_LTD é negativo; dt_anti < 0 → exp(dt_anti/τ) ∈ (0,1)
        # → contribuição é negativa (LTD), igual ao HH: −aLTD·exp(dt/τ)
        anti_mask = (dt_matrix < 0) & (dt_matrix > -_MAX_DT_MS) & (~in_overlap)
        if np.any(anti_mask):
            dt_anti     = dt_matrix[anti_mask]
            ltd_contrib = A_LTD * np.sum(np.exp(dt_anti / _TAU_POST_MS))
            dw[syn_idx] += ltd_contrib
            n_ltd_total += int(np.sum(anti_mask))

    # --- Aplicar fator global e atualizar pesos ---
    dw   *= eta
    w_arr = np.asarray(S.w)
    w_new = np.clip(w_arr + dw, W_MIN, W_MAX)
    S.w   = w_new

    return {
        'mean_dw':     float(np.mean(dw)),
        'std_dw':      float(np.std(dw)),
        'mean_w':      float(np.mean(w_new)),
        'min_w':       float(np.min(w_new)),
        'max_w':       float(np.max(w_new)),
        'n_ltp_pairs': n_ltp_total,
        'n_ltd_pairs': n_ltd_total
    }