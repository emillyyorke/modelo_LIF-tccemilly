# SimulationInitialization.py
from brian2 import *
import os
import numpy as np
from SimulationParameters import *
from iappInit import make_iapp

_RESULTS_DIR = "./results"


def make_neurons():
    """
    Cria o NeuronGroup com equações que dependem de FEEDBACK_MODE.

    - 'adaptation': inclui variável ga (corrente de adaptação lenta).
    - 'depression': inclui variável s (disponibilidade sináptica pré-sináptica).
    """
    defaultclock.dt = DT

    if FEEDBACK_MODE == 'adaptation':
        eqs = '''
        dv/dt = ( -gL*(v - EL) - ge*(v - V_syn) - ga*(v - V_theta) + I_app ) / Cm : volt (unless refractory)
        dge/dt = -ge / tau_e : siemens
        dga/dt = -ga / tau_a : siemens
        g_theta_inc : siemens (constant)
        I_app : amp
        '''
        reset_str = 'v = V_reset; ga += g_theta_inc'

    elif FEEDBACK_MODE == 'depression':
        eqs = '''
        dv/dt = ( -gL*(v - EL) - ge*(v - V_syn) + I_app ) / Cm : volt (unless refractory)
        dge/dt = -ge / tau_e : siemens
        ds/dt  = (1.0 - s) / tau_s_rec : 1
        I_app : amp
        '''
        reset_str = 'v = V_reset; s = s * (1.0 - delta_dep)'
    else:
        raise ValueError(f"FEEDBACK_MODE inválido: '{FEEDBACK_MODE}'")

    G = NeuronGroup(
        N, eqs,
        threshold='v > V_th',
        reset=reset_str,
        refractory=T_ref,
        method='euler',
        name='neurongroup'
    )

    # ---- Estados iniciais ----
    G.v = EL + (V_th - EL) * 0.2

    if FEEDBACK_MODE == 'adaptation':
        G.ge = 0 * nS
        G.ga = 0 * nS

        # Heterogeneidade por neurônio.
        # Mantido como estava no seu modelo.
        G.g_theta_inc = 'gbar_theta_min + rand() * (gbar_theta_max - gbar_theta_min)'

    elif FEEDBACK_MODE == 'depression':
        G.ge = 0 * nS
        G.s  = 1.0

    G.I_app = make_iapp()

    return G


def make_synapses(G):
    """
    Cria as sinapses.

    - Modo depressão:
      on_pre usa ge_post += w * gbar_syn * s_pre

    - Modo adaptação:
      on_pre usa ge_post += w * gbar_syn

    STDP:
    - Se STDP_MODE == 'batch':
      sinapses estáticas com variável w.
      As atualizações de peso são feitas externamente por batch_stdp.py.

    - Se STDP_MODE == 'event_driven':
      traces apre/apost do Brian2.
      AVISO: pode causar viés LTD em redes episódicas.
    """

    # ---------- String de transmissão sináptica ----------
    if FEEDBACK_MODE == 'depression':
        syn_transmit = 'ge_post += w * gbar_syn * s_pre'
    else:
        syn_transmit = 'ge_post += w * gbar_syn'

    # ---------- Escolher modelo de STDP ----------
    if STDP_ENABLED and STDP_MODE == 'event_driven':
        print("[AVISO] Usando STDP event-driven — pode causar viés LTD e morte da rede.")

        model_str = '''
        w : 1
        dapre/dt  = -apre / tau_pre  : 1 (event-driven)
        dapost/dt = -apost / tau_post : 1 (event-driven)
        '''

        on_pre_str = f'''
        {syn_transmit}
        apre += A_LTP
        w = clip(w + eta * apost, W_MIN, W_MAX)
        '''

        on_post_str = '''
        apost += A_LTD
        w = clip(w + eta * apre, W_MIN, W_MAX)
        '''

        syn_name = 'synapses_stdp_ed'

    elif STDP_ENABLED and STDP_MODE == 'batch':
        # Batch STDP: sinapses com peso w mas SEM traces.
        # As atualizações são feitas por batch_stdp.apply_batch_stdp().
        model_str = 'w : 1'
        on_pre_str = syn_transmit
        on_post_str = None
        syn_name = 'synapses_stdp_batch'

    else:
        # STDP desabilitado — sinapse estática
        model_str = 'w : 1'
        on_pre_str = syn_transmit
        on_post_str = None
        syn_name = 'synapses_static'

    # ---------- Criar objeto Synapses ----------
    if on_post_str is not None:
        S = Synapses(
            G, G,
            model=model_str,
            on_pre=on_pre_str,
            on_post=on_post_str,
            method='euler',
            name=syn_name
        )
    else:
        S = Synapses(
            G, G,
            model=model_str,
            on_pre=on_pre_str,
            method='euler',
            name=syn_name
        )

    # ---------- Conectividade ----------
    if AUTAPSES:
        S.connect(p=P_CONNECT)
    else:
        S.connect(condition='i != j', p=P_CONNECT)

    S.delay = DELAY

    # ---------- Inicialização dos pesos ----------
    #
    # Antes:
    #   - com STDP: pesos uniformes entre W_INIT_MIN e W_INIT_MAX
    #   - sem STDP: todos os pesos fixos em 1.0
    #
    # Agora:
    #   - os pesos são SEMPRE inicializados de forma uniforme;
    #   - a distribuição fica espalhada no tempo 0;
    #   - a média é normalizada para 1.0, preservando a força média antiga;
    #   - a seed fixa garante reprodutibilidade.
    #
    # Isso permite que o histograma inicial não fique em uma barra só,
    # mas também evita mudar demais o comportamento calibrado da rede.

    n_synapses = len(S.w[:])

    if RANDOM_SEED is not None:
        rng = np.random.RandomState(RANDOM_SEED)
    else:
        rng = np.random.RandomState()

    initial_weights = rng.uniform(
        W_INIT_MIN,
        W_INIT_MAX,
        size=n_synapses
    )

    # Força a média dos pesos iniciais a ser exatamente W_INIT_FIXED.
    # Exemplo:
    #   média antes  = 1.0018
    #   média depois = 1.0000
    #
    # Isso ajuda a comparar com a versão antiga em que todos os pesos eram 1.0.
    if W_INIT_NORMALIZE_MEAN:
        initial_weights = initial_weights / np.mean(initial_weights) * W_INIT_FIXED

    # Garante que nenhum peso ultrapasse os limites do modelo.
    initial_weights = np.clip(initial_weights, W_MIN, W_MAX)

    S.w = initial_weights

    print(
        f"[INFO] Pesos iniciais uniformes: "
        f"seed={RANDOM_SEED}, "
        f"min={np.min(initial_weights):.4f}, "
        f"max={np.max(initial_weights):.4f}, "
        f"media={np.mean(initial_weights):.4f}, "
        f"dp={np.std(initial_weights):.4f}"
    )

    return S


def make_monitors(G, S):
    """
    Cria monitores de spikes, taxa, pesos e variável lenta.

    Retorna:
        spikemon, ratemon, wmon, slow_mon
    """
    spk  = SpikeMonitor(G, name='spikemon')
    rate = PopulationRateMonitor(G, name='ratemon')

    # Monitor de pesos.
    # Mantido apenas se STDP estiver ativo.
    nrec = min(N_W_SAMPLES, S.N)

    idx = np.random.RandomState(123).choice(
        S.N,
        size=nrec,
        replace=False
    )

    wmon = (
        StateMonitor(S, 'w', record=idx, dt=W_MON_DT, name='wmon')
        if STDP_ENABLED else None
    )

    # Monitor da variável lenta.
    if FEEDBACK_MODE == 'depression':
        slow_mon = StateMonitor(G, 's', record=True, dt=1*ms, name='smon')
    elif FEEDBACK_MODE == 'adaptation':
        slow_mon = StateMonitor(G, 'ga', record=True, dt=1*ms, name='gamon')
    else:
        slow_mon = None

    return spk, rate, wmon, slow_mon
