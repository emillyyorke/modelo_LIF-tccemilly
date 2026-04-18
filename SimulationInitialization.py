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
        # gbar_theta HETEROGÊNEO por neurônio (Tabak escolhe aleatoriamente
        # para impedir que a adaptação homogenize a rede)
        G.g_theta_inc = 'gbar_theta_min + rand() * (gbar_theta_max - gbar_theta_min)'
    elif FEEDBACK_MODE == 'depression':
        G.ge = 0 * nS
        G.s  = 1.0

    G.I_app = make_iapp()

    return G


def make_synapses(G):
    """
    Cria as sinapses.

    - Modo depressão: on_pre usa  ge_post += w * gbar_syn * s_pre
    - Modo adaptação: on_pre usa  ge_post += w * gbar_syn

    STDP:
    - Se STDP_MODE == 'batch': sinapses estáticas com variável w.
      As atualizações de peso são feitas externamente por batch_stdp.py.
    - Se STDP_MODE == 'event_driven': traces apre/apost (Brian2 nativo).
      AVISO: causa viés LTD em redes episódicas.
    """

    # ---------- String de transmissão sináptica ----------
    if FEEDBACK_MODE == 'depression':
        syn_transmit = 'ge_post += w * gbar_syn * s_pre'
    else:  # adaptation
        syn_transmit = 'ge_post += w * gbar_syn'

    # ---------- Escolher modelo de STDP ----------
    if STDP_ENABLED and STDP_MODE == 'event_driven':
        # AVISO: Event-driven STDP não é recomendado para redes episódicas!
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
    if STDP_ENABLED:
        S.w = 'rand() * (W_INIT_MAX - W_INIT_MIN) + W_INIT_MIN'
    else:
        S.w = W_INIT_FIXED

    return S


def make_monitors(G, S):
    """
    Cria monitores de spikes, taxa, pesos e (se depressão) variável s.
    Retorna (spikemon, ratemon, wmon, smon).
    """
    spk  = SpikeMonitor(G, name='spikemon')
    rate = PopulationRateMonitor(G, name='ratemon')

    # Monitor de pesos (apenas se STDP ativo)
    nrec = min(N_W_SAMPLES, S.N)
    idx  = np.random.RandomState(123).choice(S.N, size=nrec, replace=False)
    wmon = (StateMonitor(S, 'w', record=idx, dt=W_MON_DT, name='wmon')
            if STDP_ENABLED else None)

    # Monitor da variável lenta (depressão: s, adaptação: ga)
    if FEEDBACK_MODE == 'depression':
        slow_mon = StateMonitor(G, 's', record=True, dt=1*ms, name='smon')
    elif FEEDBACK_MODE == 'adaptation':
        slow_mon = StateMonitor(G, 'ga', record=True, dt=1*ms, name='gamon')
    else:
        slow_mon = None

    return spk, rate, wmon, slow_mon
