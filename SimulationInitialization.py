# SimulationInitialization.py
from brian2 import *
import numpy as np
from SimulationParameters import *
from iappInit import make_iapp

# Pasta de resultados
_RESULTS_DIR = "./results"


def make_neurons():
    defaultclock.dt = DT

    eqs = '''
    dv/dt = ( -gL*(v-EL) - ge*(v - V_syn) - ga*(v - V_theta) + I_app )/Cm : volt (unless refractory)
    dge/dt = -ge/tau_e : siemens
    dga/dt = -ga/tau_a : siemens
    I_app : amp
    '''

    G = NeuronGroup(
        N, eqs, threshold='v>V_th', reset='v=V_reset; ga += gbar_theta',
        refractory=T_ref, method='euler', name='neurongroup'
    )

    # Estados iniciais
    G.v = EL + (V_th-EL)*0.2    # 20% do caminho ao limiar
    G.ge = 0*nS
    G.ga = 0*nS
    G.I_app = make_iapp()

    return G


def make_synapses(G):
    if STDP_ENABLED:
        S = Synapses(
            G, G,
            model='''
            w : 1
            dapre/dt  = -apre/tau_pre : 1 (event-driven)
            dapost/dt = -apost/tau_post : 1 (event-driven)
            ''',
            on_pre='''
            ge_post += w * gbar_syn
            apre += A_LTP
            w = clip(w + eta*apost, W_MIN, W_MAX)
            ''',
            on_post='''
            apost += A_LTD
            w = clip(w + eta*apre, W_MIN, W_MAX)
            ''',
            method='euler',
            name='synapses_stdp'
        )
    else:
        S = Synapses(
            G, G,
            model='w:1',
            on_pre='ge_post += w * gbar_syn',
            method='euler',
            name='synapses_static'
        )

    if AUTAPSES:
        S.connect(p=P_CONNECT)
    else:
        S.connect(condition='i!=j', p=P_CONNECT)

    S.delay = DELAY

    if STDP_ENABLED:
        # ==========================================================
        # MUDANÇA FINAL: Usando a inicialização interna do Brian2
        # ==========================================================
        # A expressão 'rand()' do Brian2 gera um número aleatório diferente para cada sinapse.
        S.w = 'rand() * (W_INIT_MAX - W_INIT_MIN) + W_INIT_MIN'
        # ==========================================================
    else:
        S.w = W_INIT_FIXED

    return S

def make_monitors(G, S):
    spk = SpikeMonitor(G, name='spikemon')
    rate = PopulationRateMonitor(G, name='ratemon')

    # Amostra de pesos (apenas se STDP ativo)
    nrec = min(N_W_SAMPLES, S.N)
    idx = np.random.RandomState(123).choice(S.N, size=nrec, replace=False)
    wmon = StateMonitor(S, 'w', record=idx, dt=W_MON_DT, name='wmon') if STDP_ENABLED else None

    return spk, rate, wmon
