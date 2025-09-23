# SimulationInitialization.py — atualizado p/ capturar pesos e gerar w_all.npz
from brian2 import *
import numpy as np
from SimulationParameters import *
from iappInit import make_iapp

# >>> NOVO: capturador de pesos
import weights_history as wh

# Se quiser usar uma pasta de resultados diferente, troque aqui:
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
        # Sem STDP: sinapse mínima, apenas transmite
        S = Synapses(
            G, G,
            model='w:1',
            on_pre='ge_post += w * gbar_syn',
            method='euler',
            name='synapses_static'
        )

    # Conexões: completo sem autapses
    if AUTAPSES:
        S.connect(p=P_CONNECT)
    else:
        S.connect(condition='i!=j', p=P_CONNECT)

    S.delay = DELAY

    # Pesos iniciais
    if STDP_ENABLED:
        np.random.seed(7)
        S.w = np.random.uniform(W_INIT_MIN, W_INIT_MAX, size=S.N)
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


# ========= NOVO: operação periódica para tirar "fotos" de todos os pesos =========
def make_weight_history_op(S):
    """
    Cria um NetworkOperation que, a cada SNAPSHOT_DT_MS, captura S.w (todos os pesos)
    e manda para o weights_history salvar depois em w_all.npz.
    Retorna:
      - snap_op: NetworkOperation (ou None, se SAVE_WEIGHTS_HISTORY=False)
    """
    # inicializa o coletor (uma única vez, antes do run)
    wh.init(results_dir=_RESULTS_DIR,
            snapshot_dt_ms=float(SNAPSHOT_DT_MS),
            enabled=bool(SAVE_WEIGHTS_HISTORY))

    if not SAVE_WEIGHTS_HISTORY:
        return None

    @network_operation(dt=SNAPSHOT_DT_MS*ms, name='weights_snapshot_op')
    def snap_op(t):
        # 't' é o tempo atual do simulador com unidade; convertemos para ms (float)
        try:
            t_ms = float(t/ms)  # Brian2: t/ms -> valor em milissegundos
            # S.w[:] devolve um vetor 1D com TODOS os pesos (dimensionaless)
            w_now = np.asarray(S.w[:], dtype=np.float32)
            # captura (sem máscara; se sua rede for esparsa, já está codificada em S)
            wh.capture(t_ms, w_now)
        except Exception as e:
            print(f"[weights_history][warn] snapshot ignorado: {e}")

    return snap_op


# ========= NOVO: finalize para ser chamado após run() no script principal =========
def finalize_weight_history():
    """Chame no final do seu script (depois de run(...)) para salvar w_all.npz."""
    try:
        wh.finalize()
    except Exception as e:
        print(f"[weights_history][warn] finalize falhou: {e}")
