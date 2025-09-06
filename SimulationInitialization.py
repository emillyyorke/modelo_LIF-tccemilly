# SimulationInitialization.py
# Sinapses excitatórias all-to-all (sem autapses) com/sem STDP

import numpy as np
from brian2 import Synapses, clip, ms
from SimulationParameters import (
    TAU_PRE, TAU_POST, W_MAX, W_MIN, D_A_PRE, D_A_POST,
    STDP_MODE, USE_STDP, W_INIT_STATIC, N_NEURONS
)

def all2all_no_autapse_indices(N):
    I_grid, J_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask = (I_grid != J_grid)
    I = I_grid[mask].astype(int).ravel()
    J = J_grid[mask].astype(int).ravel()
    return I, J

# Converte tempos e pesos para números puros (num*unidade nas strings)
from brian2 import ms
TAU_PRE_ms  = float(TAU_PRE/ms)
TAU_POST_ms = float(TAU_POST/ms)
W_MAX_f     = float(W_MAX)
W_MIN_f     = float(W_MIN)
D_A_PRE_f   = float(D_A_PRE)
D_A_POST_f  = float(D_A_POST)

synapse_model = f'''
w : 1
dApre/dt  = -Apre / ({TAU_PRE_ms}*ms) : 1 (event-driven)
dApost/dt = -Apost / ({TAU_POST_ms}*ms) : 1 (event-driven)
'''

# Bi & Poo (aditivo) — com clamp [W_MIN, W_MAX]
on_pre_additive  = f'''
g_e_post += w
Apre += {D_A_PRE_f}
w = clip(w + Apost, {W_MIN_f}, {W_MAX_f})
'''
on_post_additive = f'''
Apost += {D_A_POST_f}
w = clip(w + Apre, {W_MIN_f}, {W_MAX_f})
'''

# Van Rossum (LTD multiplicativa) — com clamp [W_MIN, W_MAX]
on_pre_weightdep  = f'''
g_e_post += w
Apre += {D_A_PRE_f}
w = clip(w + (Apost * w / {W_MAX_f}), {W_MIN_f}, {W_MAX_f})
'''
on_post_weightdep = f'''
Apost += {D_A_POST_f}
w = clip(w + Apre, {W_MIN_f}, {W_MAX_f})
'''

def make_excitatory_synapses(neurons):
    I, J = all2all_no_autapse_indices(N_NEURONS)

    if not USE_STDP:
        # --- versão ESTÁTICA (sem STDP): peso inicial = 1.0 ---
        syn = Synapses(
            neurons, neurons,
            model='w:1',
            on_pre='g_e_post += w',
            method='euler'
        )
        syn.connect(i=I, j=J)
        syn.w = W_INIT_STATIC  # pedido: 1.0
        return syn

    # --- versão COM STDP ---
    if STDP_MODE == "additive":
        on_pre, on_post = on_pre_additive, on_post_additive
    else:
        on_pre, on_post = on_pre_weightdep, on_post_weightdep

    syn = Synapses(
        neurons, neurons,
        model=synapse_model,
        on_pre=on_pre,
        on_post=on_post,
        method='euler',
        namespace={'clip': clip}
    )
    syn.connect(i=I, j=J)

    # Pedido: pesos iniciais ~ Uniform(0.5, 1.5), clamp automático garantido nas regras
    syn.w = np.random.uniform(0.5, 1.5, size=len(I))
    return syn
