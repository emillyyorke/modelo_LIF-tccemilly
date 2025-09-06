# iappInit.py
# Define o NeuronGroup (LIF) e inicializações (I_ext explícita)

import numpy as np
from brian2 import NeuronGroup, mV, ms
from SimulationParameters import (
    TAU_M, E_L, V_THR, V_RESET, REFRAC,
    TAU_E, E_EXC, N_NEURONS
)

# Converte constantes para "número * unidade"
E_L_mV      = float(E_L/mV)
E_EXC_mV    = float(E_EXC/mV)
V_THR_mV    = float(V_THR/mV)
V_RESET_mV  = float(V_RESET/mV)
TAU_M_ms    = float(TAU_M/ms)
TAU_E_ms    = float(TAU_E/ms)

# Equações do neurônio LIF (sem depender de namespace)
eqs_neuron = f'''
dv/dt = ((({E_L_mV}*mV) - v) + g_e*((({E_EXC_mV}*mV) - v)) + I_ext) / ({TAU_M_ms}*ms) : volt (unless refractory)
dg_e/dt = -g_e / ({TAU_E_ms}*ms) : 1
I_ext : volt
'''

# ----- iapp explícito (100 valores) -----
_IAPP_LIST = np.array([
-2.18619,-2.57576,-3.41212,-3.71471,-5.39247, 2.96518, 3.7141,  4.12336, 2.06839,-3.20933,
-8.78277,-8.15378, 1.66234,-5.6827, -4.08322,-2.69158,-2.59407,-1.73528, 2.83197, 3.40739,
-5.76464,-9.56374,-5.77334, 2.82739,-6.84317,-6.86834,-0.258034,-0.471816,-3.84289,-3.70281,
-5.67721, 1.8244, -8.66329, 3.99792,-6.82577,-5.59847, 1.23707,-3.68862,-4.54604, 4.11283,
-0.170141,0.237281,-9.07758,-8.10526,-8.86425,-9.28953,-3.72341, 0.655232,-5.05325,-4.68108,
-9.62691,-8.53282,-7.38884, 3.81252, 0.0715659,-7.08075,-1.63961,-3.99533,-4.09009,-4.03607,
-5.86123, 4.15403,-0.667745,-5.89236,-1.58879,-1.57781, 0.690023,-8.71548,-8.09061,-8.4962,
-0.269478, 2.26341, 0.794397, 1.01733,-7.20252, 2.92398, 0.197913,-4.24802,-6.54378,-9.74914,
-1.38554, 4.37422, 0.769219, 4.39207,-3.88913, 2.66762, 1.93197, 2.61223,-3.17728,-2.08136,
-1.50227,-7.49413,-6.292,   1.53645,-7.09815,-0.705741,1.02374,-6.51036,-0.490585,4.55962
], dtype=float)

def make_neurons():
    if len(_IAPP_LIST) != N_NEURONS:
        raise ValueError(f"IAPP tem {len(_IAPP_LIST)} valores, mas N_NEURONS={N_NEURONS}")

    neurons = NeuronGroup(
        N_NEURONS,
        eqs_neuron,
        threshold=f'v > {V_THR_mV}*mV',
        reset=f'v = {V_RESET_mV}*mV',
        refractory=REFRAC,
        method='euler'
    )
    # v inicial entre E_L e V_THR
    neurons.v = (E_L_mV + (V_THR_mV - E_L_mV) * np.random.rand(N_NEURONS)) * mV
    # I_ext explícito (em mV)
    neurons.I_ext = _IAPP_LIST * mV
    return neurons
