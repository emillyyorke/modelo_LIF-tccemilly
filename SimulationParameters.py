# SimulationParameters.py
# Parâmetros globais do modelo LIF + STDP (Brian2) — runtime puro

from brian2 import second, ms, mV

# --------- Rede / Simulação ---------
N_NEURONS = 100
SIM_TIME = 10 * second
DT = 0.1 * ms                 # passo de integração
RANDOM_SEED = 42              # reprodutibilidade

# --------- Neurônio LIF ---------
TAU_M = 10 * ms               # constante de tempo de membrana
E_L = -70 * mV                # potencial de fuga (repouso)
V_THR = -50 * mV              # limiar de disparo
V_RESET = -60 * mV            # potencial de reset
REFRAC = 2 * ms               # período refratário absoluto
TAU_E = 5 * ms                # decaimento da condutância excitatória
E_EXC = 0 * mV                # reversão excitatória (AMPA)

# --------- Corrente externa (será setada explícita no iappInit) ---------
# (mantidas aqui apenas as unidades)
I_EXT_MIN = -2 * mV
I_EXT_MAX = 30 * mV

# --------- STDP ---------
TAU_PRE = 20 * ms
TAU_POST = 20 * ms

# Limites e escala de peso
W_MIN = 0.0
W_MAX = 2.0

# Amplitudes base (ajuste fino fica a seu critério)
D_A_PRE  = 0.01 * W_MAX       # LTP
D_A_POST = -(0.01 * W_MAX) * (TAU_PRE/TAU_POST) * 1.05  # LTD com leve viés

# "additive" (Bi & Poo) ou "weight_dep" (Van Rossum)
STDP_MODE = "additive"

# --------- Ligar/Desligar STDP ---------
USE_STDP = False               # mude para False para rodar sem STDP
W_INIT_STATIC = 1.0           # << pedido: sem STDP peso inicial = 1.0


