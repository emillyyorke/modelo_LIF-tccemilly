# SimulationParameters.py
from brian2 import *

# ===========================
# Tempo de simulação
# ===========================
SIM_TIME = 480*second
DT       = 0.1*ms

# ===========================
# Tamanho da rede
# ===========================
N = 100                   # Tabak usa N=100; rede menor = mais ruído estocástico
P_CONNECT = 1.0           # completamente conectada (all-to-all)
AUTAPSES = False          # sem autoconexões

# ===========================
# Modo de feedback lento (escolha aqui)
# ===========================
# 'adaptation'  → corrente lenta de adaptação (ga, theta)
# 'depression'  → depressão sináptica pré-sináptica (s_j), Tabak et al. 2010
FEEDBACK_MODE = 'depression'  # <--- comute aqui

# ===========================
# LIF — parâmetros comuns a ambos os modos
# ===========================
Cm      = 200*pF          # Capacitância
gL      = 10*nS           # Condutância de vazamento  -> tau_m = Cm/gL = 20 ms
EL      = -70*mV          # Potencial de repouso
V_th    = -50*mV          # limiar de spike
V_reset = -60*mV          # reset pós-spike

# Sinapse excitatória — reversal
V_syn   = 0*mV            # reversal da sinapse excitatória

# ===========================
# Parâmetros POR MODO — Adaptação celular  (Tabak Table 1, col. adaptation)
# ===========================
T_ref_adapt    = 5*ms
gbar_syn_adapt = 0.14*nS
tau_e_adapt    = 10*ms
V_theta        = -80*mV
tau_a          = 2500*ms

# Incremento de ga por spike — HETEROGÊNEO por neurônio
gbar_theta_min = 0.05*nS
gbar_theta_max = 0.25*nS
gbar_theta     = 0.5 * (gbar_theta_min + gbar_theta_max)

# Corrente externa — adaptação
I_min_adapt = 80*pA
I_max_adapt = 340*pA

# ===========================
# Parâmetros POR MODO — Depressão sináptica
# ===========================
T_ref_dep      = 5*ms
gbar_syn_dep   = 0.28*nS
tau_e_dep      = 10*ms
tau_s_rec      = 5000*ms
delta_dep      = 0.025

I_DIST_DEP     = 'uniform'
I_min_dep      = 30*pA
I_max_dep      = 230*pA
I_mean_dep     = 200*pA
I_std_dep      = 50*pA

# ===========================
# Atalhos — resolvidos em função do modo ativo
# ===========================
if FEEDBACK_MODE == 'depression':
    T_ref    = T_ref_dep
    gbar_syn = gbar_syn_dep
    tau_e    = tau_e_dep
elif FEEDBACK_MODE == 'adaptation':
    T_ref    = T_ref_adapt
    gbar_syn = gbar_syn_adapt
    tau_e    = tau_e_adapt
else:
    raise ValueError(f"FEEDBACK_MODE inválido: '{FEEDBACK_MODE}'.")

# ===========================
# STDP — Configuração
# ===========================
STDP_ENABLED = True       # <--- comute aqui

# Modo de STDP:
#   'batch'        → STDP em lote (batch_stdp.py) — RECOMENDADO para redes episódicas
#   'event_driven' → STDP event-driven do Brian2 — NÃO RECOMENDADO (viés LTD)
STDP_MODE = 'batch'

# Janelas temporais
tau_pre  = 20*ms
tau_post = 20*ms

# Magnitudes
A_LTP =  0.009          # Δw por par causal (pré-antes-pós)
A_LTD = -0.009          # Δw por par anti-causal (pós-antes-pré)
eta   =  1.0            # fator global

# ===========================
# Pesos sinápticos
# ===========================

# Limites absolutos permitidos pelo STDP
W_MIN = 0.0
W_MAX = 2.0

# Valor médio de referência.
W_INIT_FIXED = 1.0

# Inicialização uniforme próxima de 1.0.
W_INIT_MIN = 0.5
W_INIT_MAX = 1.5

# Normaliza os pesos iniciais para média exatamente igual a W_INIT_FIXED.
W_INIT_NORMALIZE_MEAN = False

# Seed fixa para pesos iniciais.
# Para cada execução sair diferente, use:
#   RANDOM_SEED = None
RANDOM_SEED = 99

# Intervalo de aplicação do batch STDP
STDP_BATCH_INTERVAL_MS = 500.0

# ===========================
# Monitoramento
# ===========================
N_W_SAMPLES = 64
W_MON_DT    = 50*ms

# ===========================
# Atraso sináptico
# ===========================
DELAY = 1.5*ms
