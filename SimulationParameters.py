# SimulationParameters.py
from brian2 import *

# ===========================
# Tempo de simulação
# ===========================
SIM_TIME = 40*second
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
FEEDBACK_MODE = 'adaptation'  # <--- comute aqui

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
# Tabak normalizado:
#   I_i ∈ [0.5, 1.5]    → físico: [100, 300] pA  (rheobase = 200 pA)
#   g_syn = 1.4         → físico: 14 nS total → 0.14 nS por sinapse
#   α_θ × T_θ = 0.01    → incremento de θ por spike (~1% do máximo)
#   β_θ = 0.004         → tau_a = tau_m / 0.004 = 5000 ms
#   g_θ ∈ [0.5, 1.5]    → MÁXIMO de saturação, NÃO o incremento por spike
#   T_ref = 0.25 tau_m  = 5 ms
#
# Em unidades físicas, com tau_a = 5000 ms e taxa de disparo de ~50 Hz dentro
# de um episódio, o equilíbrio ga = inc × rate × tau_a ≈ 0.1 × 50 × 5 = 25 nS,
# o que dá uma corrente de adaptação suficiente para terminar episódios.
T_ref_adapt    = 5*ms             # refratário (Tabak: 0.25 tau_m)
gbar_syn_adapt = 0.14*nS          # 14 nS / (N-1) ≈ 0.14 nS para N=100
tau_e_adapt    = 10*ms            # decaimento sináptico
V_theta        = -80*mV           # reversal da adaptação (K-like)
tau_a          = 2500*ms          # decaimento da adaptação
                                  # (Tabak original = 5000ms, mas sem saturação em θ_max
                                  # o ga acumula mais → valor reduzido para IEI compatível
                                  # com ~107 episódios em 480s)

# Incremento de ga por spike — HETEROGÊNEO por neurônio
# Faixa ampla é importante para criar variabilidade estocástica nos episódios
# (sem heterogeneidade suficiente, a rede entra em regime determinístico
# com episódios idênticos e R_preceding cai drasticamente)
gbar_theta_min = 0.05*nS          # incremento mínimo por spike
gbar_theta_max = 0.25*nS          # incremento máximo por spike (5× o min)
gbar_theta     = 0.5 * (gbar_theta_min + gbar_theta_max)  # média (para normalização/log)

# Corrente externa — adaptação (uniforme estilo Tabak)
# Faixa ampla = mais neurônios spontaneamente ativos = mais ruído de coincidência
I_min_adapt = 80*pA               # 0.4 × rheobase
I_max_adapt = 340*pA              # 1.7 × rheobase

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
STDP_ENABLED = False       # <--- comute aqui

# Modo de STDP:
#   'batch'        → STDP em lote (batch_stdp.py) — RECOMENDADO para redes episódicas
#   'event_driven' → STDP event-driven do Brian2 — NÃO RECOMENDADO (viés LTD)
STDP_MODE = 'batch'

# Janelas temporais
tau_pre  = 20*ms
tau_post = 20*ms

# Magnitudes
# REGRA: Com N=100 e ~50 spikes/neurônio/episódio, cada sinapse recebe
# ~50 pares por episódio. Mantenha |A_LTP|, |A_LTD| ≤ 0.001.
#
# Simétrico  (A_LTP = -A_LTD): peso médio estável em ~1.0
# Assimétrico: LTP > |LTD| → fortalecimento gradual
A_LTP =  0.0006          # Δw por par causal (pré-antes-pós)
A_LTD = -0.0004         # Δw por par anti-causal (pós-antes-pré)
eta   =  1.0             # fator global

# Limites e inicialização
W_MIN = 0.5               # piso — 50% do calibrado (impede morte da rede)
W_MAX = 2.0
W_INIT_FIXED = 1.0        # sem STDP
W_INIT_MIN   = 0.95       # com STDP: próximo ao calibrado
W_INIT_MAX   = 1.05

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
