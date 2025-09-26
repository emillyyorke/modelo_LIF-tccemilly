# SimulationParameters.py
from brian2 import *

# ===========================
# Tempo de simulação
# ===========================
SIM_TIME = 10*second
DT       = 0.1*ms

# ===========================
# Tamanho da rede
# ===========================
N = 250                   # todos excitatórios
P_CONNECT = 1.0           # completamente conectada
AUTAPSES = False          # sem autoconexões

# ===========================
# LIF + condutâncias (Tabak-style names)
# ===========================
Cm      = 200*pF          # Capacitância
gL      = 10*nS           # Condutância de vazamento  -> tau_m = Cm/gL = 20 ms
EL      = -70*mV          # Potencial de repouso
V_th    = -50*mV          # limiar
V_reset = -60*mV          # reset
T_ref   = 2*ms            # período refratário

# Sinapse excitatória
gbar_syn = 0.08*nS        # ganho base multiplicado por w (→ ajuste p/ Tabak 2010)
V_syn    = 0*mV           # reversal da sinapse excitatória
tau_e    = 5*ms           # decaimento da condutância excitatória (beta_a^-1 aproximadamente)

# Adaptação (canal K/hiperpolarizante)
gbar_theta = 0.5*nS       # ganho da adaptação (incremento por spike)
V_theta    = -80*mV       # reversal da adaptação
tau_a      = 200*ms       # constante de decaimento (aprox. 1/beta_theta)

# ===========================
# Corrente externa (iapp)
# ===========================
I_mean = 200*pA           # média próxima do limiar
I_std  = 50*pA            # desvio para heterogeneidade

# ===========================
# STDP (Bi & Poo, aditivo)
# ===========================
STDP_ENABLED = False      # <--- comute aqui

# Janelas temporais
tau_pre  = 20*ms
tau_post = 20*ms

# Magnitudes (aqui mantenho LTD ligeiramente mais forte p/ evitar saturação)
A_LTP = 0.04          # Δw por pós-antes
A_LTD = -0.02        # Δw por antes-pós
eta   = 1.0               # fator global

# Limites e inicialização
W_MIN = 0.0
W_MAX = 2.0
W_INIT_FIXED = 1.0        # sem STDP
W_INIT_MIN   = 0.5        # com STDP: uniforme [0.5, 1.5]
W_INIT_MAX   = 1.5

# ===========================
# Monitoramento
# ===========================
N_W_SAMPLES = 64          # amostra de sinapses p/ monitorar pesos
W_MON_DT    = 50*ms       # passo de gravação do peso

# ===========================
# Atraso sináptico
# ===========================
DELAY = 1.5*ms
