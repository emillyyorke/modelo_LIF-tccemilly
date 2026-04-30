# iappInit.py
from brian2 import *
import numpy as np
from SimulationParameters import (
    N, FEEDBACK_MODE,
    I_min_adapt, I_max_adapt,
    I_min_dep, I_max_dep
)

# Defina aqui a ordem desejada das correntes: 'RAND', 'ASC' ou 'DES'
I_ORDER = 'RAND' 

def make_iapp():
    """
    Retorna as correntes externas fixas baseadas no modelo Hodgkin-Huxley de referência.
    Mapeia os valores brutos do HH para as faixas seguras do modelo LIF atual.
    """
    if N != 100:
        raise ValueError("Este conjunto de correntes do HH exige exatamente N=100 neurônios.")

    # ===== Valores originais do modelo HH (Ordem Aleatória) =====
    hh_raw = np.array([
        -2.18619, -2.57576, -3.41212, -3.71471, -5.39247,  2.96518,  3.71410,  4.12336,
         2.06839, -3.20933, -8.78277, -8.15378,  1.66234, -5.68270, -4.08322, -2.69158,
        -2.59407, -1.73528,  2.83197,  3.40739, -5.76464, -9.56374, -5.77334,  2.82739,
        -6.84317, -6.86834, -0.258034, -0.471816, -3.84289, -3.70281, -5.67721,  1.82440,
        -8.66329,  3.99792, -6.82577, -5.59847,  1.23707, -3.68862, -4.54604,  4.11283,
        -0.170141,  0.237281, -9.07758, -8.10526, -8.86425, -9.28953, -3.72341,  0.655232,
        -5.05325, -4.68108, -9.62691, -8.53282, -7.38884,  3.81252,  0.0715659, -7.08075,
        -1.63961, -3.99533, -4.09009, -4.03607, -5.86123,  4.15403, -0.667745, -5.89236,
        -1.58879, -1.57781,  0.690023, -8.71548, -8.09061, -8.49620, -0.269478,  2.26341,
         0.794397,  1.01733, -7.20252,  2.92398,  0.197913, -4.24802, -6.54378, -9.74914,
        -1.38554,  4.37422,  0.769219,  4.39207, -3.88913,  2.66762,  1.93197,  2.61223,
        -3.17728, -2.08136, -1.50227, -7.49413, -6.29200,  1.53645, -7.09815, -0.705741,
         1.02374, -6.51036, -0.490585,  4.55962
    ])

    # ===== Ordenação do Array =====
    if I_ORDER == 'ASC':
        hh_raw = np.sort(hh_raw)            # Crescente
    elif I_ORDER == 'DES':
        hh_raw = np.sort(hh_raw)[::-1]      # Decrescente
    elif I_ORDER == 'RAND':
        pass                                # Mantém original
    else:
        raise ValueError(f"I_ORDER inválido: '{I_ORDER}'.")

    # ===== Mapeamento para o domínio do LIF (Normalização) =====
    # Para evitar hiperpolarização do LIF por correntes negativas do HH,
    # normalizamos os dados do HH para o intervalo [0, 1]
    hh_min = np.min(hh_raw)
    hh_max = np.max(hh_raw)
    hh_norm = (hh_raw - hh_min) / (hh_max - hh_min)

    # Identifica os limites da sua implementação LIF atual
    if FEEDBACK_MODE == 'adaptation':
        I_min, I_max = float(I_min_adapt), float(I_max_adapt)
    elif FEEDBACK_MODE == 'depression':
        I_min, I_max = float(I_min_dep), float(I_max_dep)
    else:
        raise ValueError(f"FEEDBACK_MODE inválido: '{FEEDBACK_MODE}'.")

    # Escala a distribuição normalizada para caber entre I_min e I_max em pA
    I_final = I_min + hh_norm * (I_max - I_min)


    return I_final * amp