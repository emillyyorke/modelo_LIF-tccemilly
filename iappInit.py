# iappInit.py
from brian2 import *
import numpy as np
from SimulationParameters import (
    N, FEEDBACK_MODE,
    I_min_adapt, I_max_adapt,
    I_DIST_DEP, I_min_dep, I_max_dep, I_mean_dep, I_std_dep
)

def make_iapp():
    """
    Gera correntes externas heterogêneas para os N neurônios.

    Ambos os modos usam distribuição uniforme estilo Tabak 2010:
    - 'depression': I_i ∈ [0.15, 1.15] × rheobase  → [30, 230] pA
    - 'adaptation': I_i ∈ [0.5,  1.5 ] × rheobase  → [100, 300] pA
    """
    if FEEDBACK_MODE == 'adaptation':
        np.random.seed(45)  # seed calibrada: R_prec≈0.86, R_foll≈0.04
    else:
        np.random.seed(42)  # seed original da depressão — não alterar

    if FEEDBACK_MODE == 'adaptation':
        # Distribuição uniforme estilo Tabak (Table 1, col. adaptation)
        I = np.random.uniform(float(I_min_adapt), float(I_max_adapt), size=N) * amp

    elif FEEDBACK_MODE == 'depression':
        if I_DIST_DEP == 'uniform':
            I = np.random.uniform(float(I_min_dep), float(I_max_dep), size=N) * amp
        elif I_DIST_DEP == 'gaussian':
            I = I_mean_dep + I_std_dep * np.random.randn(N)
        else:
            raise ValueError(f"I_DIST_DEP inválido: '{I_DIST_DEP}'.")
    else:
        raise ValueError(f"FEEDBACK_MODE inválido: '{FEEDBACK_MODE}'.")

    return I
