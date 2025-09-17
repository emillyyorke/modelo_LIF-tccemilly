# iappInit.py
from brian2 import *
import numpy as np
from SimulationParameters import N, I_mean, I_std

def make_iapp():
    # Currents heterogêneas próximas do limiar
    np.random.seed(42)
    I = I_mean + I_std*np.random.randn(N)
    return I
