# LIF_EMILLY.py
# Executa somente em runtime (sem C++), usando gerador NumPy (sem Cython/MSVC)

import json
import numpy as np
from brian2 import (
    set_device, prefs, defaultclock, seed,
    SpikeMonitor, PopulationRateMonitor, StateMonitor, run
)
from SimulationParameters import (
    DT, SIM_TIME, RANDOM_SEED, N_NEURONS, USE_STDP, STDP_MODE
)
from iappInit import make_neurons
from SimulationInitialization import make_excitatory_synapses

# --- Força runtime puro (sem compilador) ---
set_device('runtime')        # nada de cpp_standalone
prefs.codegen.target = 'numpy'  # backend puro NumPy (não usa Cython)

defaultclock.dt = DT

def prefix():
    return f"with_stdp_{STDP_MODE}_" if USE_STDP else "no_stdp_"

def main():
    seed(RANDOM_SEED)

    # ----- Modelo -----
    neurons = make_neurons()
    syn = make_excitatory_synapses(neurons)

    # ----- Monitores -----
    spk_mon  = SpikeMonitor(neurons)
    rate_mon = PopulationRateMonitor(neurons)

    # Monitor de pesos (somente se STDP estiver ligado)
    w_mon = None
    sample_idx = None
    if USE_STDP:
        n_total_syn = N_NEURONS * (N_NEURONS - 1)  # all-to-all sem autapses
        n_sample = min(200, n_total_syn)
        rng = np.random.default_rng(RANDOM_SEED + 2)
        sample_idx = np.array(sorted(rng.choice(n_total_syn, size=n_sample, replace=False)))
        w_mon = StateMonitor(syn, 'w', record=sample_idx)

    # ----- Simulação -----
    run(SIM_TIME, report='text')

    # ----- Salvar dados -----
    pfx = prefix()
    # spikes
    np.save(pfx + 'spikes_i.npy', spk_mon.i)
    np.save(pfx + 'spikes_t.npy', spk_mon.t)
    # taxa populacional
    np.save(pfx + 'rate_t.npy', rate_mon.t)
    np.save(pfx + 'rate_rate.npy', rate_mon.rate)
    # pesos (se houver)
    if w_mon is not None:
        np.save(pfx + 'weight_t.npy', w_mon.t)
        np.save(pfx + 'weight_w.npy', w_mon.w)
        np.save(pfx + 'weight_idx.npy', sample_idx)

    # metadados p/ plot
    meta = {"USE_STDP": USE_STDP, "STDP_MODE": STDP_MODE if USE_STDP else None}
    with open(pfx + 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
