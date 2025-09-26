# LIF_EMILLY.py — principal
from brian2 import *
import numpy as np
import os

from SimulationParameters import *
from SimulationInitialization import (
    make_neurons, make_synapses, make_monitors
)

def main():
    # Garantir runtime puro (nada de C++)
    prefs.codegen.target = 'numpy'   # ou 'runtime'
    start_scope()

    # Pasta padrão de saída
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Construção da rede
    G = make_neurons()
    S = make_synapses(G)
    spk, rate, wmon = make_monitors(G, S)

    print(f"[INFO] STDP_ENABLED = {STDP_ENABLED} | sinapses = {S.N} | dt = {defaultclock.dt}")

    # Monte explicitamente a Network
    components = [G, S, spk, rate]
    if wmon is not None:
        components.append(wmon)

    net = Network(*components)
    net.run(SIM_TIME, report='text')

    # >>> Salvar resultados para o exporter (em ./results)
    np.save(os.path.join(results_dir, "spike_i.npy"), np.asarray(spk.i))
    np.save(os.path.join(results_dir, "spike_t.npy"), np.asarray(spk.t/ms, dtype=float))
    np.save(os.path.join(results_dir, "rate_t.npy"), np.asarray(rate.t/ms, dtype=float))
    np.save(
        os.path.join(results_dir, "rate_hz.npy"),
        np.asarray(rate.smooth_rate(window='gaussian', width=50*ms)/Hz, dtype=float)
    )

    if STDP_ENABLED and (wmon is not None):
        np.save(os.path.join(results_dir, "w_t.npy"), np.asarray(wmon.t/ms, dtype=float))
        # média da amostra por tempo
        if len(wmon.record) > 0:
            w_stack = np.vstack([wmon.w[k] for k in range(len(wmon.record))])
            np.save(os.path.join(results_dir, "w_mean.npy"), w_stack.mean(axis=0))
        else:
            # fallback se nada foi gravado
            np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))
    else:
        # placeholders para o plot quando STDP desligado
        np.save(os.path.join(results_dir, "w_t.npy"),   np.array([0.0]))
        np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))

    print(f"[OK] Arquivos salvos em: {os.path.abspath(results_dir)}")
    print("Pronto p/ rodar:  python export_for_plot.py --root ./results --base SEU_BASE --append-ts")

if __name__ == "__main__":
    main()
