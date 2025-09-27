# LIF_EMILLY.py — principal
from brian2 import *
import numpy as np
import os
import time  # O módulo 'time' já está sendo importado

from SimulationParameters import *
from SimulationInitialization import (
    make_neurons, make_synapses, make_monitors
)

def main():
    # NOVO: Captura o tempo de início da simulação
    start_time = time.time()

    # Garantir runtime puro (nada de C++)
    prefs.codegen.target = 'numpy'
    start_scope()

    # =================================================================
    # Geração de nome de pasta para a simulação
    # =================================================================
    base_results_dir = "./results"
    timestamp = int(time.time())
    
    if STDP_ENABLED:
        ltp_str = str(A_LTP).replace('.', 'p')
        ltd_str = str(A_LTD).replace('.', 'p').replace('-', 'm')
        run_name = f"STDP_ON_LTP_{ltp_str}_LTD_{ltd_str}_{timestamp}"
    else:
        run_name = f"STDP_OFF_{timestamp}"
    
    results_dir = os.path.join(base_results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Resultados serão salvos em: {os.path.abspath(results_dir)}")
    # =================================================================
    
    with open(os.path.join(results_dir, "params.txt"), "w") as f:
        f.write(f"STDP_ENABLED = {STDP_ENABLED}\n")
        f.write(f"A_LTP = {A_LTP}\n")
        f.write(f"A_LTD = {A_LTD}\n")

    G = make_neurons()
    S = make_synapses(G)
    spk, rate, wmon = make_monitors(G, S)

    print(f"[INFO] STDP_ENABLED = {STDP_ENABLED} | sinapses = {S.N} | dt = {defaultclock.dt}")

    components = [G, S, spk, rate]
    if wmon is not None:
        components.append(wmon)

    net = Network(*components)

    SNAPSHOT_INTERVAL = 500*ms
    simulation_steps = np.arange(0, SIM_TIME/second, SNAPSHOT_INTERVAL/second) * second

    if STDP_ENABLED:
        initial_weights = np.asarray(S.w)
        initial_save_path = os.path.join(results_dir, "weights_t_00000.npy")
        np.save(initial_save_path, initial_weights)
        print(f"[INFO] Snapshot inicial dos pesos salvo.")

    for t_start in simulation_steps:
        t_end = t_start + SNAPSHOT_INTERVAL
        print(f"[RUNNING] Simulando de {t_start} até {t_end}...")
        net.run(SNAPSHOT_INTERVAL, report='text')

        if STDP_ENABLED:
            current_weights = np.asarray(S.w)
            time_ms = int(t_end/ms)
            snapshot_save_path = os.path.join(results_dir, f"weights_t_{time_ms:05d}.npy")
            np.save(snapshot_save_path, current_weights)
            print(f"[INFO] Snapshot dos pesos em {time_ms}ms salvo.")
    
    np.save(os.path.join(results_dir, "spike_i.npy"), np.asarray(spk.i))
    np.save(os.path.join(results_dir, "spike_t.npy"), np.asarray(spk.t/ms, dtype=float))
    np.save(os.path.join(results_dir, "rate_t.npy"), np.asarray(rate.t/ms, dtype=float))
    np.save(os.path.join(results_dir, "rate_hz.npy"), np.asarray(rate.smooth_rate(window='gaussian', width=50*ms)/Hz, dtype=float))

    if STDP_ENABLED and (wmon is not None):
        np.save(os.path.join(results_dir, "w_t.npy"), np.asarray(wmon.t/ms, dtype=float))
        if len(wmon.record) > 0:
            w_stack = np.vstack([wmon.w[k] for k in range(len(wmon.record))])
            np.save(os.path.join(results_dir, "w_mean.npy"), w_stack.mean(axis=0))
        else:
            np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))
    else:
        np.save(os.path.join(results_dir, "w_t.npy"),   np.array([0.0]))
        np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[INFO] Tempo total de execução: {duration:.2f} segundos.")    

    print(f"[OK] Arquivos salvos em: {os.path.abspath(results_dir)}")
    print("\nPROXIMOS PASSOS:")
    print(f"1. Gere os plots de distribuição de pesos com: python plot_weight_evolution.py --dir \"{results_dir}\"")
    print(f"2. Exporte dados para o raster plot com: python export_for_plot.py --root \"{results_dir}\" --base N_TESTE")
    print(f"3. Gere o raster plot com: python plotRaster.py --dir \"{results_dir}\"")

if __name__ == "__main__":
    main()
