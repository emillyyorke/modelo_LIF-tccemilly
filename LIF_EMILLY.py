# LIF_EMILLY.py — principal
from brian2 import *
import numpy as np
import os
import time

from SimulationParameters import *
from SimulationInitialization import (
    make_neurons, make_synapses, make_monitors
)


def main():
    start_time = time.time()

    prefs.codegen.target = 'numpy'
    start_scope()

    # ---- Nome da pasta de resultados ----
    base_results_dir = "./results"
    timestamp = int(time.time())

    mode_tag = FEEDBACK_MODE.upper()

    if STDP_ENABLED:
        ltp_str = str(A_LTP).replace('.', 'p')
        ltd_str = str(A_LTD).replace('.', 'p').replace('-', 'm')
        stdp_tag = STDP_MODE.upper()
        run_name = f"{mode_tag}_STDP_{stdp_tag}_LTP_{ltp_str}_LTD_{ltd_str}_{timestamp}"
    else:
        run_name = f"{mode_tag}_STDP_OFF_{timestamp}"

    results_dir = os.path.join(base_results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Resultados serão salvos em: {os.path.abspath(results_dir)}")

    # ---- Salvar parâmetros ----
    with open(os.path.join(results_dir, "params.txt"), "w") as f:
        f.write(f"FEEDBACK_MODE = {FEEDBACK_MODE}\n")
        f.write(f"STDP_ENABLED = {STDP_ENABLED}\n")
        f.write(f"STDP_MODE = {STDP_MODE}\n")
        f.write(f"A_LTP = {A_LTP}\n")
        f.write(f"A_LTD = {A_LTD}\n")
        f.write(f"eta = {eta}\n")
        f.write(f"W_MIN = {W_MIN}\n")
        f.write(f"W_MAX = {W_MAX}\n")
        f.write(f"N = {N}\n")
        f.write(f"gbar_syn = {gbar_syn}\n")
        f.write(f"tau_e = {tau_e}\n")
        f.write(f"T_ref = {T_ref}\n")
        if FEEDBACK_MODE == 'depression':
            f.write(f"tau_s_rec = {tau_s_rec}\n")
            f.write(f"delta_dep = {delta_dep}\n")
            f.write(f"I_DIST_DEP = {I_DIST_DEP}\n")
        elif FEEDBACK_MODE == 'adaptation':
            f.write(f"gbar_theta_min = {gbar_theta_min}\n")
            f.write(f"gbar_theta_max = {gbar_theta_max}\n")
            f.write(f"gbar_theta_mean = {gbar_theta}\n")
            f.write(f"tau_a = {tau_a}\n")
            f.write(f"I_min_adapt = {I_min_adapt}\n")
            f.write(f"I_max_adapt = {I_max_adapt}\n")

    # ---- Criar componentes ----
    G = make_neurons()
    S = make_synapses(G)
    spk, rate, wmon, slow_mon = make_monitors(G, S)

    print(f"[INFO] FEEDBACK_MODE = {FEEDBACK_MODE} | STDP = {STDP_ENABLED} "
          f"({STDP_MODE if STDP_ENABLED else 'N/A'}) | "
          f"sinapses = {S.N} | dt = {defaultclock.dt}")

    # ---- Montar a rede ----
    components = [G, S, spk, rate]
    if wmon is not None:
        components.append(wmon)
    if slow_mon is not None:
        components.append(slow_mon)

    net = Network(*components)

    # ---- Importar batch STDP se necessário ----
    use_batch_stdp = STDP_ENABLED and (STDP_MODE == 'batch')
    if use_batch_stdp:
        from batch_stdp import apply_batch_stdp
        print(f"[INFO] Batch STDP ativo. Intervalo = {STDP_BATCH_INTERVAL_MS} ms")

    # ---- Loop de simulação ----
    SNAPSHOT_INTERVAL = 500 * ms
    simulation_steps = np.arange(0, SIM_TIME / second, SNAPSHOT_INTERVAL / second) * second

    if STDP_ENABLED:
        initial_weights = np.asarray(S.w)
        np.save(os.path.join(results_dir, "weights_t_00000.npy"), initial_weights)
        print(f"[INFO] Snapshot inicial dos pesos salvo.")

    # Variável para rastrear spike index no monitor (para batch STDP)
    prev_spike_count = 0

    # Log de evolução dos pesos (para diagnóstico)
    weight_log = []

    for t_start in simulation_steps:
        t_end = t_start + SNAPSHOT_INTERVAL
        print(f"[RUNNING] Simulando de {t_start} até {t_end}...")
        net.run(SNAPSHOT_INTERVAL, report='text')

        # ---- Batch STDP: aplicar atualizações de peso ----
        if use_batch_stdp:
            current_spike_count = spk.num_spikes
            if current_spike_count > prev_spike_count:
                # Obter todos os spikes e filtrar pelo intervalo atual
                all_spike_i = np.asarray(spk.i)
                all_spike_t = np.asarray(spk.t / ms, dtype=float)

                t_start_ms = float(t_start / ms)
                t_end_ms   = float(t_end / ms)

                stats = apply_batch_stdp(
                    S, all_spike_i, all_spike_t,
                    t_start_ms, t_end_ms
                )

                weight_log.append({
                    't_ms': t_end_ms,
                    **stats
                })

                print(f"  [STDP] <w>={stats['mean_w']:.4f}  "
                      f"[{stats['min_w']:.3f}, {stats['max_w']:.3f}]  "
                      f"<Δw>={stats['mean_dw']:.6f}  "
                      f"LTP_pairs={stats['n_ltp_pairs']}  "
                      f"LTD_pairs={stats['n_ltd_pairs']}")
            else:
                print(f"  [STDP] Sem spikes neste intervalo — pesos inalterados.")

            prev_spike_count = current_spike_count

        # ---- Salvar snapshot dos pesos ----
        if STDP_ENABLED:
            current_weights = np.asarray(S.w)
            time_ms = int(t_end / ms)
            np.save(os.path.join(results_dir, f"weights_t_{time_ms:05d}.npy"), current_weights)

    # ---- Salvar dados de spikes e taxa ----
    np.save(os.path.join(results_dir, "spike_i.npy"), np.asarray(spk.i))
    np.save(os.path.join(results_dir, "spike_t.npy"), np.asarray(spk.t / ms, dtype=float))
    np.save(os.path.join(results_dir, "rate_t.npy"), np.asarray(rate.t / ms, dtype=float))
    np.save(os.path.join(results_dir, "rate_hz.npy"),
            np.asarray(rate.smooth_rate(window='flat', width=10 * ms) / Hz, dtype=float))

    # ---- Salvar dados da variável lenta (depressão: s, adaptação: ga) ----
    if slow_mon is not None:
        if FEEDBACK_MODE == 'depression':
            s_all = np.array(slow_mon.s)           # shape: (N_neurons, n_timesteps)
            s_mean = s_all.mean(axis=0)
            s_t_ms = np.asarray(slow_mon.t / ms, dtype=float)
            np.save(os.path.join(results_dir, "s_mean.npy"), s_mean)
            np.save(os.path.join(results_dir, "s_t.npy"), s_t_ms)
            print(f"[INFO] Dados de <s> salvos ({len(s_mean)} amostras).")

        elif FEEDBACK_MODE == 'adaptation':
            ga_all = np.array(slow_mon.ga)          # shape: (N_neurons, n_timesteps)
            # Normalizar ga para adimensional: theta_i = ga_i / gbar_theta
            # Assim <theta> fica entre 0 e ~1, comparável ao Tabak Fig. 5
            ga_mean = ga_all.mean(axis=0)            # média em nS
            theta_mean = ga_mean / float(gbar_theta) # adimensional
            theta_t_ms = np.asarray(slow_mon.t / ms, dtype=float)
            np.save(os.path.join(results_dir, "theta_mean.npy"), theta_mean)
            np.save(os.path.join(results_dir, "theta_t.npy"), theta_t_ms)
            print(f"[INFO] Dados de <θ> salvos ({len(theta_mean)} amostras).")

    # ---- Salvar dados de pesos (STDP) ----
    if STDP_ENABLED and (wmon is not None):
        np.save(os.path.join(results_dir, "w_t.npy"), np.asarray(wmon.t / ms, dtype=float))
        if len(wmon.record) > 0:
            w_stack = np.vstack([wmon.w[k] for k in range(len(wmon.record))])
            np.save(os.path.join(results_dir, "w_mean.npy"), w_stack.mean(axis=0))
        else:
            np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))
    else:
        np.save(os.path.join(results_dir, "w_t.npy"), np.array([0.0]))
        np.save(os.path.join(results_dir, "w_mean.npy"), np.array([float(W_INIT_FIXED)]))

    # ---- Salvar log de evolução dos pesos (batch STDP) ----
    if use_batch_stdp and weight_log:
        import json
        with open(os.path.join(results_dir, "stdp_weight_log.json"), "w") as f:
            json.dump(weight_log, f, indent=2)
        print(f"[INFO] Log STDP salvo ({len(weight_log)} entradas).")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[INFO] Tempo total de execução: {duration:.2f} segundos.")
    print(f"[OK] Arquivos salvos em: {os.path.abspath(results_dir)}")

    # ---- Gerar relatório de métricas automaticamente ----
    try:
        from relatorio_metricas import gerar_relatorio
        print("\n[INFO] Gerando relatorio de metricas...")
        gerar_relatorio(results_dir, verbose=False)
    except Exception as e:
        print(f"[AVISO] Nao foi possivel gerar relatorio de metricas: {e}")

    print("\nPROXIMOS PASSOS:")
    print(f"1. Painel resumo (raster+correlações): python plot_summary.py --dir \"{results_dir}\"")
    print(f"2. Exporte dados para o raster plot com: python export_for_plot.py --root \"{results_dir}\" --base N_TESTE")
    print(f"3. Gere o raster plot com: python plotRaster.py --dir \"{results_dir}\"")
    print(f"4. Gere os plots de distribuição de pesos com: python plot_weight_evolution.py --dir \"{results_dir}\"")

    if FEEDBACK_MODE in ('depression', 'adaptation'):
        print(f"4. Análise Tabak (auto-detecta modo): python plot_tabak_analysis.py --dir \"{results_dir}\"")


if __name__ == "__main__":
    main()
