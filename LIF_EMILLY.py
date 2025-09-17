# LIF_EMILLY.py
from brian2 import *
from SimulationParameters import *
from SimulationInitialization import make_neurons, make_synapses, make_monitors

def main():
    # Garantir runtime puro (nada de C++)
    prefs.codegen.target = 'numpy'   # ou 'runtime'
    start_scope()

    G = make_neurons()
    S = make_synapses(G)
    spk, rate, wmon = make_monitors(G, S)

    print(f"[INFO] STDP_ENABLED = {STDP_ENABLED} | sinapses = {S.N} | dt = {defaultclock.dt}")
    run(SIM_TIME, report='text')

    # Salvar resultados leves p/ plot
    np.save('spike_i.npy', spk.i)
    np.save('spike_t.npy', spk.t/ms)
    np.save('rate_t.npy', rate.t/ms)
    np.save('rate_hz.npy', rate.smooth_rate(window='gaussian', width=50*ms)/Hz)

    if STDP_ENABLED:
        np.save('w_t.npy', wmon.t/ms)
        # média da amostra por tempo
        w_stack = np.vstack([wmon.w[k] for k in range(len(wmon.record))])
        np.save('w_mean.npy', w_stack.mean(axis=0))
    else:
        # placeholders para o plot
        np.save('w_t.npy', np.array([0.0]))
        np.save('w_mean.npy', np.array([W_INIT_FIXED]))

if __name__ == "__main__":
    main()
