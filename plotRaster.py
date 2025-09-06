# plotRaster.py
# Visualização comparativa: detecta automaticamente execuções com/sem STDP e plota overlay

import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def load_run(prefix):
    data = {}
    def maybe(path):
        files = glob.glob(prefix + path)
        return files[0] if files else None

    # essenciais
    data['spike_i'] = np.load(prefix + 'spikes_i.npy', allow_pickle=True)
    data['spike_t'] = np.load(prefix + 'spikes_t.npy', allow_pickle=True)
    data['rate_t']  = np.load(prefix + 'rate_t.npy',  allow_pickle=True)
    data['rate']    = np.load(prefix + 'rate_rate.npy', allow_pickle=True)

    # opcionais (pesos)
    wt = maybe('weight_t.npy')
    ww = maybe('weight_w.npy')
    wi = maybe('weight_idx.npy')
    data['has_weights'] = (wt is not None and ww is not None and wi is not None)
    if data['has_weights']:
        data['weight_t'] = np.load(wt, allow_pickle=True)
        data['weight_w'] = np.load(ww, allow_pickle=True)
        data['weight_idx'] = np.load(wi, allow_pickle=True)

    # meta
    meta_file = maybe('meta.json')
    if meta_file:
        with open(meta_file, 'r', encoding='utf-8') as f:
            data['meta'] = json.load(f)
    else:
        data['meta'] = {}
    return data

def find_available_runs():
    runs = []
    candidates = []
    candidates += glob.glob('with_stdp_*_spikes_i.npy')
    candidates += glob.glob('no_stdp_*_spikes_i.npy')
    prefixes = sorted(set([c.replace('spikes_i.npy','') for c in candidates]))
    for p in prefixes:
        runs.append((p, load_run(p)))
    return runs

runs = find_available_runs()
if not runs:
    raise SystemExit("Nenhum resultado encontrado. Rode o script principal primeiro.")

def label_for(run):
    meta = run.get('meta', {})
    if meta.get('USE_STDP', False):
        mode = meta.get('STDP_MODE', 'stdp')
        return f"Com STDP ({mode})"
    return "Sem STDP"

# Raster + Taxa
fig1, axs1 = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

for _, data in runs:
    axs1[0].scatter(data['spike_t'], data['spike_i'], s=2, label=label_for(data), alpha=0.7)
axs1[0].set_title('Raster Plot (comparativo)')
axs1[0].set_ylabel('Índice do neurônio')
axs1[0].set_xlabel('Tempo (s)')
axs1[0].legend(loc='upper right')

for _, data in runs:
    axs1[1].plot(data['rate_t'], data['rate'], label=label_for(data))
axs1[1].set_title('Atividade média da rede (Population Rate)')
axs1[1].set_ylabel('Taxa (Hz)')
axs1[1].set_xlabel('Tempo (s)')
axs1[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

# Pesos (se houver)
any_weights = any(d['has_weights'] for _, d in runs)
if any_weights:
    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 4))
    for _, data in runs:
        if not data['has_weights']:
            continue
        mean_w = data['weight_w'].mean(axis=0)
        ax2.plot(data['weight_t'], mean_w, linewidth=2, label=label_for(data))
        # mostra até 3 pesos individuais para ilustrar
        k_show = min(3, data['weight_w'].shape[0])
        for k in range(k_show):
            ax2.plot(data['weight_t'], data['weight_w'][k], alpha=0.5, linewidth=0.8)
    ax2.set_title('Evolução dos pesos sinápticos (média + exemplos)')
    ax2.set_ylabel('Peso (adim.)')
    ax2.set_xlabel('Tempo (s)')
    ax2.legend(loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum conjunto com STDP (pesos) encontrado — gráficos de pesos não serão exibidos.")
