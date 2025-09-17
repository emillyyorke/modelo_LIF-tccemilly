# plotRaster.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    si = np.load('spike_i.npy')
    st = np.load('spike_t.npy')
    rt = np.load('rate_t.npy')
    rhz = np.load('rate_hz.npy')

    wt = np.load('w_t.npy')
    wmean = np.load('w_mean.npy')

    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Resultados', fontsize=14)

    ax1 = plt.subplot2grid((2,2), (0,0))
    ax1.scatter(st/1000.0, si, s=2, c='r')
    ax1.set_title('Raster (rede)')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Neurônio')

    ax2 = plt.subplot2grid((2,2), (1,0))
    ax2.plot(rt/1000.0, rhz, lw=1.5)
    ax2.set_title('Atividade média da rede (Population Rate)')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Taxa (Hz)')

    ax3 = plt.subplot2grid((2,2), (0,1), rowspan=2)
    if wt.size > 1:
        ax3.plot(wt/1000.0, wmean, lw=2.0, label='média(w)')
        ax3.set_title('Evolução de pesos (amostra + média)')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Peso (adimensional)')
        ax3.legend()
    else:
        ax3.set_title('Pesos fixos (STDP desligado)')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Peso (adimensional)')
        ax3.axhline(wmean[0], ls='--', lw=2)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
