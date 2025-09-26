# plotRaster.py 
import matplotlib.pyplot as plt
import numpy as np
import importlib
import sys, os, glob
from matplotlib.ticker import FuncFormatter, MaxNLocator

# =========================
# CONFIG
# =========================
tT   = 10000.0
dt   = 0.01
root = "./results/"
base = ""                 # deixe "" para auto-inferir do *_Spikes.py
thA, thDA = 0.35, 0.14    # (não usados neste layout, mas mantidos)

# === preset visual desejado (1ª figura) ===
PRESET = "classic_fig1"   # mude para "paper" se quiser voltar ao anterior

def apply_style(preset):
    if preset == "classic_fig1":
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
    else:  # fallback “paper”
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

apply_style(PRESET)

# =========================
# localizar arquivos
# =========================
assert os.path.isdir(root), f"Diretório não existe: {root}"
if base == "":
    candidates = sorted(glob.glob(os.path.join(root, "*_Spikes*.py")), key=os.path.getmtime)
    assert len(candidates) > 0, "Nenhum *_Spikes*.py em ./results/"
    spike_module_path = candidates[-1]
    spike_module_name = os.path.basename(spike_module_path)[:-3]
    if spike_module_name.endswith("_Spikes_pN-1"):
        base = spike_module_name[:-len("_Spikes_pN-1")]
    elif spike_module_name.endswith("_Spikes"):
        base = spike_module_name[:-len("_Spikes")]
    else:
        base = spike_module_name.split("_Spikes")[0]
else:
    spike_module_name = base + "_Spikes"

sys.path.append(root)
moduleSpikeTimes = importlib.import_module(spike_module_name)
spikeTimes = moduleSpikeTimes.spikeTimes
nNeurons = len(spikeTimes)

np_path = os.path.join(root, base + "_np.txt")
assert os.path.exists(np_path), f"Arquivo não encontrado: {np_path}"
dc = np.loadtxt(np_path, delimiter="\t")
if dc.ndim == 1:
    dc = dc.reshape(-1, 4)
# colunas: t_ms, rate_hz, A_norm, S_norm
t = dc[:, 0].astype(float)     # ms
rate_hz = dc[:, 1].astype(float)
A = dc[:, 2].astype(float)
S = dc[:, 3].astype(float)
nT = len(A)
if nT <= 1:
    raise RuntimeError("Série muito curta em _np.txt")
tT = float(t[-1]) if t[-1] > 0 else tT
dt = np.mean(np.diff(t))

# =========================
# util: picos (para linhas verticais, estilo fig.1)
# =========================
def find_peaks_simple(y, prominence=0.05, min_dist_ms=200):
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    cand = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if len(cand) == 0:
        return np.array([], dtype=int)
    ymin, ymax = np.min(y), np.max(y)
    span = max(1e-9, ymax - ymin)
    good = cand[y[cand] >= (ymin + prominence * span)]
    if len(good) <= 1:
        return good
    min_dist = max(1, int(min_dist_ms / dt))
    keep = [int(good[0])]
    last = good[0]
    for idx in good[1:]:
        if idx - last >= min_dist:
            keep.append(int(idx))
            last = idx
    return np.asarray(keep, dtype=int)

peak_idx = find_peaks_simple(A, prominence=0.10, min_dist_ms=400)
peak_t_ms = t[peak_idx] if peak_idx.size else np.array([])

# =========================
# ISI (0–100 ms para hist)
# =========================
allisi = []
for n in range(nNeurons):
    st = np.asarray(spikeTimes[n], dtype=float)
    if st.size > 1:
        isi = np.diff(st)
        isi = isi[np.isfinite(isi)]
        allisi.append(isi)
allisi = np.concatenate(allisi) if len(allisi) else np.array([])
if allisi.size:
    allisi = allisi[(allisi > 0) & (allisi < 100.0)]

# =========================
# figure (layout da 1ª imagem)
# =========================
from matplotlib import gridspec
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

fig.suptitle(os.path.join(root, spike_module_name), y=0.98, fontsize=9)

# 1) raster (top-left)
ax0 = fig.add_subplot(gs[0, 0])
for n in range(nNeurons):
    st = np.asarray(spikeTimes[n], dtype=float)
    if st.size:
        ax0.scatter(st, np.full_like(st, n), s=3, marker='.', c='red', alpha=0.9, linewidths=0)
ax0.set_xlim([0, tT])
ax0.set_ylabel("Neuron")
ax0.invert_yaxis()
ax0.grid(alpha=0.2, linestyle=":")
ax0.tick_params('x', labelbottom=False)

# 2) ISI hist (top-right)
ax1 = fig.add_subplot(gs[0, 1])
if allisi.size:
    iqr = np.subtract(*np.percentile(allisi, [75, 25]))
    binw = 2 * iqr * (allisi.size ** (-1/3)) if iqr > 0 else None
    nb = int(np.clip((np.ptp(allisi) / binw) if binw else np.sqrt(allisi.size), 10, 40))
    ax1.hist(allisi, bins=nb, edgecolor='black', alpha=0.85)
    ax1.set_xlim(0, 100)
    ax1.set_title("ISI Dist. (0–100 ms)")
    ax1.set_xlabel("ISI (ms)")
ax1.grid(alpha=0.2, linestyle=":")

# 3) atividade + linhas verticais (bottom-left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t/1000.0, A, lw=1.2, color='black', label='Activity')
for tt in peak_t_ms:
    ax2.axvline(tt/1000.0, color='gray', lw=0.8, ls='--', alpha=0.6)
ax2.set_xlim([0, tT/1000.0])
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Activity")
ax2.grid(alpha=0.2, linestyle=":")

# 4) distribuição de A (bottom-right)
ax3 = fig.add_subplot(gs[1, 1])
iqrA = np.subtract(*np.percentile(A, [75, 25]))
binwA = 2 * iqrA * (len(A) ** (-1/3)) if iqrA > 0 else None
nba = int(np.clip((np.ptp(A) / binwA) if binwA else np.sqrt(len(A)), 15, 60))
hist, bins = np.histogram(A, bins=nba)
cent = 0.5*(bins[:-1] + bins[1:])
ax3.barh(cent, hist, height=(bins[1]-bins[0])*0.85, edgecolor='black', alpha=0.85)
ax3.set_ylim([A.min(), A.max()])
ax3.set_xlabel("Frequency")
ax3.set_title("Activity Distribution")
ax3.grid(alpha=0.2, linestyle=":")

def human_int(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{int(x)}"
ax3.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune='upper'))
ax3.xaxis.set_major_formatter(FuncFormatter(human_int))

# ===== salvar figura principal =====
out_png = os.path.join(root, f"{base}_classic_fig1.png")
try:
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[ok] Figura salva em: {out_png}")
except Exception as e:
    print(f"[warn] Não foi possível salvar a figura: {e}")

plt.show()
