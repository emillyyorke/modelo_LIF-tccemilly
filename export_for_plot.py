# export_for_plot.py — gera BASE_Spikes.py e BASE_np.txt
import os, argparse, textwrap
import numpy as np
from datetime import datetime

INVALID = '<>:"/\\|?*,'

# ------------------------------ util básicos ------------------------------
def sanitize_base(s: str) -> str:
    s = (s or "").strip()
    for ch in INVALID: s = s.replace(ch, "-")
    s = s.replace(" ", "_")
    while "__" in s: s = s.replace("__", "_")
    while "--" in s: s = s.replace("--", "-")
    return s.strip("._-")

def load_npy_must(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return np.load(path)

def _norm01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

# ------------------------------ spikes / série ------------------------------
def build_spikeTimes(spike_i, spike_t, n_neurons=None):
    spike_i = np.asarray(spike_i).astype(int)
    spike_t = np.asarray(spike_t).astype(float)
    if spike_i.shape != spike_t.shape:
        raise ValueError(f"Dimensões divergentes: spike_i {spike_i.shape} vs spike_t {spike_t.shape}")
    if n_neurons is None:
        n_neurons = int(spike_i.max()) + 1 if spike_i.size else 0
    buckets = [list() for _ in range(n_neurons)]
    order = np.argsort(spike_t, kind="mergesort")
    for idx in order:
        i = int(spike_i[idx])
        if 0 <= i < n_neurons:
            buckets[i].append(float(spike_t[idx]))
    return [np.asarray(b, dtype=float) for b in buckets]

def save_spikes_py(spikeTimes, out_py_path):
    header = textwrap.dedent("""\
    # Gerado automaticamente para o plotRaster (formato LIF)
    import numpy as np
    # spikeTimes: lista com um np.array de tempos (ms) por neurônio
    spikeTimes = [
    """)
    with open(out_py_path, "w", encoding="utf-8") as f:
        f.write(header)
        for arr in spikeTimes:
            arr_str = np.array2string(arr, separator=", ", threshold=100000, floatmode="maxprec", precision=6)
            f.write(f"    np.asarray({arr_str}, dtype=float),\n")
        f.write("]\n")

def save_np_txt(t_ms, rate_hz, w_t=None, w_mean=None, out_txt_path=None):
    t_ms = np.asarray(t_ms, dtype=float)
    rate_hz = np.asarray(rate_hz, dtype=float)
    if t_ms.ndim != 1 or rate_hz.ndim != 1 or t_ms.size != rate_hz.size:
        raise ValueError("rate_t.npy e rate_hz.npy devem ser vetores 1D com o mesmo tamanho.")
    A_norm = _norm01(rate_hz)
    if w_t is not None and w_mean is not None and w_t.ndim == 1 and w_mean.ndim == 1 and w_t.size == w_mean.size and w_t.size >= 2:
        order = np.argsort(w_t, kind="mergesort")
        w_interp = np.interp(t_ms, w_t[order], w_mean[order], left=w_mean[order][0], right=w_mean[order][-1])
        S_norm = _norm01(w_interp)
    else:
        S_norm = np.zeros_like(t_ms, dtype=float)
    M = np.column_stack([t_ms, rate_hz, A_norm, S_norm])
    np.savetxt(out_txt_path, M, fmt="%.6f", delimiter="\t")

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Exporta arquivos de dados para o script de plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", default="./results/", help="Pasta dos .npy e saídas")
    ap.add_argument("--base", default="LIF", help="Prefixo do experimento")
    ap.add_argument("--n", type=int, default=None, help="Força n_neurons (opcional)")
    ap.add_argument("--append-ts", dest="append_ts", action="store_true", help="Anexa timestamp ao base")
    ap.add_argument("--ts-format", default="%Y-%m-%d_%H%M%S", help="Formato do timestamp")
    args = ap.parse_args()

    root = args.root
    os.makedirs(root, exist_ok=True)

    base = sanitize_base(args.base)
    if args.append_ts:
        base = f"{base}_{datetime.now().strftime(args.ts_format)}"

    # Entradas obrigatórias
    p_spike_i = os.path.join(root, "spike_i.npy")
    p_spike_t = os.path.join(root, "spike_t.npy")
    p_rate_t  = os.path.join(root, "rate_t.npy")
    p_rate_hz = os.path.join(root, "rate_hz.npy")

    # Opcionais (só p/ S_norm no np.txt)
    p_w_t     = os.path.join(root, "w_t.npy")
    p_w_mean  = os.path.join(root, "w_mean.npy")

    # ----- gerar spikes.py e np.txt -----
    spike_i = load_npy_must(p_spike_i)
    spike_t = load_npy_must(p_spike_t)
    rate_t  = load_npy_must(p_rate_t)
    rate_hz = load_npy_must(p_rate_hz)
    w_t = np.load(p_w_t) if os.path.exists(p_w_t) else None
    w_mean = np.load(p_w_mean) if os.path.exists(p_w_mean) else None

    spikeTimes = build_spikeTimes(spike_i, spike_t, n_neurons=args.n)
    out_spikes_py = os.path.join(root, f"{base}_Spikes.py")
    save_spikes_py(spikeTimes, out_spikes_py)
    print(f"[ok] Gerado: {out_spikes_py}  (neurônios: {len(spikeTimes)})")

    out_np_txt = os.path.join(root, f"{base}_np.txt")
    save_np_txt(rate_t, rate_hz, w_t=w_t, w_mean=w_mean, out_txt_path=out_np_txt)
    print(f"[ok] Gerado: {out_np_txt}  (amostras: {len(rate_t)})")

if __name__ == "__main__":
    main()
