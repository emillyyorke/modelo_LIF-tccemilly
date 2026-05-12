# plot_weight_snapshots.py
"""
Gera 4 snapshots da distribuição de pesos nos instantes 0%, 25%, 50% e 100%
da simulação, além de um painel legível com os 4 lado a lado e um relatório TXT.

Pode ser chamado manualmente ou importado pelo LIF_EMILLY.py.

Uso:
    python plot_weight_snapshots.py --dir ./results/DEPRESSION_STDP_BATCH_...
"""
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from SimulationParameters import W_MIN, W_MAX
except ImportError:
    W_MIN, W_MAX = 0.5, 2.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _read_params(results_dir):
    params = {}
    p = os.path.join(results_dir, "params.txt")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    params[k.strip()] = v.strip()
    return params


def _build_run_label(params):
    """Monta string de identificação para títulos de figuras."""
    mode = params.get("FEEDBACK_MODE", "?").upper()
    n    = params.get("N", "?")
    stdp = params.get("STDP_ENABLED", "False").lower() == "true"

    if stdp:
        altp      = params.get("A_LTP", "?")
        altd      = params.get("A_LTD", "?")
        stdp_mode = params.get("STDP_MODE", "batch").upper()
        return (f"Modo: {mode}  |  N = {n}  |  "
                f"STDP: ON ({stdp_mode})  |  A_LTP = {altp}  |  A_LTD = {altd}")
    else:
        return f"Modo: {mode}  |  N = {n}  |  STDP: OFF"


def _parse_time_from_file(fpath):
    """Extrai o tempo em ms do nome do arquivo weights_t_XXXXX.npy."""
    try:
        return int(os.path.basename(fpath).split('_t_')[1].split('.')[0])
    except Exception:
        return None


def _select_four_snapshots(snapshot_files):
    """
    Dado uma lista ordenada de arquivos weights_t_XXXXX.npy,
    retorna os 4 arquivos mais próximos de 0%, 25%, 50%, 100%.

    Retorna: list of (pct_float, time_ms, filepath)
    """
    timed = []
    for f in snapshot_files:
        t = _parse_time_from_file(f)
        if t is not None:
            timed.append((t, f))
    if not timed:
        return []
    timed.sort()

    t_vals  = [t for t, _ in timed]
    t_min   = t_vals[0]
    t_max   = t_vals[-1]
    t_range = t_max - t_min if t_max != t_min else 1

    selected = []
    for pct in [0.0, 0.25, 0.50, 1.0]:
        target = t_min + pct * t_range
        closest = min(timed, key=lambda x: abs(x[0] - target))
        selected.append((pct, closest[0], closest[1]))
    return selected


# ─────────────────────────────────────────────
# Figuras individuais
# ─────────────────────────────────────────────

def _plot_single(weights, time_ms, pct, run_label, output_path):
    bins   = np.linspace(W_MIN, W_MAX, 51)
    counts, edges = np.histogram(weights, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width   = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(centers, counts, width=width * 0.9,
           color='steelblue', edgecolor='black', alpha=0.85)

    mean_w = np.mean(weights)
    std_w  = np.std(weights)
    ax.axvline(mean_w, color='red', linestyle='--', linewidth=1.5,
               label=f"Média = {mean_w:.4f}")

    ax.set_xlabel("Peso Sináptico (w)", fontsize=12)
    ax.set_ylabel("Contagem de Sinapses", fontsize=12)
    ax.set_xlim(W_MIN, W_MAX)
    ax.set_title(
        f"Distribuição de Pesos — {int(pct * 100)}% da Simulação  (t = {time_ms} ms)\n"
        f"{run_label}",
        fontsize=9, pad=10
    )
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    stats_text = (
        f"N sináptico : {len(weights):,}\n"
        f"Média (μ)   : {mean_w:.5f}\n"
        f"DP    (σ)   : {std_w:.5f}\n"
        f"Mínimo      : {np.min(weights):.5f}\n"
        f"Máximo      : {np.max(weights):.5f}\n"
        f"Mediana     : {np.median(weights):.5f}"
    )
    ax.text(0.98, 0.96, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.85),
            family='monospace')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────
# Painel 2 × 2
# ─────────────────────────────────────────────

_PANEL_COLOR  = 'steelblue'
_PANEL_LABELS = ['0% — Início', '25%', '50%', '100% — Final']


def _plot_panel(snapshots_data, run_label, output_path):
    """
    snapshots_data : list of (pct_float, time_ms, weights_array)
    Layout: grade 2×2 (linha superior = 0% e 25%, inferior = 50% e 100%)
    """
    bins = np.linspace(W_MIN, W_MAX, 51)

    # Espaçamento reduzido entre os painéis:
    # hspace controla o espaço vertical
    # wspace controla o espaço horizontal
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        dpi=150,
        gridspec_kw={
            'hspace': 0.22,
            'wspace': 0.18
        }
    )

    axes_flat = axes.flatten()   # ordem: [0%  25%  50%  100%]

    for ax, (pct, time_ms, weights), label in zip(
            axes_flat, snapshots_data, _PANEL_LABELS):

        counts, edges = np.histogram(weights, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        width   = edges[1] - edges[0]

        ax.bar(centers, counts, width=width * 0.9,
               color=_PANEL_COLOR, edgecolor='black', alpha=0.82)

        mean_w = np.mean(weights)
        std_w  = np.std(weights)
        ax.axvline(mean_w, color='red', linestyle='--', lw=1.5,
                   label=f"μ = {mean_w:.4f}")

        ax.set_xlim(W_MIN, W_MAX)
        ax.set_xlabel("Peso Sináptico (w)", fontsize=10)
        ax.set_ylabel("Contagem de Sinapses", fontsize=10)
        ax.set_title(f"{label}   —   t = {time_ms} ms",
                     fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.45)
        ax.legend(fontsize=9, loc='upper right')

        ax.text(0.03, 0.97,
                f"σ = {std_w:.4f}\nN = {len(weights):,}",
                transform=ax.transAxes, fontsize=8.5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='lightgray', alpha=0.85),
                family='monospace')

    fig.suptitle(
        f"Distribuição de Pesos Sinápticos — 4 Momentos\n{run_label}",
        fontsize=11,
        y=0.97
    )

    # Ajuste fino das margens externas e do espaço interno.
    # Esses valores deixam a área branca menor sem sobrepor títulos/eixos.
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.90,
        hspace=0.22,
        wspace=0.18
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Painel 4 snapshots (2×2): {os.path.basename(output_path)}")


# ─────────────────────────────────────────────
# Arquivo de dados TXT
# ─────────────────────────────────────────────

def _save_data_txt(snapshots_data, run_label, output_path):
    lines = []
    sep  = "=" * 65
    sep2 = "-" * 65

    lines += [
        sep,
        "  SNAPSHOTS DE DISTRIBUIÇÃO DE PESOS — 4 MOMENTOS",
        sep,
        f"  {run_label}",
        "",
    ]

    for pct, time_ms, weights in snapshots_data:
        pct_str = f"{int(pct * 100)}%"
        lines += [
            sep2,
            f"  Momento : {pct_str:>4}  —  t = {time_ms} ms  ({time_ms/1000:.3f} s)",
            sep2,
            f"  N sinapses   : {len(weights):>10,}",
            f"  Média    (μ) : {np.mean(weights):>14.6f}",
            f"  DP       (σ) : {np.std(weights):>14.6f}",
            f"  Mínimo       : {np.min(weights):>14.6f}",
            f"  Máximo       : {np.max(weights):>14.6f}",
            f"  Mediana      : {np.median(weights):>14.6f}",
            f"  Percentil 25 : {np.percentile(weights, 25):>14.6f}",
            f"  Percentil 75 : {np.percentile(weights, 75):>14.6f}",
            "",
            "  Histograma (20 faixas):",
            f"  {'Faixa de w':^22}  {'Contagem':>8}  {'(%)':>6}",
        ]

        bins = np.linspace(W_MIN, W_MAX, 21)
        counts, edges = np.histogram(weights, bins=bins)
        for i in range(len(counts)):
            if counts[i] > 0:
                pct_bin = 100.0 * counts[i] / len(weights)
                lines.append(
                    f"  [{edges[i]:6.3f} – {edges[i+1]:6.3f}]  "
                    f"{counts[i]:>8,}  {pct_bin:>5.1f}%"
                )
        lines.append("")

    lines += [sep, "  Fim do relatório de snapshots de pesos", sep]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  [OK] Dados TXT: {os.path.basename(output_path)}")


# ─────────────────────────────────────────────
# Função pública
# ─────────────────────────────────────────────

def run_weight_snapshots(results_dir):
    """
    Ponto de entrada principal. Pode ser chamado por LIF_EMILLY.py ou
    diretamente pela linha de comando.

    Retorna o caminho da pasta criada, ou None se não houver snapshots.
    """
    params    = _read_params(results_dir)
    run_label = _build_run_label(params)

    pattern       = os.path.join(results_dir, "weights_t_*.npy")
    all_files     = sorted(glob.glob(pattern))

    if not all_files:
        print(f"[AVISO] Nenhum snapshot de pesos encontrado em: {results_dir}")
        return None

    selected = _select_four_snapshots(all_files)
    if not selected:
        print("[AVISO] Não foi possível selecionar snapshots.")
        return None

    if len(selected) < 4:
        print(f"[AVISO] Apenas {len(selected)} snapshot(s) disponíveis — continuando com o que há.")

    # Carregar pesos
    snapshots_data = []
    for pct, time_ms, fpath in selected:
        weights = np.load(fpath)
        snapshots_data.append((pct, time_ms, weights))
        print(f"  [INFO] Snapshot {int(pct*100):3d}% → t = {time_ms} ms  "
              f"(μ={np.mean(weights):.4f}, σ={np.std(weights):.4f})")

    # Criar pasta de saída
    out_dir = os.path.join(results_dir, "weight_snapshots_4pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Salvando em: {os.path.abspath(out_dir)}")

    # ── 4 figuras individuais ──────────────────────────────────────────
    for pct, time_ms, weights in snapshots_data:
        fname = f"snapshot_{int(pct * 100):03d}pct_t{time_ms:05d}.png"
        _plot_single(weights, time_ms, pct, run_label,
                     os.path.join(out_dir, fname))
        print(f"  [OK] Individual: {fname}")

    # ── Painel 2×2 ────────────────────────────────────────────────────
    panel_path = os.path.join(out_dir, "painel_4_snapshots.png")
    _plot_panel(snapshots_data, run_label, panel_path)

    # ── Arquivo de dados TXT ──────────────────────────────────────────
    txt_path = os.path.join(out_dir, "dados_4_snapshots.txt")
    _save_data_txt(snapshots_data, run_label, txt_path)

    print(f"[OK] weight_snapshots_4pt concluído.")
    return out_dir


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera snapshots de distribuição de pesos em 0%%, 25%%, 50%% e 100%% "
            "da simulação, painel 4-em-1 e relatório TXT."
        )
    )
    parser.add_argument("--dir", required=True,
                        help="Pasta de resultados da simulação.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERRO] Diretório não encontrado: {args.dir}")
        return
    run_weight_snapshots(args.dir)


if __name__ == "__main__":
    main()