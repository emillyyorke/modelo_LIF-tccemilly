# plot_tlepis_comparison.py
"""
Gera um gráfico de T_LEPIS (tempo do último episódio) × Amplitude STDP
a partir de múltiplas pastas de resultados já simuladas.

T_LEPIS é lido do arquivo tlepis.txt gerado automaticamente pelo LIF_EMILLY.py.

Uso — escanear subpastas automaticamente:
    python plot_tlepis_comparison.py --root ./results

Uso — especificar pastas manualmente:
    python plot_tlepis_comparison.py --dirs ./results/sim_1 ./results/sim_2 ...

Opções:
    --output caminho/para/grafico.png   (padrão: tlepis_comparison.png na pasta raiz)
    --sort-by {a_ltp, a_ltd, amplitude} (padrão: a_ltp)
"""
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Leitura de dados
# ─────────────────────────────────────────────

def _read_tlepis(results_dir):
    """Lê tlepis.txt e retorna dict com os campos, ou None se não encontrado."""
    p = os.path.join(results_dir, "tlepis.txt")
    if not os.path.exists(p):
        return None
    data = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    try:
        return {
            'tlepis_s'   : float(data["T_LEPIS_S"]),
            'tlepis_ms'  : float(data["T_LEPIS_MS"]),
            'n_episodes' : int(data["N_EPISODES"]),
            'sim_time_s' : float(data.get("SIM_TIME_S", 0)),
        }
    except (KeyError, ValueError) as e:
        print(f"[AVISO] Erro ao ler tlepis.txt em {results_dir}: {e}")
        return None


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


def collect_data(dirs):
    """
    Varre as pastas e coleta (A_LTP, T_LEPIS, metadados).
    Retorna lista de dicts.
    """
    rows = []
    for d in sorted(dirs):
        tl = _read_tlepis(d)
        pr = _read_params(d)

        if tl is None:
            print(f"[AVISO] tlepis.txt ausente em: {os.path.basename(d)} — pulando.")
            continue

        stdp_on = pr.get("STDP_ENABLED", "False").lower() == "true"

        try:
            a_ltp = float(pr.get("A_LTP", "0"))
        except ValueError:
            a_ltp = 0.0
        try:
            a_ltd = float(pr.get("A_LTD", "0"))
        except ValueError:
            a_ltd = 0.0

        # "amplitude" = |A_LTP| (convenção: A_LTP positivo = LTP)
        amplitude = abs(a_ltp)

        rows.append({
            'dir'        : d,
            'label'      : os.path.basename(d),
            'a_ltp'      : a_ltp,
            'a_ltd'      : a_ltd,
            'amplitude'  : amplitude,
            'stdp_on'    : stdp_on,
            'stdp_mode'  : pr.get("STDP_MODE", "batch"),
            'tlepis_s'   : tl['tlepis_s'],
            'tlepis_ms'  : tl['tlepis_ms'],
            'n_episodes' : tl['n_episodes'],
            'sim_time_s' : tl['sim_time_s'],
            'mode'       : pr.get("FEEDBACK_MODE", "?").upper(),
        })

    print(f"[INFO] {len(rows)} simulações com tlepis.txt válido encontradas.")
    return rows


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_tlepis(rows, output_path, sort_by='a_ltp'):
    if not rows:
        print("[ERRO] Nenhum dado para plotar.")
        return

    # Separar STDP ON / OFF
    stdp_rows = [r for r in rows if r['stdp_on']]
    off_rows  = [r for r in rows if not r['stdp_on']]

    # Ordenar STDP ON pelo eixo escolhido
    stdp_rows.sort(key=lambda r: r[sort_by])

    # ── Figura ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Linha baseline STDP OFF (se houver)
    for r in off_rows:
        ax.axhline(r['tlepis_s'], color='gray', linestyle='--', lw=1.8,
                   zorder=2,
                   label=f"STDP OFF  (T$_{{LEPIS}}$ = {r['tlepis_s']:.1f} s)")

    # Curva STDP ON
    if stdp_rows:
        x_key = sort_by          # coluna do eixo X
        x_lbl = {
            'a_ltp'     : 'Amplitude LTP  (A_LTP)',
            'a_ltd'     : 'Amplitude LTD  (|A_LTD|)',
            'amplitude' : '|A_LTP|  (amplitude simétrica)',
        }.get(sort_by, sort_by)

        x = [r[x_key]      for r in stdp_rows]
        y = [r['tlepis_s'] for r in stdp_rows]
        n = [r['n_episodes'] for r in stdp_rows]

        # Linha de tendência + pontos
        ax.plot(x, y, 'o-', color='steelblue', lw=2.2, ms=8,
                zorder=3, label="STDP ON")

        # Anotações por ponto
        for xi, yi, ni in zip(x, y, n):
            ax.annotate(
                f"T={yi:.1f}s\n(n={ni})",
                xy=(xi, yi),
                xytext=(6, 5), textcoords='offset points',
                fontsize=7.5, color='steelblue',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='steelblue', alpha=0.7)
            )
    else:
        x_lbl = 'Amplitude STDP'

    # Metadados para o título
    mode_str    = rows[0]['mode'] if rows else "?"
    n_sims      = len(stdp_rows)
    stdp_mode_s = stdp_rows[0]['stdp_mode'].upper() if stdp_rows else "?"

    ax.set_xlabel(x_lbl, fontsize=12)
    ax.set_ylabel(r"$T_{LEPIS}$  (s)  —  tempo do último episódio", fontsize=12)
    ax.set_title(
        f"Tempo do Último Episódio × Amplitude STDP\n"
        f"Modo: {mode_str}  |  STDP: {stdp_mode_s}  |  {n_sims} simulações STDP ON",
        fontsize=11
    )

    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.45)

    if stdp_rows:
        xs = [r[x_key] for r in stdp_rows]
        margin = max((max(xs) - min(xs)) * 0.08, 1e-4)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        # Y começa em 0 para leitura intuitiva
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gráfico salvo: {output_path}")


# ─────────────────────────────────────────────
# Salvar tabela resumo TXT
# ─────────────────────────────────────────────

def save_summary_txt(rows, output_path):
    lines = [
        "=" * 75,
        "  RESUMO: T_LEPIS × AMPLITUDE STDP",
        "=" * 75,
        f"  {'Pasta (resumida)':^35}  {'A_LTP':>8}  "
        f"{'STDP':>5}  {'T_LEPIS(s)':>10}  {'N_ep':>5}  {'T_sim(s)':>8}",
        "-" * 75,
    ]
    for r in sorted(rows, key=lambda r: (not r['stdp_on'], r['a_ltp'])):
        label = r['label'][:35]
        stdp  = "ON"  if r['stdp_on'] else "OFF"
        lines.append(
            f"  {label:<35}  {r['a_ltp']:>8.5f}  "
            f"{stdp:>5}  {r['tlepis_s']:>10.2f}  {r['n_episodes']:>5}  "
            f"{r['sim_time_s']:>8.1f}"
        )
    lines += ["=" * 75, ""]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"[OK] Tabela resumo: {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plota T_LEPIS × amplitude STDP para múltiplas simulações. "
            "Cada pasta de resultados deve conter tlepis.txt e params.txt."
        )
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--root",
        help="Pasta raiz dos resultados. Escaneia subpastas com tlepis.txt."
    )
    grp.add_argument(
        "--dirs", nargs='+',
        help="Lista de pastas de resultados individuais."
    )
    parser.add_argument(
        "--output", default=None,
        help="Caminho do PNG de saída (padrão: tlepis_comparison.png na pasta raiz)."
    )
    parser.add_argument(
        "--sort-by", default='a_ltp',
        choices=['a_ltp', 'a_ltd', 'amplitude'],
        help="Variável do eixo X (padrão: a_ltp)."
    )
    args = parser.parse_args()

    # ── Encontrar pastas ──────────────────────────────────────────────
    if args.root:
        root = args.root
        pattern = os.path.join(root, "*", "tlepis.txt")
        found = glob.glob(pattern)
        dirs = [os.path.dirname(f) for f in found]
        if not dirs:
            print(f"[AVISO] Nenhuma subpasta com tlepis.txt em: {root}")
            return
    else:
        root = os.path.commonpath(args.dirs) if len(args.dirs) > 1 else os.path.dirname(args.dirs[0])
        dirs = args.dirs

    rows = collect_data(dirs)
    if not rows:
        print("[ERRO] Nenhum dado válido para plotar.")
        return

    # ── Saída ─────────────────────────────────────────────────────────
    out_png = args.output or os.path.join(root, "tlepis_comparison.png")
    out_txt = out_png.replace(".png", "_tabela.txt")

    plot_tlepis(rows, out_png, sort_by=args.sort_by.replace('-', '_'))
    save_summary_txt(rows, out_txt)

    print(f"\n[INFO] {len(rows)} simulação(ões) incluída(s).")
    print(f"  Gráfico  : {out_png}")
    print(f"  Tabela   : {out_txt}")


if __name__ == "__main__":
    main()
    