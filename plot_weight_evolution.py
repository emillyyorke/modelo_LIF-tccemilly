# plot_weight_evolution.py
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse 
from SimulationParameters import W_MIN, W_MAX


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos de distribuição de pesos a partir de uma pasta de simulação.")
    parser.add_argument("--dir", required=True, help="O caminho para a pasta de resultados da simulação específica.")
    args = parser.parse_args()

    results_dir = args.dir
    if not os.path.isdir(results_dir):
        print(f"[ERRO] Diretório não encontrado: {results_dir}")
        return

    # Cria uma subpasta para salvar os gráficos dentro do diretório da simulação
    output_path = os.path.join(results_dir, "weight_distribution_plots")
    os.makedirs(output_path, exist_ok=True)
    print(f"Salvando gráficos em: {os.path.abspath(output_path)}")

    # Encontra todos os arquivos de snapshot de pesos
    file_pattern = "weights_t_*.npy"
    search_path = os.path.join(results_dir, file_pattern)
    snapshot_files = sorted(glob.glob(search_path))

    if not snapshot_files:
        print(f"[ERRO] Nenhum arquivo de snapshot ('{file_pattern}') encontrado em '{results_dir}'.")
        return

    weight_bins = np.linspace(W_MIN, W_MAX, 101)

    for i, file_path in enumerate(snapshot_files):
        weights = np.load(file_path)
        try:
            time_ms = int(os.path.basename(file_path).split('_t_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"[AVISO] Não foi possível extrair o tempo do arquivo: {file_path}. Pulando.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.hist(weights, bins=weight_bins, edgecolor='black', alpha=0.8)
        ax.set_title(f"Distribuição dos Pesos Sinápticos em t = {time_ms} ms", fontsize=14)
        ax.set_xlabel("Peso Sináptico (w)", fontsize=12)
        ax.set_ylabel("Contagem de Sinapses", fontsize=12)
        ax.set_xlim(W_MIN, W_MAX)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        stats_text = f"Total de Sinapses: {len(weights):,}\nMédia: {mean_w:.4f}\nDesvio Padrão: {std_w:.4f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        output_filename = f"weight_dist_{time_ms:05d}.png"
        save_path = os.path.join(output_path, output_filename)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        print(f"[{i+1}/{len(snapshot_files)}] Gráfico gerado: {output_filename}")

    print("[OK] Processo concluído.")

if __name__ == "__main__":
    main()
