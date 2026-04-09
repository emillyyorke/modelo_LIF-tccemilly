# plot_weight_evolution.py
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from SimulationParameters import W_MIN, W_MAX

def main():
    parser = argparse.ArgumentParser(description="Gera gráficos e dados de contagem de pesos a partir de uma pasta de simulação.")
    parser.add_argument("--dir", required=True, help="O caminho para a pasta de resultados da simulação específica.")
    args = parser.parse_args()

    results_dir = args.dir
    if not os.path.isdir(results_dir):
        print(f"[ERRO] Diretório não encontrado: {results_dir}")
        return

    # Pasta para os gráficos (como antes)
    plot_output_path = os.path.join(results_dir, "weight_distribution_plots")
    os.makedirs(plot_output_path, exist_ok=True)
    print(f"Salvando gráficos em: {os.path.abspath(plot_output_path)}")

    # ==========================================================
    # NOVO: Cria uma pasta para os arquivos de dados de contagem
    # ==========================================================
    data_output_path = os.path.join(results_dir, "weight_distribution_data")
    os.makedirs(data_output_path, exist_ok=True)
    print(f"Salvando dados de contagem em: {os.path.abspath(data_output_path)}")
    # ==========================================================

    file_pattern = "weights_t_*.npy"
    search_path = os.path.join(results_dir, file_pattern)
    snapshot_files = sorted(glob.glob(search_path))

    if not snapshot_files:
        print(f"[ERRO] Nenhum arquivo de snapshot ('{file_pattern}') encontrado em '{results_dir}'.")
        return

    weight_bins = np.linspace(W_MIN, W_MAX, 101)

    # Preparação para o gráfico resumo (mosaico)
    n_plots = len(snapshot_files)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols 
    fig_summary, axes_summary = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes_summary = axes_summary.flatten()
    
    # Itera sobre cada arquivo de snapshot
    for i, file_path in enumerate(snapshot_files):
        weights = np.load(file_path)
        try:
            time_ms = int(os.path.basename(file_path).split('_t_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"[AVISO] Não foi possível extrair o tempo do arquivo: {file_path}. Pulando.")
            continue

        # --- Parte 1: Calcula os dados do histograma UMA VEZ ---
        counts, bin_edges = np.histogram(weights, bins=weight_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # ==========================================================
        # NOVO: Salva os dados de contagem em um arquivo .txt
        # ==========================================================
        data_filename = f"counts_t_{time_ms:05d}.txt"
        data_save_path = os.path.join(data_output_path, data_filename)
        with open(data_save_path, 'w') as f:
            f.write("weight_bin_center\tsynapse_count\n")  # Cabeçalho
            for j in range(len(counts)):
                # Salva apenas as faixas que têm sinapses
                if counts[j] > 0:
                    f.write(f"{bin_centers[j]:.4f}\t\t{counts[j]}\n")
        print(f"[{i+1}/{n_plots}] Dados de contagem salvos em: {data_filename}")
        # ==========================================================

        # --- Parte 2: Gera o gráfico individual ---
        # (Usa os dados já calculados para desenhar)
        fig_individual, ax_individual = plt.subplots(figsize=(10, 6), dpi=100)
        ax_individual.bar(bin_centers, counts, width=bin_width*0.95, edgecolor='black', alpha=0.8)
        
        ax_individual.set_title(f"Distribuição dos Pesos Sinápticos em t = {time_ms} ms", fontsize=14)
        ax_individual.set_xlabel("Peso Sináptico (w)", fontsize=12)
        ax_individual.set_ylabel("Contagem de Sinapses", fontsize=12)
        ax_individual.set_xlim(W_MIN, W_MAX)
        ax_individual.grid(axis='y', linestyle='--', alpha=0.7)
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        stats_text = f"Total de Sinapses: {len(weights):,}\nMédia: {mean_w:.4f}\nDesvio Padrão: {std_w:.4f}"
        ax_individual.text(0.98, 0.95, stats_text, transform=ax_individual.transAxes, fontsize=10,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        output_filename = f"weight_dist_{time_ms:05d}.png"
        save_path = os.path.join(plot_output_path, output_filename)
        fig_individual.savefig(save_path, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"[{i+1}/{n_plots}] Gráfico individual gerado: {output_filename}")

        # --- Parte 3: Adiciona o mesmo gráfico ao mosaico ---
        ax_subplot = axes_summary[i]
        ax_subplot.bar(bin_centers, counts, width=bin_width*0.95, edgecolor='black', alpha=0.8)
        
        ax_subplot.set_title(f"t = {time_ms} ms", fontsize=10)
        ax_subplot.set_xlabel("Peso (w)", fontsize=8)
        ax_subplot.set_ylabel("Contagem", fontsize=8)
        ax_subplot.tick_params(axis='both', which='major', labelsize=7)
        ax_subplot.set_xlim(W_MIN, W_MAX)
        ax_subplot.grid(axis='y', linestyle='--', alpha=0.6)
        stats_text_small = f"Média: {mean_w:.3f}\nDP: {std_w:.3f}"
        ax_subplot.text(0.98, 0.98, stats_text_small, transform=ax_subplot.transAxes, fontsize=7,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    # Finaliza e salva o gráfico resumo
    for i in range(n_plots, len(axes_summary)):
        axes_summary[i].set_visible(False)

    fig_summary.suptitle("Evolução da Distribuição dos Pesos Sinápticos", fontsize=18, y=1.0)
    
    summary_save_path = os.path.join(plot_output_path, "weight_distribution_summary.png")
    fig_summary.savefig(summary_save_path, bbox_inches='tight', dpi=150)
    plt.close(fig_summary)
    
    print(f"\n[OK] Gráfico resumo salvo em: {summary_save_path}")
    print("[OK] Processo concluído.")

if __name__ == "__main__":
    main()
