import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter



# =====================================================
# Performance Profile (Dolan–Moré) — tabela direta
# =====================================================
def performance_profile_from_table(
    df,
    tau_max=None,
    n_tau=1000,
    save_path=None,
    title=None,
    dpi=300,
    show=False
):
    """
    Performance Profile para métricas de maximização.

    df:
        DataFrame onde:
        - colunas = métodos
        - linhas  = instâncias
    """

    # -------------------------
    # Dados
    # -------------------------
    methods = df.columns.tolist()
    values = df.values.astype(float)      # (N, M)
    values = values.T                     # (M, N)

    # Proteção contra zeros
    values = np.clip(values, 1e-12, None)

    # -------------------------
    # Razões de performance
    # -------------------------
    best = values.max(axis=0)              # melhor método por instância
    ratios = best / values                 # >= 1

    if tau_max is None:
        tau_max = ratios.max() * 1.05

    tau_grid = np.linspace(1.0, tau_max, n_tau)

    # -------------------------
    # PP e AUC
    # -------------------------
    pp = {}
    auc = {}

    for m, r in zip(methods, ratios):
        rho = [(r <= tau).mean() for tau in tau_grid]
        pp[m] = np.array(rho)
        auc[m] = np.trapezoid(pp[m], tau_grid)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8, 6))

    def normalize_name(m: str) -> str:
    # Caso 1: MBEEGNET → EEGNet
        if m == "eegnet":
            return "EEGNet"
        
        if m.startswith("multikernel_"):
            m = m.replace("multikernel_", "MK ", 1)

        if m.endswith("safka_csp"):
            m = m.replace("safka_csp", "SAFKA CSP", 1)

        if m == "safka":
            return "SAFKA"
        if m == "fbcsp":
            return "FBCSP"
        return m

    # Uso no loop
    for m in methods:
        label = normalize_name(m)
        plt.plot(tau_grid, pp[m], label=label)


    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\rho(\tau)$")
    plt.xlim(1, tau_max)
    plt.ylim(0, 1.01)
    # plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=18) 


    # Remover notação científica (×10^0) do eixo x
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))


    


    # if title:
    #     plt.title(title)
    # else:
    #     plt.title("Performance Profile")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")


    if show:
        plt.show()
    else:
        plt.close()

    return tau_grid, pp, auc


# =====================================================
# Média e desvio padrão
# =====================================================
def mean_std_from_table(df):
    mean = df.mean().to_dict()
    std = df.std(ddof=1).to_dict()
    return mean, std


# =====================================================
# EXEMPLO DE USO
# =====================================================

# # Se estiver em CSV
# df1 = pd.read_csv("resultsPaper/summary_comMIBIF2575_bciciv2a.csv")
# df2 = pd.read_csv("resultsPaper/summary_comMIBIF2575_bciciv2b.csv")
# df3 = pd.read_csv("resultsPaper/summary_comMIBIF2575_CBCIC.csv")
# df4 = pd.read_csv("resultsPaper/summary_comMIBIF_bciciv2a.csv")
# df5 = pd.read_csv("resultsPaper/summary_comMIBIF_bciciv2b.csv")
# df6 = pd.read_csv("resultsPaper/summary_comMIBIF_CBCIC.csv")


# #fazer concatenar colunas de df1, df2, df3 em df4 , df5, df6 respectivamente
# df1 = pd.concat([df1, df4],axis=1)
# df2 = pd.concat([df2, df5],axis=1)
# df3 = pd.concat([df3, df6],axis=1)
# print("DataFrames carregados.")
# print(df1)
# # Se já estiver em memória:
# # df = pd.DataFrame(...)
# dfs = [ df1, df2, df3]
# nomes = ["2a_comMIBIF", "2b_comMIBIF", "CBCIC_comMIBIF"]


df1 = pd.read_csv("results/tables_and_graphs/accuracy/2a_accuracies_Final.csv")
df2 = pd.read_csv("results/tables_and_graphs/accuracy/2b_accuracies_Final.csv")
df3 = pd.read_csv("results/tables_and_graphs/accuracy/cbcic_accuracies_Final.csv")
# retira a primeira coluna (sujeito)
df1 = df1.iloc[:, 1:]
df2 = df2.iloc[:, 1:]
df3 = df3.iloc[:, 1:]



dfs = [ df1, df2, df3]
nomes = ["2a", "2b", "CBCIC"]
# -------------------------
# Performance Profile
# -------------------------
counter = 0
for df in dfs:
    tau, pp, auc = performance_profile_from_table(
        df,
        save_path=f"results/tables_and_graphs/performance_profiles/summary_performance_profile{nomes[counter]}.pdf",
        title=f"Performance Profile Dataset {nomes[counter]}",
        show=False
    )

    # -------------------------
    # Estatísticas
    # -------------------------
    mean_acc, std_acc = mean_std_from_table(df)

    df_results = pd.DataFrame({
        "AUC_PP": auc,
        "Mean_Accuracy": mean_acc,
        "Std_Accuracy": std_acc
    }).sort_values("AUC_PP", ascending=False)

    df_results.to_csv(f"results/tables_and_graphs/performance_profiles/summary_results{nomes[counter]}.csv")
    print(df_results)
    counter += 1

# ========================
# All dataframes together
# ========================
all_df = pd.concat(dfs, ignore_index=True)
print("DataFrames concatenados para análise geral.")
print(all_df)
tau, pp, auc = performance_profile_from_table(
    all_df,
    save_path="results/tables_and_graphs/performance_profiles/summary_performance_profile_all_datasets.pdf",
    title="Performance Profile – All Datasets",
    show=False
)
