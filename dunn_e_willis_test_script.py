import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal
import scikit_posthocs as sp
import matplotlib.pyplot as plt

def correct_probability(df):
    """
    Retorna um vetor com a probabilidade do correto.
    Para cada linha:
    prob = df[ true_label ]
    """
    if "true_label" not in df.columns:
        raise ValueError("Coluna 'true_label' não encontrada")

    probs = []
    for _, row in df.iterrows():
        label = row["true_label"]
        if label not in df.columns:
            raise ValueError(f"Coluna da classe '{label}' não encontrada no CSV")
        probs.append(row[label])

    return probs

def collect_probabilities(
    dataset_name,
    keys,
    subjects,
    window_sizes,
    cue_starts,
    base_dir="results",
):
    """
    Retorna DataFrame:
    linhas  = trials (todos os sujeitos concatenados)
    colunas = algoritmos
    """
    data = {key: [] for key in keys}

    for key in keys:
        for subject in subjects:
            for ws in window_sizes:
                for cue in cue_starts:
                    filename = (
                        f"{key}_subject_{subject}_start_{cue:.2f}_window_{ws:.2f}"
                        .replace(".", "_") +".csv"
                    )

                    filepath = Path(base_dir) / dataset_name / filename
                    if not filepath.exists():
                        continue

                    df = pd.read_csv(filepath)
                    probs = correct_probability(df)

                    data[key].extend(probs)

    return pd.DataFrame(data)


def run_kruskal(df):
    samples = [df[col].dropna().values for col in df.columns]
    stat, p = kruskal(*samples)
    return stat, p

def df_to_dunn_arrays(df):
    """
    Converte DataFrame em:
    - X: lista de arrays (uma por coluna)
    - labels: nomes das colunas
    """
    X = []
    labels = []

    for col in df.columns:
        values = df[col].dropna().values  # remove NaNs
        if len(values) == 0:
            continue  # ignora colunas vazias

        X.append(values)
        labels.append(col)

    return X, labels


def run_dunn_from_arrays(X, labels, p_adjust="bonferroni"):
    """
    Executa Dunn a partir de lista de arrays.
    Retorna DataFrame com rótulos corretos.
    """
    dunn = sp.posthoc_dunn(X, p_adjust=p_adjust)
    dunn.index = labels
    dunn.columns = labels
    return dunn


def plot_dunn_heatmap(
    dunn_df,
    dataset_name = "All",
    alpha=0.05,
    cmap="viridis",
    figsize=(14, 12),
    annotate=False
):
    """
    Plota matriz de calor do teste de Dunn.

    Parâmetros:
    - dunn_df : DataFrame (p-valores)
    - alpha   : nível de significância
    - cmap    : mapa de cores
    - annotate: escreve p-valor nos blocos
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Máscara para diagonal
    mask = np.eye(len(dunn_df), dtype=bool)

    data = dunn_df.values.copy()
    data[mask] = np.nan

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Barra de cores
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("p-valor (Dunn)", fontsize=12)

    # Eixos
    ax.set_xticks(np.arange(len(dunn_df.columns)))
    ax.set_yticks(np.arange(len(dunn_df.index)))

    ax.set_xticklabels(dunn_df.columns, rotation=90)
    ax.set_yticklabels(dunn_df.index)

    # ax.set_title("Teste post-hoc de Dunn (p-valores)", fontsize=14)

    # Linhas separadoras
    ax.set_xticks(np.arange(-.5, len(dunn_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(dunn_df.index), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Anotação opcional dos p-valores
    if annotate:
        for i in range(len(dunn_df.index)):
            for j in range(len(dunn_df.columns)):
                if not mask[i, j] and not np.isnan(data[i, j]):
                    txt = f"{data[i, j]:.2e}" if data[i, j] < alpha else f"{data[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"results/tables_and_graphs/pvalue/{dataset_name}_DunnTestHeatMap_Final.png")
    plt.close()

datasets_cfg = {
    "2a": {"maxSubjects": 9},
    "2b": {"maxSubjects": 9},
    "cbcic":    {"maxSubjects": 10},
}

window_sizes = [2.0]
cue_starts = [2.5, 3.5]

keys = [
    'fbcsp',
    'eegnet',
    'safka_csp' ,
    'multikernel_safka_csp' ,
    'csp',
]

df_probs_all = []
for dataset_name, cfg in datasets_cfg.items():
    subjects = range(1, cfg["maxSubjects"] + 1)

    df_probs = collect_probabilities(
        dataset_name=dataset_name,
        keys=keys,
        subjects=subjects,
        window_sizes=window_sizes,
        cue_starts=cue_starts
    )
    
    df_probs_all.append(df_probs)
    df_probs.to_csv(f"results/tables_and_graphs/pvalue/dfprobs_{dataset_name}_Final.csv")
    print(f"\nDataset: {dataset_name}")
    # print(df_probs.head())

    # Kruskal–Wallis
    H, p_kw = run_kruskal(df_probs)
    print(f"Kruskal–Wallis: H = {H:.4f}, p = {p_kw:.4e}")

    # Dunn
    X, labels = df_to_dunn_arrays(df_probs)

    dunn_df = run_dunn_from_arrays(X, labels, p_adjust="bonferroni")

    print("Algoritmos:")
    print(labels)

    print("\nMatriz Dunn:")
    # print(dunn_df)

    plot_dunn_heatmap(
    dunn_df,
    dataset_name,
    cmap="inferno",
    annotate=False
    )
    # opcional: salvar
    df_probs.to_csv(f"results/tables_and_graphs/pvalue/{dataset_name}_probabilidades_Final.csv", index=False)
    dunn_df.to_csv(f"results/tables_and_graphs/pvalue/{dataset_name}_dunn_Final.csv")

all_df = pd.concat(df_probs_all, ignore_index=True)
all_df.to_csv(f"results/tables_and_graphs/pvalue/dfprobs_All_Final.csv")
print(f"\nDataset: All")
# print(df_probs.head())

# Kruskal–Wallis
H, p_kw = run_kruskal(all_df)
print(f"Kruskal–Wallis: H = {H:.4f}, p = {p_kw:.4e}")

# Dunn
X, labels = df_to_dunn_arrays(all_df)

dunn_df = run_dunn_from_arrays(X, labels, p_adjust="bonferroni")

print("Algoritmos:")
print(labels)

print("\nMatriz Dunn:")
# print(dunn_df)

plot_dunn_heatmap(
dunn_df,
cmap="inferno",
annotate=False
)
# opcional: salvar
df_probs.to_csv(f"results/tables_and_graphs/pvalue/All_probabilidades_Final.csv", index=False)
dunn_df.to_csv(f"results/tables_and_graphs/pvalue/All_dunn_Final.csv")