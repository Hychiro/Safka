import pandas as pd
from pathlib import Path
from bciflow.modules.analysis.metric_functions import accuracy


def collect_accuracies(
    dataset_name,
    keys,
    subjects,
    window_sizes,
    cue_starts,
    base_dir="results",
    acc_column="accuracy"
):
    """
    Lê resultados existentes e retorna um dicionário:
    { metodo: [acc_s1, acc_s2, ...] }
    """
    acc_dict = {key: [] for key in keys}

    for key in keys:
        for subject in subjects:
            acc_subject = []

            for ws in window_sizes:
                for cue in cue_starts:
                    start = cue

                    filename = (
                        f"{key}_subject_{subject}_start_{start:.2f}_window_{ws:.2f}"
                        .replace(".", "_") +".csv"
                    )

                    filepath = Path(base_dir) / dataset_name / filename

                    if not filepath.exists():
                        continue

                    df = pd.read_csv(filepath)

                    acc = accuracy(df)
                    print(f"{filepath}: {acc:.4f}")
                    # Assume uma acurácia por arquivo
                    acc_subject.append(acc)

            # média do sujeito (caso haja múltiplas janelas)
            if len(acc_subject) > 0:
                acc_dict[key].append(sum(acc_subject) / len(acc_subject))
            else:
                acc_dict[key].append(None)

    return acc_dict


def save_accuracy_csv(acc_dict, dataset_name, output_dir="results/tables_and_graphs/accuracy"):
    """
    Salva um CSV onde:
    linhas = sujeitos
    colunas = métodos
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(acc_dict)
    df.index.name = "subject"

    out_path = Path(output_dir) / f"{dataset_name}_accuracies_Final.csv"
    df.to_csv(out_path)

    print(f"Salvo: {out_path}")
    return df


datasets_cfg = {
    "2a": {"maxSubjects": 9},
    "2b": {"maxSubjects": 9},
    "cbcic":    {"maxSubjects": 10},
}

methods = [
    'fbcsp',
    'eegnet',
    'safka_csp' ,
    'multikernel_safka_csp' ,
    'csp',
]
window_sizes = [2.0]      # ajuste se necessário
cue_starts = [2.5, 3.5]        # ajuste se necessário
for dataset_name, cfg in datasets_cfg.items():
    subjects = range(1, cfg["maxSubjects"] + 1)

    acc_dict = collect_accuracies(
        dataset_name=dataset_name,
        keys=methods,
        subjects=subjects,
        window_sizes=window_sizes,
        cue_starts=cue_starts
    )

    df_acc = save_accuracy_csv(acc_dict, dataset_name)
    print(df_acc.head())
