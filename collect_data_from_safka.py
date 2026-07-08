import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from methods.bandpass import bandpass_conv, bandpass_conv_kernel
from bciflow.modules.core.kfold import kfold
from bciflow.modules.fs import MIBIF
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.analysis.metric_functions import accuracy
from methods.safmkaFinal2 import safka_filter
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda


def safka_starter(subject,num_kernels,a =1,b=0,itrs=500,
                  alpha=0.995,sigma0=0.2,kernel_size=65,
                  collect_traces=False,seed =42):
    pre_folding = {}
    pos_folding = {}
    if num_kernels == 1:
        pre_folding = {}

        sf = csp()
        fe = logpower
        
        clf = lda()

        pos_folding = {
            'tf': (safka_filter(num_kernels=num_kernels,a =a,b=b,subject=subject,
                                itrs=itrs,alpha=alpha,sigma0=sigma0,seed=seed,
                                kernel_size=kernel_size,collect_traces=collect_traces),{}),
            'sf': (sf, {}),
            'fe': (fe, {'flating': True}),
            
            'clf': (clf, {})
        }
    else:
        pre_folding = {}

        sf = csp()
        fe = logpower
        fs = MIBIF(8, clf=lda())
        clf = lda()

        pos_folding = {
            'tf': (safka_filter(num_kernels=num_kernels,a =a,b=b,subject=subject,
                                itrs=itrs,alpha=alpha,sigma0=sigma0,seed=seed,
                                kernel_size=kernel_size,collect_traces=collect_traces),{}),
            'sf': (sf, {}),
            'fe': (fe, {'flating': True}),
            'fs': (fs, {}),
            'clf': (clf, {})
        }
    return pre_folding, pos_folding


# rodaScriptArtigo.py
import sys

if len(sys.argv) < 2:
    raise ValueError("Informe o nome do método")

experiment = sys.argv[1]
num_kernels = int(sys.argv[2])
itrs = int(sys.argv[3])
alpha = float(sys.argv[4])
sigma0 = float(sys.argv[5])
kernel_size = int(sys.argv[6])
collect_traces = bool((sys.argv[7]))
seed = int(sys.argv[8])
print(f"Rodando tipo do experimento: {experiment}")
EXPERIMENTS = {
    "convergence": f"{experiment}_seed_{seed}",
    "kernel_size": f"{experiment}_kernel_size_{kernel_size}_seed{seed}",
    "num_kernels": f"{experiment}_num_kernels_{num_kernels}_seed{seed}",
    "sigma": f"{experiment}_sigma_{sigma0}_seed{seed}",
    "alpha": f"{experiment}_alpha_{alpha}_seed{seed}",
}
key = EXPERIMENTS[experiment]
accDict = {key: []}


for dataset_name in ['2b','cbcic']:
    print(f"Rodando Dataset: {dataset_name}")
    if dataset_name == 'cbcic':
        dataset = cbcic
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC'
        subjects = 10 #1 a 10
    elif dataset_name == '2a':
        dataset = bciciv2a
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2a'
        subjects = 1 #1 a 9
    elif dataset_name == '2b':
        dataset = bciciv2b
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2b'
        subjects = 7 #1 a 9
    for key in accDict.keys():
            time_used = []
            kernels_all_subjects = []
            
            data = dataset(subject=subjects, path=path, labels=['left-hand', 'right-hand'])
            data['subj'] = subjects
            pre_folding, pos_folding = safka_starter(subject=subjects,num_kernels=num_kernels,itrs=itrs,
                  alpha=alpha,sigma0=sigma0,kernel_size=kernel_size,
                  collect_traces=collect_traces,seed =seed)
            ac = pos_folding["tf"][0]
            ws = 2.0
            start = time.time()
            results = kfold(
                target=data,
                start_window=data['events']['cue'][0] + 0.5,
                window_size=2.0,
                pre_folding=pre_folding,
                pos_folding=pos_folding
            )
            
            filename = f"results/safka_analysis/{dataset_name}/{key}_subject_{subjects}_start_{data['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}".replace(".", "_") +".csv"

            results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *data["y_dict"].keys()])
            df = pd.DataFrame(results)
            acc = accuracy(results)
            print(f"Accuracy for subject {subjects} with filter {key}: {acc:.4f}")
            plt.figure()
            for eachFold in range(0,5):
                energies = ac.explorationEnergy[f"subject_{subjects}"][eachFold]
                plt.plot(energies, label=f'Fold {eachFold}')
            plt.legend(f"fold {eachFold}")
            plt.title('SAFKA Energy Trace')
            plt.xlabel('Iterations')
            plt.ylabel('Energy')
            plt.legend()
            plt.savefig(f"results/safka_analysis/tables_and_graphs/convergence/{experiment}/{dataset_name}_{key}_subject_{subjects}_fold_{eachFold}_energy_trace.png")
            plt.close()
            plt.figure()
            for eachFold in range(0,5):
                energies = ac.bestEnergy[f"subject_{subjects}"][eachFold]
                plt.plot(energies, label=f'Fold {eachFold}')
            plt.legend(f"fold {eachFold}")
            plt.title('SAFKA Best Energy Trace')
            plt.xlabel('Iterations')
            plt.ylabel('Energy')
            plt.legend()
            plt.savefig(f"results/safka_analysis/tables_and_graphs/convergence/{experiment}/{dataset_name}_{key}_subject_{subjects}_fold_{eachFold}_best_energy_trace.png")
            plt.close()
            plt.figure()
            for eachFold in range(0,5):
                energies = ac.bestLogLoss[f"subject_{subjects}"][eachFold]
                plt.plot(energies, label=f'Fold {eachFold}')
            plt.legend(f"fold {eachFold}")
            plt.title('SAFKA Best LogLoss Trace')
            plt.xlabel('Iterations')
            plt.ylabel('Energy')
            plt.legend()
            plt.savefig(f"results/safka_analysis/tables_and_graphs/convergence/{experiment}/{dataset_name}_{key}_subject_{subjects}_fold_{eachFold}_best_logloss_trace.png")
            plt.close()
            plt.figure()
            for eachFold in range(0,5):
                energies = ac.bestSeparation[f"subject_{subjects}"][eachFold]
                plt.plot(energies, label=f'Fold {eachFold}')
            plt.legend(f"fold {eachFold}")
            plt.title('SAFKA Best Separation Trace')
            plt.xlabel('Iterations')
            plt.ylabel('Energy')
            plt.legend()
            plt.savefig(f"results/safka_analysis/tables_and_graphs/convergence/{experiment}/{dataset_name}_{key}_subject_{subjects}_fold_{eachFold}_best_separation_trace.png")
            plt.close()

print("Processo concluído.")