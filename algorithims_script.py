from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf import EMD, filterbank, cubic_resample,fft_resample, wavelet 
from bciflow.modules.fs import MIBIF
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.analysis.metric_functions import accuracy
from methods.safmkaFinal import safka_filter
from methods.eegnet import Eegnet
from methods.bandpass import bandpass_conv
import numpy as np
import time
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import matplotlib.pyplot as plt

# rodaScriptArtigo.py
import sys

if len(sys.argv) < 2:
    raise ValueError("Informe o nome do método")

metodo = sys.argv[1]
print(f"Rodando método: {metodo}")

accDict = {metodo: []}
for dataset_name in ['2a','2b','cbcic']:
    print(f"Rodando Dataset: {dataset_name}")
    if dataset_name == 'cbcic':
        dataset = cbcic
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC'
        maxSubjects = 11 #1 a 10
    elif dataset_name == '2a':
        dataset = bciciv2a
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2a'
        maxSubjects = 10 #1 a 9
    elif dataset_name == '2b':
        dataset = bciciv2b
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2b'
        maxSubjects = 10 #1 a 9
    for key in accDict.keys():
            time_used = []
            kernels_all_subjects = []
            for i in range(1,maxSubjects):
                start = time.time()
                data = dataset(subject=i, path=path, labels=['left-hand', 'right-hand'])
                if key == 'fbcsp':
                    pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}

                    sf = csp()
                    fe = logpower
                    fs = MIBIF(8, clf=lda())
                    clf = lda()

                    pos_folding = {
                        'sf': (sf, {}),
                        'fe': (fe, {'flating': True}),
                        'fs': (fs, {}),
                        'clf': (clf, {})
                    }
                elif key == 'safka_csp':
                    pre_folding = {}

                    sf = csp()
                    fe = logpower
                    
                    clf = lda()

                    pos_folding = {
                        'tf': (safka_filter(a =1,b=0,subject=i,itrs=500,
                                                    alpha=0.995,sigma0=0.2,kernel_size=65,
                                                    collect_traces=False),{}),
                        'sf': (sf, {}),
                        'fe': (fe, {'flating': True}),
                        
                        'clf': (clf, {})
                    }
                elif key == 'multikernel_safka_csp':
                    pre_folding = {}

                    sf = csp()
                    fe = logpower
                    fs = MIBIF(8, clf=lda())
                    clf = lda()

                    pos_folding = {
                        'tf': (safka_filter(num_kernels = 9,a =1,b=0,subject=i,itrs=500,
                                                    alpha=0.995,sigma0=0.2,kernel_size=65,
                                                    collect_traces=False),{}),
                        'sf': (sf, {}),
                        'fe': (fe, {'flating': True}),
                        'fs': (fs, {}),
                        'clf': (clf, {})
                    }
                elif key == 'csp':
                    
                    pre_folding = {'tf': (bandpass_conv, {})}

                    sf = csp()
                    fe = logpower
                    clf = lda()

                    pos_folding = {
                        'sf': (sf, {}),
                        'fe': (fe, {'flating': True}),
                        'clf': (clf, {})
                    }
                elif key == 'eegnet':

                    pre_folding = {
                    'tf1': (bandpass_conv, {}),
                    'tf2': (cubic_resample,{'new_sfreq':128.0}),
                    }
                    pos_folding = {
                        'clf': (Eegnet(),{})
                    }
                    
                ws =2.0
                start = time.time()
                results = kfold(
                    target=data,
                    start_window=data['events']['cue'][0] + 0.5,
                    window_size=2.0,
                    pre_folding=pre_folding,
                    pos_folding=pos_folding
                )
                end = time.time()
                print(f"Time taken for subject {i} with filter {key}: {end - start:.2f} seconds")
                time_used.append(end - start)
                
                filename = f"results/{dataset_name}/{key}_subject_{i}_start_{data['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}".replace(".", "_") + ".csv"
                results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *data["y_dict"].keys()])
                df = pd.DataFrame(results)
                acc = accuracy(results)
                print(f"Accuracy for subject {i} with filter {key}: {acc:.4f}")
                accDict[key].append(acc)
                if key == 'safka_csp' or key == 'multikernel_safka_csp':
                    # save kernels for the subject  then put them in csv file, each file in a folder named by dataset and filter name
                    kernels = pos_folding['tf'][0].bestWs
                    kernels_all_subjects.append(kernels)




            total_time = sum(time_used)
            time_per_subject = total_time / len(time_used)
            dict_time = {'subject': list(range(1, maxSubjects)), 'time_taken': time_used}
            dict_time['subject'].append("total_time")
            dict_time['time_taken'].append(total_time)
            dict_time['subject'].append("time_per_subject")
            dict_time['time_taken'].append(time_per_subject)
            print(f"size of time_used: {len(dict_time['time_taken'])} and size of subject: {len(dict_time['subject'])}")
            dataset_time = pd.DataFrame(dict_time)            
            dataset_time.to_csv(f"results/tables_and_graphs/times/{dataset_name}_{key}_time_data.csv", index=False)
            if key == 'safka_csp' or key == 'multikernel_safka_csp':
                dict_kernels = {'subject': list(range(1, maxSubjects)), 'best_kernels': kernels_all_subjects}
                dataset_kernels = pd.DataFrame(dict_kernels)
                dataset_kernels.to_csv(f"results/tables_and_graphs/best_kernels/{dataset_name}_{key}_kernel_data.csv", index=False)
    
    df = pd.DataFrame(accDict)
    df.to_csv(f"results/tables_and_graphs/summary_{dataset_name}_{key}.csv", index=False)
print("Processo concluído.")