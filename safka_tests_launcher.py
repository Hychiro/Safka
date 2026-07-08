import subprocess
import time

BEST_ITRS = 500

convergence_experiments = {
    f"seed_{i}": {
        "num_kernels": 1,
        "itrs": 2000,
        "alpha": 0.995,
        "sigma0": 10,
        "kernel_size": 65,
        "collect_traces": True,
        "seed": i
    }
    for i in range(30)
}

kernel_size_experiments = {
    f"kernel_{k}": {
        "num_kernels": 1,
        "itrs": BEST_ITRS,
        "alpha": 0.995,
        "sigma0": 10,
        "kernel_size": k,
        "collect_traces": True,
        "seed": 42
    }
    for k in  [
    3, 5, 7, 9, 11,
    13, 17, 21, 25, 33,
    41, 49, 57, 65, 81,
    97, 113, 129, 161, 193
    ]
}

num_kernel_experiments = {
    f"kernels_{n}": {
        "num_kernels": n,
        "itrs": BEST_ITRS,
        "alpha": 0.995,
        "sigma0": 10,
        "kernel_size": 65,
        "collect_traces": True,
        "seed": 42
    }
    for n in [1, 2, 4, 8, 10, 15, 20]
}

sigma_experiments = {
    f"sigma_{s}": {
        "num_kernels": 1,
        "itrs": BEST_ITRS,
        "alpha": 0.995,
        "sigma0": s,
        "kernel_size": 65,
        "collect_traces": True,
        "seed": 42
    }
    for s in [0.05, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10, 25, 50, 75, 100]
}

alpha_experiments = {
    f"alpha_{a}": {
        "num_kernels": 1,
        "itrs": BEST_ITRS,
        "alpha": a,
        "sigma0": 10,
        "kernel_size": 65,
        "collect_traces": True,
        "seed": 42
    }
    for a in [0.5,0.6,0.75,0.8,0.9,0.95,0.98, 0.985, 0.99, 0.995, 0.997, 0.999]
}

# ======================================================
# Todos os experimentos
# ======================================================

EXPERIMENTS = {
    "convergence": convergence_experiments,
    "kernel_size": kernel_size_experiments,
    "num_kernels": num_kernel_experiments,
    "sigma": sigma_experiments,
    "alpha": alpha_experiments,
}

# SELECTED_EXPERIMENT = "convergence"
# SELECTED_EXPERIMENT = "kernel_size"
# SELECTED_EXPERIMENT = "num_kernels"
# SELECTED_EXPERIMENT = "sigma"
SELECTED_EXPERIMENT = "alpha"

git_bash = r"C:/Program Files/Git/bin/bash.exe"

projeto = "C:/Users/Hychiro/Documents/Mestrado/Tese de Mestrado/Safka"
venv = "eegnet_tester/Scripts/activate"

experiments = EXPERIMENTS[SELECTED_EXPERIMENT]

processes = []
i = 1
for name, params in experiments.items():
    # print(params)
    comando = f'''
    cd "{projeto}"
    source "{venv}"
    py collect_data_from_safka.py "{SELECTED_EXPERIMENT}" '{params["num_kernels"]}' '{params["itrs"]}' '{params["alpha"]}' '{params["sigma0"]}' '{params["kernel_size"]}' '{params["collect_traces"]}' '{params["seed"]}'
    exec bash
    '''
    print(comando)
    p = subprocess.Popen(
        [git_bash, "-c", comando],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    processes.append(p)
    i=i+1
    if i % 10 == 0 and i < len(experiments):
        print("10 experimentos iniciados. Aguardando 10 minutos...")
        time.sleep(10 * 60)   # 600 segundos
    

print(f"{len(processes)} processos iniciados.")