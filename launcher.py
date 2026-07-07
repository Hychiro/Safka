import subprocess

methods = [
    # 'fbcsp',
    'eegnet',
    # 'safka_csp' ,
    # 'multikernel_safka_csp' ,
    'csp',
]


git_bash = r"C:/Program Files/Git/bin/bash.exe"

projeto = "C:/Users/Hychiro/Documents/Mestrado/Tese de Mestrado/Safka"
venv = "eegnet_tester/Scripts/activate"
for m in methods:
    comando = f'''
    cd "{projeto}"
    source "{venv}"
    py algorithims_script.py "{m}"
    exec bash
    '''

    subprocess.Popen(
        [git_bash, "-c", comando],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

print("Todos os processos foram iniciados.")