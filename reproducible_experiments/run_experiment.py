import os
import subprocess


def run_experiment(experiment_params, main_program_name, run_on_slurm, cpus=None, gpus=None):

    # & for windows or ; for mac
    if os.name == 'nt':
        separate = '&'
    else:
        separate = ';'


    if run_on_slurm:
        assert cpus is not None and gpus is not None
        slurm_prefix = f'srun -c {cpus} --gres=gpu:{gpus} -J plsNoKil'
    else:
        slurm_prefix = ''

    command = f'cd .. {separate} {slurm_prefix} python {main_program_name}.py'

    for param in list(experiment_params.keys()):
        command += f' --{param} {experiment_params[param]} '

    command += separate

    process = subprocess.Popen(command, shell=True)

    return process