from itertools import product
import time
from run_experiment import run_experiment



def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


real_datasets = ['tetuan_power', 'energy', 'traffic', 'wind', 'prices']
syn_datasets = ['window_x_dim_5_z_dim_3_len_500']

processes_to_run_in_parallel = 1

seeds = list(range(0, 20))
suppress_plots = 1
alphas = [0.1]

real_main_baseline_params = {
    'main_program_name': ['main'],
    'seed': seeds,
    'dataset_name': real_datasets,
    'alpha': alphas,
    'ds_type': ['REAL'],
    'lr': [1e-4],
    'lstm_layers': [1],
    'bs': [1],
    'lstm_in_layers': ['"[32, 64]"'],
    'lstm_out_layers': ['"[32]"', ],
    'lstm_hidden_size': [64, ],
    'suppress_plots': [suppress_plots],
    'train_all_q': [0],
    'cal_split': [0, 1],
    'use_best_hyperparams': [1],

}

syn_main_baseline_params = {
    'main_program_name': ['main'],
    'seed': seeds,
    'dataset_name': syn_datasets,
    'alpha': alphas,
    'ds_type': ['SYN'],
    'lr': [1e-4],
    'lstm_layers': [1],
    'bs': [1],
    'lstm_in_layers': ['"[32, 64]"'],
    'lstm_out_layers': ['"[32]"', ],
    'lstm_hidden_size': [64, ],
    'suppress_plots': [suppress_plots],
    'train_all_q': [0],
    'cal_split': [0, 1],
    'use_best_hyperparams': [1],

}

mqr_experiment_params = {
    'main_program_name': ['main'],
    'method_type': ['mqr'],
    'seed': [0],
    'dataset_name': ['tetuan_power'],
    'alpha': alphas,
    'ds_type': ['real'],
    'suppress_plots': [0],
    'train_all_q': [0],
    'cal_split': [0],
    'use_best_hyperparams': [1],

}

params = list(cartesian_product(real_main_baseline_params))
params += list(cartesian_product(syn_main_baseline_params))
params += list(cartesian_product(mqr_experiment_params))

processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))

if __name__ == '__main__':

    print("jobs to do: ", len(params))
    # initializing proccesses_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        main_program_name = curr_params['main_program_name']
        curr_params.pop('main_program_name')
        p = run_experiment(curr_params, main_program_name, run_on_slurm=False)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                main_program_name = curr_params['main_program_name']
                curr_params.pop('main_program_name')
                p = run_experiment(curr_params, main_program_name, run_on_slurm=False)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
            time.sleep(10)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
