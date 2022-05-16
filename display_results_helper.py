import copy
import itertools
import os
from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import re
import seaborn as sns
from helper import create_folder_if_it_doesnt_exist
from hyperparameters import get_best_hyperparams, get_best_gamma, Hyperparameters, CALIBRATION_METHODS


def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))


def atoi(text):
    return float(text) if '.' in text else int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+\.?\d*)', text)]


def get_base_path(is_real, dataset_name, hyperparameters: Hyperparameters, figures=False):
    data_type_addition = 'real' if is_real else 'syn'
    figures_addition = '/figures' if figures else ''
    base_path = f'results/test{figures_addition}/{data_type_addition}/{dataset_name}'
    return base_path


def get_baseline_hyperparams(hyperparams):
    if hyperparams == 'all':
        baseline_lr = [1e-4, 5e-4]
        baseline_lstm_in_layers = ['[32, 64, 128]', '[32]', '[32, 64]']
        baseline_lstm_out_layers = ['[128, 64, 32]', '[32]', '[64]', '[64, 32]']
        baseline_lstm_hd = [64, 128]
        baseline_lstm_nl = [1]
    elif hyperparams == 'syn':
        baseline_lr = [1e-4, 5e-4]
        baseline_lstm_in_layers = ['[32, 64, 64]', '[32, 64, 128]', '[32]', '[32, 64]']
        baseline_lstm_out_layers = ['[128, 64, 32]', '[32]', '[64]', '[64, 32]', '[128]']
        baseline_lstm_hd = [64, 128]
        baseline_lstm_nl = [1]
    elif hyperparams == 'real':
        baseline_lr = [1e-4, 5e-4]
        baseline_lstm_in_layers = ['[32, 64, 128]', '[32]', '[32, 64]']
        baseline_lstm_out_layers = ['[32]', '[64]', '[64, 32]', '[128]']
        baseline_lstm_hd = [64, 128]
        baseline_lstm_nl = [1]
    else:
        raise Exception("hyperparams must be one of: 'syn', 'real' or 'all'")

    baseline_hyperparams = {
        'lr': baseline_lr,
        'lstm_hd': baseline_lstm_hd,
        'lstm_nl': baseline_lstm_nl,
        'lstm_in_hd': baseline_lstm_in_layers,
        'lstm_out_hd': baseline_lstm_out_layers,
        'train_all_q': [1],
        'calibration_method': [
            'ACI+CQR',
            'RCI+CQR with cal',
            'RCI_Y',
            'RCI_Stretched_Y',
            'RCI_Stretched_Exp_e_Y',
            'RCI_Stretched_Exp_5_Y',
            'RCI_Tau',
            'RCI_Stretched_Tau',
        ],
        'cal_split': [1],
        # 'gamma': [0.05]
        'gamma': [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    }

    return baseline_hyperparams


def read_method_results(folder_path=None, seeds=20, apply_mean=True, base_path=None,
                        hyperparameters: Hyperparameters = None, display_errors=False):
    df = read_method_results_aux(folder_path, seeds, apply_mean, display_errors=display_errors)

    if base_path is not None and hyperparameters is not None and hyperparameters.is_calibrated:
        try:
            uncalibrated = deepcopy(hyperparameters)
            uncalibrated.is_calibrated = False
            uncalibrated_path = f"{base_path}/{uncalibrated.to_folder_name()}"
            non_calibrated_df = read_method_results_aux(uncalibrated_path, seeds=seeds, apply_mean=apply_mean,
                                                        display_errors=display_errors)
            df['uncalibrated coverage'] = non_calibrated_df['Estimated quantiles coverage']
        except Exception as e:
            # print("got here")
            print("didn't found uncalibrated df: ", e)
    return df


def read_method_results_aux(folder_path, seeds=20, apply_mean=True, display_errors=False):
    df = pd.DataFrame()

    for seed in range(seeds):
        save_path = f"{folder_path}/seed={seed}.csv"
        try:
            seed_df = pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')

            if 'z r2 score' in seed_df.columns:
                seed_df['z r2 score (%)'] = seed_df['z r2 score'] * 100
                seed_df = seed_df.drop(['z r2 score'], axis=1, errors='ignore')

            for col in ['test predictor_loss', 'test kl_loss', 'test rec_loss', 'Time HSIC dependence',
                        'True z Time HSIC dependence']:
                if col in seed_df.columns:
                    seed_df[f'log {col}'] = np.log(seed_df[col])
                    seed_df = seed_df.drop([col], axis=1, errors='ignore')
            if 'calibration Q' in seed_df.columns:
                seed_df['abs(calibration Q)'] = abs(seed_df['calibration Q'])

            if abs(seed_df['Estimated quantiles coverage'].item() - 0) < 0.01:
                # print(f"{folder_path}/seed={seed}.csv has 0 coverage")
                if np.isnan(seed_df['Estimated quantiles average length']).any():
                    print(
                        f"{folder_path}/seed={seed}.csv has invalid average length. the value is: {seed_df['Estimated quantiles average length'].item()}")
                    display(seed_df)
                    # print("got here")
                    continue
            if 'Estimated quantiles (miscoverage streak) average length' in df.columns and \
                    np.isnan(seed_df['Estimated quantiles (miscoverage streak) average length']).any():
                print(
                    f"{folder_path}/seed={seed}.csv has invalid Estimated quantiles (miscoverage streak) average length")
                print("the value is: ", seed_df['Estimated quantiles (miscoverage streak) average length'].item())
                display(seed_df)

            df = pd.concat([df, seed_df], axis=0)
        except Exception as e:
            # print("got an exception")
            if display_errors:
                print(e)

    if len(df) == 0:
        # print(f"{folder_path} had 0 an error")
        save_path = f"{folder_path}/seed=0.csv"
        pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')  # raises an exception
        raise Exception(f"could not find results in path {folder_path}")

    if apply_mean:
        df = df.apply(np.mean).to_frame().T

    return df


def get_method_name_to_display(method, method_folder, show_full_params, params_to_show, params: Hyperparameters):
    if show_full_params:
        method_name_to_display = method_folder
    else:
        method_name_to_display = method.replace("_", " ")
        for param_to_show in params_to_show:
            try:
                param = params[
                    param_to_show]
                method_name_to_display += f" {param_to_show}={param}"
            except Exception:
                pass

    if params.is_calibrated:
        method_name_to_display = f'calibrated {params.calibration_method} ' + method_name_to_display

    return method_name_to_display


def get_method_df(base_path, method, hyperparams, seeds, show_full_params, nTop, params_to_show,
                  is_calibrated=False,
                  sort_by='Estimated quantiles average length',
                  display_errors=False,
                  method_hyperparams=None):
    hyperparams = get_baseline_hyperparams(hyperparams)

    if method_hyperparams is not None:
        hyperparams = {**hyperparams, **method_hyperparams}

    df = pd.DataFrame()
    cols = []

    folders_done = {}
    for params in cartesian_product(hyperparams):
        try:
            params['is_calibrated'] = is_calibrated
            params = Hyperparameters(**params)
            params.method_type = method
            method_folder = params.to_folder_name()
            # print(method_folder)
            if method_folder in folders_done:
                continue
            folder_path = f"{base_path}/{method_folder}"
            model_df = read_method_results(folder_path, seeds=seeds,
                                           base_path=base_path,
                                           hyperparameters=params,
                                           display_errors=display_errors)
            df = pd.concat([df, model_df], axis=0)
            method_name_to_display = get_method_name_to_display(method, method_folder, show_full_params, params_to_show,
                                                                params)
            folders_done[method_folder] = 1
            cols += [f'{method_name_to_display}']

        except Exception as e:
            pass

    if len(df) > 0:
        df.index = cols
        df = df.T
        if sort_by in df.index:
            df = df.sort_values([sort_by], axis=1)
        elif callable(sort_by):
            df = sort_by(df)
        else:
            df = df.sort_values(['Estimated quantiles average length'], axis=1)

        df = df.T[:nTop].T.T
    return df


def get_results_df(dataset_name, ds_type, is_offline, is_calibrated, nTop=1, show_full_params=False,
                   params_to_show=None,
                   seeds=20, hyperparams='all', sort_by='name', display_errors=False,
                   method_hyperparams=None):
    if params_to_show is None:
        params_to_show = []
    base_path = f'results/test/{ds_type}/{dataset_name}'

    df = get_method_df(base_path, 'baseline', hyperparams, seeds, show_full_params, nTop,
                       params_to_show=params_to_show, is_calibrated=is_calibrated,
                       sort_by=sort_by,
                       display_errors=display_errors,
                       method_hyperparams=method_hyperparams)

    if len(df) == 0:
        return df

    df = df.drop(list(filter(lambda x: 'distr.' in x, df.columns)), axis=1)
    df = np.round(df.T, 2)
    if sort_by == 'efficiency':
        df = df.sort_values(['Estimated quantiles average length'], axis=1)
    elif sort_by == 'name':
        columns = list(df.columns)
        columns.sort(key=natural_keys)
        df = df[columns]
    elif sort_by in df.index:
        df = df.sort_values([sort_by], axis=1)

    elif callable(sort_by):
        df = sort_by(df)

    if 'syn' in ds_type.lower():
        df = df.T
        true_quantiles_columns = [col for col in df.columns if 'true' in col.lower()]
        estimated_quantiles_columns = [col for col in df.columns if col not in true_quantiles_columns]
        df = df[estimated_quantiles_columns + true_quantiles_columns]
        df = df.T

    df = df.T.rename(columns={col: col.replace('Estimated quantiles ', "") for col in df.T.columns}).T
    df = df.drop([f"coverage in day {i}" for i in range(0, 7)], axis=0, errors='ignore')

    return df


def find_best_gamma_for_all_datasets(real_datasets, syn_datasets, seeds=1, display_errors=False):
    datasets = [(d, 'real') for d in real_datasets] + [(d, 'syn') for d in syn_datasets]
    best_gammas = {}
    for dataset_name, ds_type in datasets:
        best_gammas[dataset_name] = {}
        for train_all_q in [0, 1]:
            best_gammas[dataset_name][f'train_all_q={train_all_q}'] = {}

            base_hyperparameters = {'train_all_q': train_all_q}
            best_gamma_per_cal_method = find_best_gamma(dataset_name, ds_type, False,
                                                        display_errors=display_errors,
                                                        base_hyperparameters=base_hyperparameters,
                                                        seeds=seeds)
            best_gammas[dataset_name][f'train_all_q={train_all_q}'] = best_gamma_per_cal_method
    return best_gammas


def get_best_hp(is_calibrated, dataset_name, is_real, base_hyperparameters=None, calibration_method=None,cal_split=0):
    method_hyperparams = {}
    if calibration_method is not None:
        method_hyperparams['calibration_method'] = [calibration_method]
    if base_hyperparameters is not None:
        for key in base_hyperparameters:
            method_hyperparams[key] = [base_hyperparameters[key]]
    best_hp = get_best_hyperparams(Hyperparameters(
        is_calibrated=is_calibrated,
        train_all_q=method_hyperparams['train_all_q'][0],
        calibration_method=calibration_method,
        cal_split=cal_split,
        ),
        dataset_name, is_real=is_real)
    method_hyperparams['lr'] = [best_hp['lr']]
    method_hyperparams['lstm_hd'] = [best_hp['lstm_hidden_size']]
    method_hyperparams['lstm_in_hd'] = [best_hp['lstm_in_layers']]
    method_hyperparams['lstm_out_hd'] = [best_hp['lstm_out_layers']]
    if base_hyperparameters is not None:
        for key in base_hyperparameters:
            method_hyperparams[key] = [base_hyperparameters[key]]
    return method_hyperparams


def find_best_gamma(dataset_name, ds_type, is_offline, display_errors=False, base_hyperparameters=None, seeds=1):
    calibration_methods = copy.copy(CALIBRATION_METHODS)
    calibration_methods.remove('ACI+CQR')
    method_hyperparams = get_best_hp(True, dataset_name, is_real=ds_type.lower() == 'real',
                                     base_hyperparameters=base_hyperparameters,
                                     calibration_method='ACI+CQR',
                                     cal_split=1)
    method_hyperparams['gamma'] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.35]

    metric_1_name = 'average miscoverage streak length'
    metric_2_name = 'average length'

    ACI_CQR_df = get_results_df(dataset_name, ds_type, is_offline, True,
                                params_to_show=['gamma'],
                                nTop=1000,
                                hyperparams='all',
                                sort_by='none',
                                seeds=seeds, show_full_params=True,
                                display_errors=display_errors,
                                method_hyperparams=method_hyperparams)
    method_to_gamma = {}
    if len(ACI_CQR_df) == 0:
        return method_to_gamma
    base_cal_methods_with_valid_msl = ACI_CQR_df.T[ACI_CQR_df.loc[metric_1_name] <= 1.35].index
    if len(base_cal_methods_with_valid_msl) == 0:
        base_cal_methods_with_valid_msl = ACI_CQR_df.T[abs(ACI_CQR_df.loc[metric_1_name] -
                                                           ACI_CQR_df.loc[
                                                               metric_1_name].min().item()
                                                           ) <= 0.1].index
    base_cal_best_col_name = min(base_cal_methods_with_valid_msl,
                                 key=lambda method: ACI_CQR_df.loc[metric_2_name][method].item())
    get_gamma_from_col_name = lambda col_name: float(
        list(filter(lambda param: 'gamma=' in param, col_name.split("_")))[0].replace("gamma=", ""))
    base_cal_best_gamma = get_gamma_from_col_name(base_cal_best_col_name)
    base_cal_best_col = ACI_CQR_df[base_cal_best_col_name]
    base_cal_best_msl = base_cal_best_col[metric_1_name]
    method_to_gamma['ACI+CQR'] = base_cal_best_gamma

    for calibration_method in calibration_methods:
        method_hyperparams = get_best_hp(True, dataset_name, is_real=ds_type.lower() == 'real',
                                         base_hyperparameters=base_hyperparameters,
                                         calibration_method=calibration_method,
                                         cal_split=0)
        method_hyperparams['gamma'] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.35]
        method_df = get_results_df(dataset_name, ds_type, is_offline, True,
                                   params_to_show=['gamma'],
                                   nTop=1000,
                                   hyperparams='all',
                                   sort_by='none',
                                   seeds=seeds, show_full_params=True,
                                   display_errors=display_errors,
                                   method_hyperparams=method_hyperparams,
                                   )
        if len(method_df) == 0:
            print(f"didn't find results for cal method: {calibration_method}")
            continue
        # trying to find a method with better msl and better length
        found_method_with_better_msl = False
        msl_deltas = [0., 0.03]
        if base_cal_best_msl > 1.15:
            msl_deltas += [0.04]
        if base_cal_best_msl > 1.25:
            msl_deltas += [0.8]
        msl_deltas = list(reversed(msl_deltas))
        # print("msl_deltas:", msl_deltas)
        for msl_delta in msl_deltas:
            methods_with_better_msl = list(method_df.T[method_df.loc[metric_1_name] <
                                                       base_cal_best_msl - msl_delta].T.columns)
            if len(methods_with_better_msl) > 0:
                best_method_name = min(methods_with_better_msl,
                                       key=lambda method: method_df.loc[metric_2_name][method].item())
                best_method = method_df[best_method_name]
                if best_method.loc[metric_2_name].item() < base_cal_best_col[metric_2_name].item():
                    method_to_gamma[calibration_method] = get_gamma_from_col_name(best_method_name)
                    found_method_with_better_msl = True
                    break
        if found_method_with_better_msl:
            continue

        msl_deltas = [0.005, 0.01, 0.02, 0.03]
        found_method_with_better_msl = False
        for msl_delta in msl_deltas:
            methods_with_valid_msl = list(method_df.T[method_df.loc[metric_1_name] <
                                                      base_cal_best_msl + msl_delta].T.columns)
            if len(methods_with_valid_msl) == 0:
                continue
            best_method_name = min(methods_with_valid_msl,
                                   key=lambda method: method_df.loc[metric_2_name][method].item())
            best_method_col = method_df[best_method_name]
            if best_method_col[metric_2_name] < base_cal_best_col[metric_2_name]:
                best_gamma = get_gamma_from_col_name(best_method_name)
                method_to_gamma[calibration_method] = best_gamma
                found_method_with_better_msl = True
                break
        if found_method_with_better_msl:
            continue

        methods_with_better_length = list(method_df.T[method_df.loc[metric_2_name] <
                                                      base_cal_best_msl].T.columns)
        if len(methods_with_better_length) > 0:
            best_method_name = min(methods_with_better_length,
                                   key=lambda method: method_df.loc[metric_2_name][method].item())
        else:
            methods_with_valid_msl = list(method_df.T[abs(method_df.loc[metric_1_name] -
                                                          method_df.loc[
                                                              metric_1_name].min()) <= 0.1].T.columns)
            best_method_name = min(methods_with_valid_msl,
                                   key=lambda method: method_df.loc[metric_2_name][method].item())
        best_gamma = get_gamma_from_col_name(best_method_name)
        method_to_gamma[calibration_method] = best_gamma

    return method_to_gamma


def syn_data_plot(seeds=20, desired_coverage_level=90,
                  full_dataset_name='miscoverage_streak_x_dim_5_z_dim_3_len_5',
                  path='syn_data_plots',
                  calibration_methods=[],
                  display_errors=False,
                  train_all_qs=[False, True],
                  **kwargs):
    metrics = [
        '(coverage streak) coverage',
        '(coverage streak) average length',
        '(miscoverage streak) coverage',
        '(miscoverage streak) average length',
        #  'miscoverage streak length', 'average length',
    ]
    base_hyperparameters = Hyperparameters(**kwargs)

    tmp_path = '.'
    for metric in metrics:
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            os.remove(f"{tmp_path}/{metric}.png")
        metric_display_name = metric \
            .replace("average miscoverage streak length", "conditional coverage metric") \
            .replace("(miscoverage streak)", "Group 1") \
            .replace("(coverage streak)", "Group 0")
        total_df = pd.DataFrame()
        for calibration_method in calibration_methods:
            for train_all_q in train_all_qs:
                try:
                    hyperparameters = base_hyperparameters
                    if calibration_method == 'not calibrated':
                        hyperparameters.is_calibrated = False
                    else:
                        hyperparameters.is_calibrated = True

                    hyperparameters.calibration_method = calibration_method
                    hyperparameters.train_all_q = train_all_q

                    data_params = get_best_hyperparams(hyperparameters, full_dataset_name, False)
                    hyperparameters.lstm_hd = data_params['lstm_hidden_size']
                    hyperparameters.lstm_in_hd = data_params['lstm_in_layers']
                    hyperparameters.lstm_out_hd = data_params['lstm_out_layers']
                    hyperparameters.lr = data_params['lr']
                    hyperparameters.gamma = get_best_gamma(hyperparameters, full_dataset_name, False)
                    base_path = get_base_path(False, full_dataset_name, hyperparameters)
                    folder_path = f"{base_path}/{hyperparameters.to_folder_name()}"
                    df = read_method_results(folder_path, seeds=seeds, apply_mean=False,
                                             base_path=base_path, hyperparameters=hyperparameters,
                                             display_errors=display_errors)
                    # cov = df['Estimated quantiles coverage']
                    # print("calibration_method: ", calibration_method, "coverage: ", np.round(cov.mean(),3))
                    df = df.rename(
                        columns={col: col.replace('Estimated quantiles ', "").replace("days ", "")
                            .replace("average miscoverage streak length", "miscoverage streak length")
                            .replace("(miscoverage streak)", "Group 1") \
                            .replace("(coverage streak)", "Group 0")

                                 for col in df.columns}).T
                    df = df.drop([f"coverage in day {i}" for i in range(0, 7)], axis=0, errors='ignore')
                    df = df.T[metric_display_name].to_frame()
                    df = pd.DataFrame(data={'seed': range(len(df)), metric_display_name: df[metric_display_name]})

                    df['Train over all quantile levels'] = train_all_q
                    df['Calibration method'] = calibration_method \
                        .replace("RCI_Y", "Proposed") \
                        .replace("RCI_Stretched_Exp_e_Y", "Our Method") \
                        .replace("RCI+CQR with cal", "RCI with cal") \
                        .replace("ACI+CQR", "ACI")
                    total_df = total_df.append(df)
                except Exception as e:
                    if display_errors:
                        print(e)

        if len(total_df) > 0:
            total_df.index = range(len(total_df))
            plt.figure(figsize=(5, 4.8))
            if len(train_all_qs) == 1:
                graph = sns.boxplot(x='Calibration method', y=metric_display_name,
                                    data=total_df)
                graph.axes.set_xlabel("")
                axes = [graph.axes]
            else:
                graph = sns.boxplot(x='Calibration method', y=metric_display_name,
                                    hue='Train over all quantile levels',
                                    data=total_df, linewidth=2.5)
                axes = graph.axes[0]
            if 'coverage' in metric and 'length' not in metric:
                for ax in axes:
                    ax.axhline(desired_coverage_level, ls='--')
                    ax.set_ylim(86, 94)
                    ticks = list(filter(lambda x: 0 <= x <= 100, ax.get_yticks()))
                    ticks = list(filter(lambda x: x < 90, ticks)) + [90] + list(filter(lambda x: x > 90, ticks))
                    ax.set_yticks(ticks)

            if 'miscoverage streak length' in metric:
                for ax in axes:
                    ax.axhline(100 / desired_coverage_level, ls='--')

            plt.savefig(f"{tmp_path}/{metric}.png", dpi=300, bbox_inches='tight')
            plt.show()

    im = Image.open(f"{tmp_path}/{metrics[0]}.png")
    im_width = im.width
    im_height = im.height

    for metric in metrics:
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            im = Image.open(f"{tmp_path}/{metric}.png")
            im_width = max(im.width, im_width)
            im_height = max(im.height, im_height)

    new_im = Image.new('RGB', (int(im_width * 2), int(im_height * 2)),
                       color=(255, 255, 255, 0))

    curr_width = 0
    curr_height = 0
    for metric in metrics:
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            im = Image.open(f"{tmp_path}/{metric}.png")
            new_im.paste(im, (curr_width, curr_height))

            if (curr_height // im_height) % 2 == 1:
                curr_width += im_width
                curr_height = 0
            else:
                curr_height += im_height
    for metric in metrics:
        os.remove(f"{tmp_path}/{metric}.png")
    create_folder_if_it_doesnt_exist(path)
    new_im.save(f"{path}/{full_dataset_name.replace('_', ' ')}.png")


def real_data_plot(seeds=20, desired_coverage_level=90,
                   dataset_names=[],
                   path='real_data_plots',
                   calibration_methods=[],
                   display_errors=False,
                   train_all_qs=[False, True],
                   calibration_method_to_scale_intervals=None,
                   save_name='real dataset plot',
                   **kwargs):
    metrics = ['average miscoverage streak length', 'coverage', 'average length',
               'avg. Δ-coverage',
               # 'median length',
               # 'corr',
               # "HSIC",
               ]
    base_hyperparameters = Hyperparameters(**kwargs)

    tmp_path = path
    create_folder_if_it_doesnt_exist(tmp_path)

    for metric in metrics:
        total_df = pd.DataFrame()
        print("metric: ", metric)
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            os.remove(f"{tmp_path}/{metric}.png")
        metric_display_name = metric \
            .replace("average miscoverage streak length", "MSL") \
            .replace("(miscoverage streak)", "group 1") \
            .replace("(coverage streak)", "group 0") \
            .replace("average length", "average length (scaled)") \
            .replace("avg. Δ-coverage", "ΔCoverage") \
            .replace("HSIC", "log HSIC")

        for full_dataset_name in dataset_names:
            dataset_display_name = full_dataset_name.replace("_", " ").replace("tetuan ", "")
            for calibration_method in calibration_methods:
                for train_all_q in train_all_qs:
                    try:
                        hyperparameters = base_hyperparameters
                        if calibration_method == 'not calibrated':
                            hyperparameters.is_calibrated = False
                        else:
                            hyperparameters.is_calibrated = True

                        hyperparameters.calibration_method = calibration_method
                        hyperparameters.train_all_q = train_all_q
                        data_params = get_best_hyperparams(hyperparameters, full_dataset_name, True)
                        hyperparameters.lstm_hd = data_params['lstm_hidden_size']
                        hyperparameters.lstm_in_hd = data_params['lstm_in_layers']
                        hyperparameters.lstm_out_hd = data_params['lstm_out_layers']
                        hyperparameters.lr = data_params['lr']
                        hyperparameters.gamma = get_best_gamma(hyperparameters, full_dataset_name, True)

                        base_path = get_base_path(True, full_dataset_name, hyperparameters)
                        folder_path = f"{base_path}/{hyperparameters.to_folder_name()}"
                        df = read_method_results(folder_path, seeds=seeds, apply_mean=False,
                                                 base_path=base_path, hyperparameters=hyperparameters,
                                                 display_errors=display_errors)
                        df = df.rename({col: col
                                       .replace("Estimated quantiles ", "")
                                       .replace("days ", "")
                                        for col in df.columns}, axis=1)
                        df = df.drop([f"coverage in day {i}" for i in range(0, 7)], axis=0, errors='ignore')
                        df = df[metric].to_frame()
                        if metric == "HSIC":
                            df[metric] = np.log(df[metric])
                        df = pd.DataFrame(data={'seed': range(len(df)), metric_display_name: df[metric]})

                        df['Train over all quantile levels'] = train_all_q
                        if 'ACI+CQR' in calibration_methods:
                            df['Calibration method'] = calibration_method \
                                .replace("RCI_Stretched_Exp_e_Y", "RCI") \
                                .replace("RCI+CQR with cal", "RCI with cal") \
                                .replace("ACI+CQR", "ACI-Online")
                        else:
                            df['Calibration method'] = calibration_method \
                                .replace("RCI_Stretched_Exp_e_Y", r"RCI $\varphi^{{exp.}}_e$") \
                                .replace("RCI_Stretched_Exp_5_Y", r"RCI $\varphi^{{exp.}}_5$") \
                                .replace("RCI_Y", r"RCI $\varphi^{{linear}}$")\
                                .replace("RCI_Stretched_Y",  r"RCI $\varphi^{{poly.}}$")

                        df['Dataset'] = dataset_display_name
                        total_df = total_df.append(df)
                    except Exception as e:
                        print("got an error: ", e)
                        pass

            if len(total_df) == 0:
                continue

            average_length_metric_display_name = 'average length (scaled)'
            median_length_metric_display_name = 'median length'
            if metric_display_name == average_length_metric_display_name or metric_display_name == median_length_metric_display_name:
                if calibration_method_to_scale_intervals is None:
                    reset_calibration_method_to_scale_intervals = True
                    calibration_method_to_scale_intervals = total_df['Calibration method'].iloc[
                        total_df[metric_display_name].argmin()]
                    # print("calibration_method_to_scale_intervals: ",calibration_method_to_scale_intervals)
                else:
                    reset_calibration_method_to_scale_intervals = False

                if calibration_method_to_scale_intervals in list(total_df['Calibration method']):
                    total_df = total_df.reset_index()
                    total_df = total_df.drop(['index'], axis=1)
                    curr_data_idx = total_df['Dataset'] == dataset_display_name
                    method_to_scale_idx = (total_df[
                                               'Calibration method'] == calibration_method_to_scale_intervals) & curr_data_idx
                    mean_interval_len = total_df[method_to_scale_idx][metric_display_name].mean()
                    curr_data_idx = total_df[curr_data_idx].index
                    tmp_df = total_df.loc[curr_data_idx]
                    tmp_df[metric_display_name] /= mean_interval_len
                    total_df.loc[curr_data_idx] = tmp_df

                if reset_calibration_method_to_scale_intervals:
                    calibration_method_to_scale_intervals = None

        if len(total_df) > 0:
            total_df.index = range(len(total_df))
            plt.figure(figsize=(8, 4.5))
            if 'ACI+CQR' in calibration_methods:
                palette=['tab:blue', 'tab:green', 'tab:orange']
            else:
                palette = None
            graph = sns.boxplot(x='Dataset', y=metric_display_name, hue='Calibration method',
                                data=total_df, linewidth=2.5,
                                palette=palette,
                                width=0.7)
            axes = [graph.axes]

            plt.legend(loc="upper left", edgecolor = "black")
            for ax in axes:
                if metric != metrics[0]:
                    ax.get_legend().remove()
                else:
                    # ax.set_ylim(1.02, 1.6)
                    legend = ax.get_legend()
                    legend.get_frame().set_alpha(None)
                    legend.get_frame().set_facecolor((0, 0, 0, 0.1))
            if 'coverage' in metric and 'length' not in metric and 'Δ' not in metric:
                for ax in axes:
                    ax.axhline(desired_coverage_level, ls='--')
                    ticks = list(filter(lambda x: 0 <= x <= 100, ax.get_yticks()))
                    ticks = list(filter(lambda x: x < 90, ticks)) + [90] + list(filter(lambda x: x > 90, ticks))
                    ax.set_yticks(ticks)

            if 'miscoverage streak length' in metric:
                for ax in axes:
                    ax.axhline(100 / desired_coverage_level, ls='--')
            plt.savefig(f"{tmp_path}/{metric}.png", dpi=300,  bbox_inches='tight')
            plt.show()

    im = Image.open(f"{tmp_path}/{metrics[0]}.png")
    im_width = im.width
    im_height = im.height

    for metric in metrics:
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            im = Image.open(f"{tmp_path}/{metric}.png")
            im_width = max(im.width, im_width)
            im_height = max(im.height, im_height)

    new_im = Image.new('RGB', (int(im_width * 2), int(im_height * 2)),
                       color=(255, 255, 255, 0))

    curr_width = 0
    curr_height = 0
    for metric in metrics:
        if os.path.isfile(f"{tmp_path}/{metric}.png"):
            im = Image.open(f"{tmp_path}/{metric}.png")
            new_im.paste(im, (curr_width, curr_height))

            if (curr_height // im_height) % 2 == 1:
                curr_width += im_width
                curr_height = 0
            else:
                curr_height += im_height
    # for metric in metrics:
    #     os.remove(f"{tmp_path}/{metric}.png")
    create_folder_if_it_doesnt_exist(path)
    new_im.save(f"{path}/{save_name}.png")
