import numpy as np
import pandas as pd
import torch
import helper
from datasets.datasets import SynDataGenerator, DataGenerator, WindowSynDataGenerator
from helper import create_folder_if_it_doesnt_exist
from hyperparameters import Hyperparameters
from plot_helper import plot_syn_data_results, plot_mqr_results
from utils.Calibration.RCICalibration import RCInTauScale
from utils.Calibration.ACICalibration import ACIOnlineWithCQR
import traceback
from utils.TSCVAE import QuantileRegionSet
from utils.TSModel import UncertaintyEstimationSet, PredictionIntervalSet


def get_avg_miscoverage_streak_len(y, upper_q, lower_q):
    v = (y <= upper_q) & (y >= lower_q)
    return get_avg_miscoverage_streak_len_aux(v)


def get_avg_miscoverage_streak_len_aux(coverages):
    v = coverages
    streaks_lengths = []
    curr_streak_len = 0
    i = 0
    while i < len(v) and v[i]:
        i += 1

    while i < len(v):
        if not v[i]:
            curr_streak_len += 1
            i += 1
        else:
            streaks_lengths += [curr_streak_len]
            curr_streak_len = 0
            while i < len(v) and v[i]:
                i += 1
    return np.round(np.mean(streaks_lengths), 3)


def corr(cov_identifier, average_len):
    return helper.pearsons_corr(cov_identifier.float(), average_len.float()).item()


def calculate_pearson_corr(y, upper_q, lower_q):
    cov_identifier = ((y <= upper_q) & (y >= lower_q)).float()
    average_len = upper_q - lower_q
    return corr(cov_identifier, average_len)


def hsic(cov_identifier: torch.Tensor, average_len: torch.Tensor):
    return helper.HSIC(cov_identifier.float().unsqueeze(-1), average_len.float().unsqueeze(-1)).item()


def calculate_hsic(y, upper_q, lower_q):
    cov_identifier = ((y <= upper_q) & (y >= lower_q)).float()
    average_len = (upper_q - lower_q)
    return hsic(cov_identifier, average_len)


def calculate_coverage(y, upper_q, lower_q):
    return ((y <= upper_q) & (y >= lower_q)).float().mean().item() * 100


def calculate_average_length(upper_q, lower_q):
    return (upper_q - lower_q).mean().item()


def calculate_median_length(upper_q, lower_q):
    return (upper_q - lower_q).median().item()


def add_results_into_results_dict(results, x, y, upper_q, lower_q, is_true_quantiles, true_upper_q, true_lower_q,
                                  title_prefix, args, estimation_error_prefix=''):
    try:
        results[f'{title_prefix} average miscoverage streak length'] = get_avg_miscoverage_streak_len(y, upper_q,
                                                                                                      lower_q)
        results[f'{title_prefix} coverage'] = calculate_coverage(y, upper_q, lower_q)
        results[f'{title_prefix} average length'] = calculate_average_length(upper_q, lower_q)
        results[f'{title_prefix} median length'] = calculate_median_length(upper_q, lower_q)
        results[f'{title_prefix} corr'] = calculate_pearson_corr(y, upper_q, lower_q)
        results[f'{title_prefix} HSIC'] = calculate_hsic(y, upper_q, lower_q)

        try:
            has_all_days = (
                    torch.unique(x[:, -1]) == torch.Tensor([0., 1., 2., 3., 4., 5., 6.]).to(x.device)).all().item()
        except Exception:
            has_all_days = False

        if args.is_real_data and has_all_days:
            for day in range(0, 7):
                idx = x[:, -1] == day
                results[f'{title_prefix} coverage in day {day}'] = calculate_coverage(y[idx], upper_q[idx],
                                                                                      lower_q[idx])
            coverage_per_day = np.array([results[f'{title_prefix} coverage in day {day}'] for day in range(0, 6)])
            results[f'{title_prefix} days avg. Δ-coverage'] = np.abs((coverage_per_day -
                                                                      results[f'{title_prefix} coverage'])).mean()

        if not is_true_quantiles and true_upper_q is not None and true_lower_q is not None:
            results[f'{estimation_error_prefix}Upper quantile estimation error'] = (
                    (upper_q - true_upper_q) ** 2).mean().item()
            results[f'{estimation_error_prefix}Lower quantile estimation error'] = (
                    (lower_q - true_lower_q) ** 2).mean().item()
    except:
        traceback.print_exc()


def args_to_txt(args, is_calibrated=False):
    hp = Hyperparameters.from_args(args, is_calibrated)
    hp.is_calibrated = is_calibrated
    return hp.to_folder_name()


def quantile_regression_save_performance_metrics(model, data_generator: DataGenerator, training_data_info,
                                                 test_data_info, x_test, y_test,
                                                 prediction_interval: PredictionIntervalSet, args, figures_save_dir,
                                                 results_info_save_dir, alpha, initial_time, x_train=None,
                                                 y_train=None, train_losses=None, calibration=None):
    intervals = prediction_interval.unscaled_intervals
    lower_q_pred, upper_q_pred = intervals[:, 0], intervals[:, 1]
    results = {}
    if isinstance(data_generator, SynDataGenerator):
        n_samples = 2000
        T = len(y_test)
        _, samples, _ = data_generator.generate_data(T, x=x_test, previous_data_info=training_data_info,
                                                     n_samples=n_samples, device=x_test.device,
                                                     current_process_info=test_data_info, use_constant_seed=False)

        true_upper_q = samples.quantile(dim=0, q=1 - alpha / 2).to(y_test.device)
        true_lower_q = samples.quantile(dim=0, q=alpha / 2).to(y_test.device)
        add_results_into_results_dict(results, x_test, y_test, true_upper_q, true_lower_q, True, true_upper_q,
                                      true_lower_q,
                                      title_prefix='True quantiles', args=args)
    else:
        samples = None
        true_upper_q = true_lower_q = None

    if isinstance(data_generator, WindowSynDataGenerator):
        assert true_lower_q is not None and true_lower_q is not None
        coverage_streak_idx = data_generator.data_type_idx[initial_time:initial_time + y_test.shape[0]] % 2 == 0
        miscoverage_streak_idx = ~coverage_streak_idx
        add_results_into_results_dict(results, x_test[coverage_streak_idx], y_test[coverage_streak_idx],
                                      upper_q_pred[coverage_streak_idx], lower_q_pred[coverage_streak_idx], False,
                                      true_upper_q[coverage_streak_idx], true_lower_q[coverage_streak_idx],
                                      title_prefix='Estimated quantiles (coverage streak)', args=args,
                                      estimation_error_prefix='Coverage streak ')
        add_results_into_results_dict(results, x_test[miscoverage_streak_idx], y_test[miscoverage_streak_idx],
                                      upper_q_pred[miscoverage_streak_idx], lower_q_pred[miscoverage_streak_idx], False,
                                      true_upper_q[miscoverage_streak_idx], true_lower_q[miscoverage_streak_idx],
                                      title_prefix='Estimated quantiles (miscoverage streak)', args=args,
                                      estimation_error_prefix='Miscoverage streak ')

    # Marginal results
    add_results_into_results_dict(results, x_test, y_test, upper_q_pred, lower_q_pred, False, true_upper_q,
                                  true_lower_q,
                                  title_prefix='Estimated quantiles', args=args)

    if calibration is not None:
        if isinstance(calibration, ACIOnlineWithCQR):
            Q = calibration.Q
            Q = Q.cpu().item() if torch.is_tensor(Q) else Q
            results['calibration Q'] = Q
            results['avg. calibration Q'] = np.mean(calibration.previous_Qs)
            results['alpha_t'] = calibration.alpha_t
        if isinstance(calibration, RCInTauScale):
            results["calibration alpha_t"] = calibration.tau_t
            results['avg. calibration alpha_t'] = np.mean(calibration.previous_tau_t)

    plot_syn_data_results(y_test, samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q, initial_time,
                          figures_save_dir, args, calibration=calibration)

    create_folder_if_it_doesnt_exist(results_info_save_dir)
    pd.DataFrame(results, index=[args.seed]).to_csv(f"{results_info_save_dir}/seed={args.seed}.csv")

    return results


def multiple_output_uq_save_performance_metrics(model, data_generator: DataGenerator, training_data_info,
                                                test_data_info, x_test, y_test,
                                                quantile_regions: QuantileRegionSet, args, figures_save_dir,
                                                results_info_save_dir, alpha, initial_time, x_train=None,
                                                y_train=None, train_losses=None, calibration=None):
    results = {}
    coverage = quantile_regions.is_in_region(y_test, is_scaled=False)
    area = quantile_regions.areas
    results['coverage'] = coverage.float().mean().item()
    results['average area'] = area.float().mean().item()
    results['median area'] = area.float().median().item()
    results['corr'] = corr(coverage, area)
    results['hsic'] = hsic(coverage, area)

    plot_mqr_results(y_test, quantile_regions, initial_time, alpha, figures_save_dir, args, calibration=calibration)

    create_folder_if_it_doesnt_exist(results_info_save_dir)
    pd.DataFrame(results, index=[args.seed]).to_csv(f"{results_info_save_dir}/seed={args.seed}.csv")


def save_performance_metrics(model, data_generator: DataGenerator, training_data_info, test_data_info, x_test, y_test,
                             uncertainty_estimation_set: UncertaintyEstimationSet, args, figures_save_dir,
                             results_info_save_dir, alpha, initial_time, x_train=None,
                             y_train=None, train_losses=None, calibration=None):
    if isinstance(uncertainty_estimation_set, PredictionIntervalSet):
        return quantile_regression_save_performance_metrics(model, data_generator, training_data_info,
                                                            test_data_info, x_test, y_test,
                                                            uncertainty_estimation_set, args, figures_save_dir,
                                                            results_info_save_dir, alpha, initial_time, x_train=x_train,
                                                            y_train=y_train, train_losses=train_losses,
                                                            calibration=calibration)
    elif isinstance(uncertainty_estimation_set, QuantileRegionSet):
        multiple_output_uq_save_performance_metrics(model, data_generator, training_data_info,
                                                    test_data_info, x_test, y_test,
                                                    uncertainty_estimation_set, args, figures_save_dir,
                                                    results_info_save_dir, alpha, initial_time, x_train=x_train,
                                                    y_train=y_train, train_losses=train_losses, calibration=calibration)
    else:
        raise Exception("not implemented yet")
