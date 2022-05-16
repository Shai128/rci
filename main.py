from typing import List
import numpy as np
import torch as torch
from datasets.datasets import DataGeneratorFactory, DataSet
from helper import set_seeds
from hyperparameters import get_best_hyperparams_from_args, Hyperparameters
import argparse
import os
import warnings
import ast
import matplotlib
from sys import platform
from results_helper import args_to_txt, save_performance_metrics
from utils.Calibration.RCICalibration import RCIInYScale, \
    RCInTauScale, RCIInStretchedYScale, \
    RCIInExpStretchedYScale, RCIInStretchedTauScale, RCIInMultivariateYScale, RCIInStretchedMultivariateYScale, \
    RCIInStretchedExpMultivariateYScale
from utils.TSCVAE import TSCVAE
from utils.TSQR import TSQR
from utils.Calibration.TSCalibration import TSCalibration
from utils.Calibration.ACICalibration import ACIOnlineWithCQR, ACIOnlineWithDCP

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def parse_args_utils(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device
    args.lstm_out_layers = ast.literal_eval(args.lstm_out_layers)
    args.lstm_in_layers = ast.literal_eval(args.lstm_in_layers)
    if args.backward_size == 0:
        args.lstm_hidden_size = args.lstm_in_layers[-1]
        args.lstm_layers = 0

    assert args.ds_type.lower() in ['real', 'syn', 'synthetic'], "ds_type must be either 'real' or 'synthetic'"
    args.is_real_data = args.ds_type.lower() == 'real'
    args.suppress_plots = False if args.suppress_plots == 0 else 1
    args.train_all_q = args.train_all_q == 1
    args.use_best_hyperparams = args.use_best_hyperparams == 1
    args.cal_split = args.cal_split == 1
    args.apply_calibration = args.apply_calibration == 1
    if args.method_type not in ['baseline', 'mqr']:
        raise Exception("method type must be either 'baseline' or 'mqr'")
    if args.use_best_hyperparams:
        args = get_best_hyperparams_from_args(args)

    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='quantile level')

    parser.add_argument('--dataset_name', type=str, default='window_x_dim_5_z_dim_3_len_500',
                        help='dataset to use')

    parser.add_argument('--method_type', type=str, default='baseline',
                        help="either 'baseline' or 'mqr'")

    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')

    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--z_dim', type=int, default=3,
                        help='dimension of z space (for the CVAE model)')

    parser.add_argument('--backward_size', type=int, default=3,
                        help='')
    parser.add_argument('--wait', type=int, default=100,
                        help='')
    parser.add_argument('--num_ep', type=int, default=1000,
                        help='')
    parser.add_argument('--bs', type=int, default=1,
                        help='')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='')

    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                        help='')

    parser.add_argument('--lstm_in_layers', type=str, default="[32, 64]",
                        help='hidden dimensions')

    parser.add_argument('--lstm_out_layers', type=str, default="[64, 32]",
                        help='hidden dimensions')

    parser.add_argument('--non_linearity', type=str, default="lrelu",
                        help='')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--ds_type', type=str, default="SYN",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')
    parser.add_argument('--calibration_size', type=int, default=300,
                        help="")
    parser.add_argument('--train_all_q', type=int, default=0,
                        help="")
    parser.add_argument('--calibration_starting_update_index', type=int, default=5000,
                        help="")
    parser.add_argument('--use_best_hyperparams', type=int, default=1,
                        help="")
    parser.add_argument('--cal_split', type=int, default=0,
                        help="")
    parser.add_argument('--apply_calibration', type=int, default=1,
                        help="")

    args = parser.parse_args()

    args = parse_args_utils(args)

    return args


def get_model(args, dataset):
    x_dim = dataset.x_train.shape[1]
    y_dim = dataset.y_train.shape[1]
    if args.method_type == 'baseline':
        model = TSQR(x_dim, y_dim,
                     lstm_hidden_size=args.lstm_hidden_size,
                     lstm_layers=args.lstm_layers, lstm_in_layers=args.lstm_in_layers,
                     lstm_out_layers=args.lstm_out_layers, dropout=args.dropout,
                     lr=args.lr, wd=args.wd, device=args.device, non_linearity=args.non_linearity, args=args,
                     tau=args.alpha,
                     dataset=dataset)

    elif args.method_type == 'mqr':
        model = TSCVAE(x_dim, y_dim, args.z_dim, args.device, args.alpha, args, non_linearity = args.non_linearity,
                       dropout = args.dropout, lr = args.lr, wd = args.wd,
                        dataset = dataset)
    else:
        raise Exception("invalid method type")
    model.initialize_scalers(dataset.x_train, dataset.y_train)

    return model


def fit_predict_online(model, dataset, desired_coverage_level, calibrations: List[TSCalibration], args):
    uncalibrated_train_intervals, calibration_train_intervals = model.predict_test_online(
        torch.zeros_like(dataset.x_train)[:args.backward_size + 1].to(dataset.x_train.device),
        torch.zeros_like(dataset.y_train)[:args.backward_size + 1].to(dataset.y_train.device),
        dataset.x_train, dataset.y_train, desired_coverage_level,
        args.backward_size,
        args=args, save_new_model=True, calibrations=calibrations,
        fit_on_train_set=False)

    uncalibrated_test_intervals, calibrated_test_intervals = model.predict_test_online(dataset.x_train, dataset.y_train,
                                                                                       dataset.x_test, dataset.y_test,
                                                                                       desired_coverage_level,
                                                                                       args.backward_size,
                                                                                       calibrations=calibrations,
                                                                                       args=args,
                                                                                       fit_on_train_set=True)
    return uncalibrated_train_intervals, calibration_train_intervals, uncalibrated_test_intervals, \
           calibrated_test_intervals


def get_calibration_schemes(desired_coverage_level, model, args):
    if isinstance(model, TSQR):
        return get_tsqr_calibration_schemes(desired_coverage_level, model, args)
    elif isinstance(model, TSCVAE):
        return get_tsmqr_calibration_schemes(desired_coverage_level, model, args)
    else:
        raise Exception("invalid model type for calibration")


def get_tsmqr_calibration_schemes(desired_coverage_level, model, args):
    calibrations = []
    calibration_params = {'calibration_size': args.calibration_size,
                          'calibration_starting_update_index': args.calibration_starting_update_index,
                          'model': model,
                          'args': args}

    gammas_to_check = [0.1]
    # gammas_to_check = [0.01, 0.05, 0.1, 0.2]
    for gamma in gammas_to_check:
        # calibrations += [RCIInMultivariateYScale(desired_coverage_level, gamma, **calibration_params)]
        # calibrations += [RCIInStretchedMultivariateYScale(desired_coverage_level, gamma, **calibration_params)]
        calibrations += [RCIInStretchedExpMultivariateYScale(desired_coverage_level, gamma, **calibration_params, base=np.e)]
    return calibrations


def get_tsqr_calibration_schemes(desired_coverage_level, model, args):
    calibrations = []
    calibration_params = {'calibration_size': args.calibration_size,
                          'calibration_starting_update_index': args.calibration_starting_update_index,
                          'model': model,
                          'args': args}
    calibrations += [ACIOnlineWithCQR(desired_coverage_level, gamma=0., **calibration_params)]
    if not args.apply_calibration:
        return calibrations

    gammas_to_check = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.35]
    # gammas_to_check = [0.025, 0.05]
    for gamma in gammas_to_check:
        calibrations += [
            ACIOnlineWithCQR(desired_coverage_level, gamma=gamma, **calibration_params),
        ]
        if not args.cal_split:
            calibrations += [
                RCIInYScale(desired_coverage_level, gamma=gamma, **calibration_params),
                RCIInStretchedYScale(desired_coverage_level, gamma=gamma,
                                     **calibration_params),
                RCIInExpStretchedYScale(desired_coverage_level,
                                        base=5,
                                        gamma=gamma, **calibration_params),
                RCIInExpStretchedYScale(desired_coverage_level,
                                        base=np.e,
                                        gamma=gamma, **calibration_params),
            ]

    if args.train_all_q:
        calibrations += [
            ACIOnlineWithDCP(desired_coverage_level, gamma=0.,
                             extrapolate_quantiles=False, **calibration_params),
        ]
        for gamma in gammas_to_check:
            calibrations += [
                ACIOnlineWithDCP(desired_coverage_level, gamma=gamma, **calibration_params,
                                 extrapolate_quantiles=False),
            ]
            if not args.cal_split:
                calibrations += [
                    RCInTauScale(desired_coverage_level, gamma=gamma,
                                 extrapolate_quantiles=True,
                                 **calibration_params),
                    RCInTauScale(desired_coverage_level, gamma=gamma,
                                 extrapolate_quantiles=False,
                                 **calibration_params),
                    RCIInStretchedTauScale(desired_coverage_level, gamma=gamma,
                                           extrapolate_quantiles=True,
                                           **calibration_params),
                ]
    return calibrations


def main():
    args = parse_args()
    set_seeds(args.seed)
    desired_coverage_level = 1 - args.alpha
    is_real_data = args.is_real_data

    device = args.device
    print("device: ", device)

    data_generator = DataGeneratorFactory.get_data_generator(args.dataset_name, is_real_data, args)

    if args.method_type == 'mqr':
        T = min(15000, data_generator.max_data_size)
        args.calibration_starting_update_index = 10000
    else:
        T = min(10000, data_generator.max_data_size)

    dataset = DataSet(data_generator, is_real_data, device, T, args.test_ratio)
    model = get_model(args, dataset)
    calibrations = get_calibration_schemes(desired_coverage_level, model, args)
    uncalibrated_train_intervals, calibration_train_intervals, uncalibrated_test_intervals, calibrated_test_intervals = \
        fit_predict_online(model, dataset, desired_coverage_level, calibrations=calibrations,
                           args=args)

    if not args.suppress_plots:
        model.plot_losses()

    if is_real_data:
        data_params = 'real/'
    else:
        data_params = 'syn/'
    data_params += args.dataset_name
    args_txt = args_to_txt(args, is_calibrated=False)

    base_save_dir = 'results'

    figures_save_dir = f'{base_save_dir}/test/figures/{data_params}/{args_txt}'
    results_info_save_dir = f'{base_save_dir}/test/{data_params}/{args_txt}'
    save_performance_metrics(model, data_generator, dataset.pre_test_data_info, dataset.test_data_info, dataset.x_test,
                             dataset.y_test.squeeze(),
                             uncalibrated_test_intervals, args,
                             figures_save_dir, results_info_save_dir, alpha=args.alpha,
                             initial_time=dataset.starting_test_time,
                             x_train=dataset.x_train,
                             y_train=dataset.y_train,
                             calibration=None)

    if args.calibration_starting_update_index > dataset.x_train.shape[0]:
        return

    for calibration in calibrations:
        args.calibration_method = Hyperparameters.get_cal_method(calibration.calibration_method, args.cal_split)
        args.gamma = calibration.gamma
        args_txt = args_to_txt(args, is_calibrated=True)
        calibrated_preds = calibrated_test_intervals[calibration]
        figures_save_dir = f'{base_save_dir}/test/figures/{data_params}/{args_txt}'
        results_info_save_dir = f'{base_save_dir}/test/{data_params}/{args_txt}'
        save_performance_metrics(model, data_generator, dataset.pre_test_data_info, dataset.test_data_info,
                                 dataset.x_test,
                                 dataset.y_test.squeeze(),
                                 calibrated_preds,
                                 args, figures_save_dir,
                                 results_info_save_dir, alpha=args.alpha, initial_time=dataset.starting_test_time,
                                 x_train=dataset.x_train,
                                 y_train=dataset.y_train, calibration=calibration)


if __name__ == '__main__':
    main()
