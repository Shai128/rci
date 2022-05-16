import abc
from copy import deepcopy
import tqdm
import torch
from abc import ABC, abstractmethod
from typing import List

from utils.Calibration.TSCalibration import TSCalibration
from utils.DataScaler import DataScaler
from utils.Model import Model


class UncertaintyEstimation(abc.ABC):
    def __init__(self):
        pass


class UncertaintyEstimationSet(abc.ABC):
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, test_size: int, scaler: DataScaler):
        self.x_train = x_train
        self.y_train = y_train
        self.test_size = test_size
        self.scaler = scaler

    @abstractmethod
    def add_prediction_intervals(self, new_prediction_intervals: UncertaintyEstimation, idx: list, x: torch.Tensor):
        pass


class PredictionIntervals(UncertaintyEstimation):
    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        self.intervals = intervals


class PredictionIntervalSet(UncertaintyEstimationSet):

    def __init__(self, x_train: torch.Tensor, y_train, test_size, scaler: DataScaler):
        super().__init__(x_train, y_train, test_size, scaler)
        self.device = x_train.device
        self.intervals = torch.zeros(test_size, 2).to(self.device)

    def add_prediction_intervals(self, new_prediction_intervals: PredictionIntervals, idx, x):
        self.intervals[idx] = new_prediction_intervals.intervals

    @property
    def unscaled_intervals(self):
        y_lower = self.scaler.unscale_y(self.intervals[:, 0])
        y_upper = self.scaler.unscale_y(self.intervals[:, 1])
        return torch.cat([y_lower.unsqueeze(-1), y_upper.unsqueeze(-1)], dim=-1)



class TSModel(Model, ABC):

    def __init__(self, dataset, args):
        Model.__init__(self, dataset)
        self.calibration_starting_update_index = args.calibration_starting_update_index
        self.args = args

    @abstractmethod
    def get_uncertainty_quantification_set_class(self):
        pass

    @abstractmethod
    def __construct_interval_aux(self, xi, previous_ys, desired_coverage, true_y=None, **kwargs):
        pass

    def get_x_y_repeated(self, x, x_train, y, y_train, backward_size):
        x_repeated = self.create_x_repeated(x, x_train, backward_size)
        y_repeated = self.create_x_repeated(y, y_train, backward_size)
        return x_repeated, y_repeated

    def create_x_repeated(self, x, x_train, backward_size):
        if x_train is None:
            x_train = torch.zeros_like(x)[:backward_size + 1]
        if backward_size > 0:
            x = torch.cat([x_train[-backward_size:], x], dim=0)

        x_repeated = torch.zeros(x.shape[0] - backward_size, backward_size + 1, x.shape[1]).to(x.device)
        for i in range(backward_size + 1):
            x_repeated[:, i, :] = x[i:i + x.shape[0] - backward_size]

        return x_repeated

    def __construct_interval(self, xi, previous_ys, desired_coverage, is_train, true_y=None, **kwargs):
        if is_train:
            self.train()
            interval = self.__construct_interval_aux(xi, previous_ys, desired_coverage, true_y=true_y, **kwargs)

        else:
            self.eval()
            with torch.no_grad():
                interval = self.__construct_interval_aux(xi, previous_ys, desired_coverage, true_y=true_y, **kwargs)

        return interval

    def __construct_interval_online(self, x_train, y_train, x_test, y_test, coverage_level, backward_size,
                                    args, calibrations=None, unscaled_y_train=None,
                                    unscaled_y_test=None,
                                    fit_on_train_set=True):
        uncertainty_estimation_set_class = self.get_uncertainty_quantification_set_class()
        if calibrations is None:
            calibrations = []
        calibration_sizes = [calibration.calibration_size for calibration in calibrations]
        assert len(calibration_sizes) == 0 or min(calibration_sizes) == max(calibration_sizes)
        if self.args.cal_split and len(calibration_sizes) > 0:
            cal_shift = calibration_sizes[0]
        else:
            cal_shift = 0

        n_test_points = len(y_test)
        q = torch.Tensor([1 - coverage_level]).to(args.device)
        uncalibrated_preds = uncertainty_estimation_set_class(x_train, y_train,
                                                              n_test_points,
                                                              self.scaler)
        calibration_results = {}
        for calibration in calibrations:
            calibration_results[calibration] = uncertainty_estimation_set_class(x_train, y_train,
                                                                                n_test_points,
                                                                                self.scaler)

        all_x = self.create_x_repeated(torch.cat([x_train, x_test], dim=0), torch.zeros_like(x_train), backward_size)
        all_y = self.create_x_repeated(torch.cat([y_train, y_test], dim=0), torch.zeros_like(y_train), backward_size)
        begin_index = max(0 if fit_on_train_set else x_train.shape[0], x_train.shape[0] - cal_shift)
        i = 0
        for curr_idx in tqdm.tqdm(range(begin_index, all_y.shape[0] - cal_shift)):
            xi, yi = all_x[curr_idx].unsqueeze(0), all_y[curr_idx].unsqueeze(0)
            with torch.no_grad():
                self.eval()
                yi_test = all_y[curr_idx + cal_shift].unsqueeze(0)
                previous_ys_test = yi_test[:, :-1]
                xi_test = all_x[curr_idx + cal_shift].unsqueeze(0)
                predicted_interval = self.__construct_interval(xi_test, previous_ys_test,
                                                               desired_coverage=coverage_level,
                                                               is_train=False,
                                                               true_y=None)
                idx = list(range(i, min(i + xi_test.shape[0], n_test_points)))
                uncalibrated_preds.add_prediction_intervals(predicted_interval, idx, xi_test)

                if curr_idx >= self.calibration_starting_update_index:
                    if cal_shift > 0:
                        x_cal = all_x[curr_idx: curr_idx + cal_shift]
                        y_cal = all_y[curr_idx: curr_idx + cal_shift]
                    else:
                        x_cal = y_cal = None
                    for calibration in calibrations:
                        calibration.fit(x_cal=x_cal, y_cal=y_cal)
                        calibrated_preds = calibration.calibrate(xi_test, yi_test, predicted_interval)

                        calibration_results[calibration].add_prediction_intervals(calibrated_preds, idx, xi_test)

                        calibration.update(x_t=xi_test, y_t=yi_test,
                                           uncalibrated_interval_t=predicted_interval,
                                           calibrated_interval_t=calibrated_preds)

            xi.requires_grad = True
            self.train()
            self.loss(xi, yi, all_x[:curr_idx + 1], all_y[:curr_idx + 1], q, True, take_step=True, args=args)
            i += xi.shape[0]

        return uncalibrated_preds, calibration_results

    def predict_test_online(self, x_train, y_train, x_test, y_test, coverage_level, backward_size,
                            args,
                            save_new_model=False, calibrations: List[TSCalibration] = None, fit_on_train_set=True):
        models_copy = deepcopy(self.models)
        optimizers_copy = deepcopy(self.optimizers)
        x_train = self.scaler.scale_x(x_train)
        scaled_y_train = self.scaler.scale_y(y_train)
        x_test = self.scaler.scale_x(x_test)
        scaled_y_test = self.scaler.scale_y(y_test)

        uncalibrated_intervals, calibrated_intervals = self.__construct_interval_online(x_train, scaled_y_train, x_test,
                                                                                        scaled_y_test, coverage_level,
                                                                                        backward_size,
                                                                                        calibrations=calibrations,
                                                                                        unscaled_y_train=y_train,
                                                                                        unscaled_y_test=y_test,
                                                                                        args=args,
                                                                                        fit_on_train_set=fit_on_train_set)

        if not save_new_model:
            self.update_models(models_copy)
            for i in range(len(self.models)):
                self.models[i] = models_copy[i]
            for i in range(len(self.optimizers)):
                self.optimizers[i] = optimizers_copy[i]

        return uncalibrated_intervals, calibrated_intervals

    def update_va_loss(self, va_loss, curr_ep):
        if va_loss < self.best_va_loss:
            self.best_va_loss = va_loss
            self.best_va_ep = curr_ep
            self.best_va_models = deepcopy(self.models)
        else:
            if curr_ep - self.best_va_ep > self.num_wait:
                self.keep_training = False

    @abstractmethod
    def loss_aux(self, all_pre_x, all_pre_y, desired_coverage_level):
        pass

    def loss(self, x, y, all_pre_x, all_pre_y, q_list, batch_q, take_step, args):
        self.train()

        previous_ys = y[:, :-1, :]
        previous_ys.required_grad = True
        y = y[:, -1, :]
        desired_coverage = (1 - q_list).item()

        if all_pre_x is None or all_pre_y is None:
            all_pre_x = x
            all_pre_y = torch.cat([previous_ys, y.reshape(y.shape[0], 1, 1)], dim=1)
        loss = self.loss_aux(all_pre_x, all_pre_y, desired_coverage)

        for optimizer in self.optimizers:
            optimizer.zero_grad()
            if loss.requires_grad and take_step:
                loss.backward()
            optimizer.step()
        return loss
