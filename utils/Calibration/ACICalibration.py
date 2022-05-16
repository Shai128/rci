from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch
from matplotlib import pyplot as plt
import helper
from utils.Calibration.TSCalibration import TSCalibration
from utils.TSQR import TSQR
from scipy.stats.mstats import mquantiles
from utils.TSModel import TSModel, PredictionIntervals


def get_is_in_interval(y, interval: PredictionIntervals):
    interval = interval.intervals
    return ((y <= interval.squeeze()[1]) & (y >= interval.squeeze()[0])).float().item()


class ACICalibration(TSCalibration, ABC):
    def __init__(self, desired_coverage_level, gamma, calibration_size, calibration_starting_update_index, args,
                 **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, **kwargs)
        self.calibration_starting_update_index = calibration_starting_update_index
        self.alpha = self.alpha_t = 1 - desired_coverage_level
        self.gamma = gamma
        self.calibration_size = calibration_size
        self.args = args

    @staticmethod
    def get_updated_alpha(y_t, calibrated_interval, alpha, gamma, alpha_t):
        y_t = y_t[:, -1, :].squeeze()
        with torch.no_grad():
            not_in_interval = 1 - get_is_in_interval(y_t, calibrated_interval)
            alpha_t = alpha_t + gamma * (alpha - not_in_interval)
        return alpha_t

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        self.alpha_t = ACICalibration.get_updated_alpha(y_t, calibrated_interval_t, self.alpha, self.gamma,
                                                        self.alpha_t)

    def plot_parameter_vs_time_aux(self, parameter: Union[list, torch.Tensor, np.ndarray], starting_time,
                                   ending_time, parameter_name, save_dir=None):
        if type(parameter) == list:
            parameter = torch.Tensor(parameter)
        elif type(parameter) == np.ndarray:
            parameter = torch.from_numpy(parameter)
        elif torch.is_tensor(parameter):
            parameter = parameter.detach().cpu()
        updating_time = self.calibration_starting_update_index
        if self.args.cal_split:
            parameter = torch.cat([torch.zeros(self.calibration_size), parameter], dim=0)
        try:
            plt.plot(np.arange(starting_time, ending_time),
                     parameter[starting_time - updating_time:ending_time - updating_time])
        except Exception as e:
            print("can't plot calibration parameter because", e)
            print("parameter.shape: ", parameter.shape)
            return
        plt.xlabel("Time")
        plt.ylabel(parameter_name)
        helper.create_folder_if_it_doesnt_exist(save_dir)
        save_path = f"{save_dir}/{self.calibration_method}_{parameter_name}_T=[{starting_time},{ending_time}].png"
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.show()

    @abstractmethod
    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        pass


class ACIOnlineWithCQR(ACICalibration):

    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSModel = None, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.Q = None
        self.last_uncalibrated_intervals = []
        self.last_ys = []
        self.calibration_size = calibration_size
        self.calibration_method = 'ACI+CQR'
        self.previous_Qs = []
        self.alpha_t = 1 - desired_coverage_level
        self.model = model

    # should be called after each time stamp
    def fit(self, x_cal=None, y_cal=None, **kwargs):
        if x_cal is not None and y_cal is not None:
            intervals = self.model._TSModel__construct_interval(x_cal, y_cal[:, :-1], 1 - self.alpha, False,
                                                                true_y=None).intervals
            y_cal = y_cal[:, -1]
        else:
            if len(self.last_ys) == 0:
                self.Q = 0
                return
            y_cal = torch.stack(self.last_ys[-self.calibration_size:])
            intervals = torch.stack(self.last_uncalibrated_intervals[-self.calibration_size:])

        y_lower = intervals[:, 0]
        y_upper = intervals[:, 1]
        q = 1 - self.alpha_t + (1 / y_lower.shape[0])
        q = min(1, max(0, q))
        self.Q = torch.quantile(torch.max(y_lower - y_cal, y_cal - y_upper), q=q).item()
        self.previous_Qs += [self.Q]

    def calibrate(self, x, y, predicted_interval: PredictionIntervals, **kwargs):
        with torch.no_grad():
            if self.Q is None:
                return predicted_interval
            predicted_interval = predicted_interval.intervals
            if len(predicted_interval.shape) == 1:
                predicted_interval = predicted_interval.unsqueeze(0)
            calibrated_interval = predicted_interval.clone()
            calibrated_interval[:, 0] -= self.Q
            calibrated_interval[:, 1] += self.Q
        calibrated_interval = PredictionIntervals(calibrated_interval)
        return calibrated_interval

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        self.alpha_t = ACICalibration.get_updated_alpha(y_t, calibrated_interval_t, self.alpha, self.gamma,
                                                        self.alpha_t)
        y_t = y_t[:, -1].squeeze()
        self.last_ys += [y_t.detach()]
        self.last_uncalibrated_intervals += [uncalibrated_interval_t.intervals.detach().squeeze()]
        self.last_ys = self.last_ys[-self.calibration_size:]
        self.last_uncalibrated_intervals = self.last_uncalibrated_intervals[-self.calibration_size:]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        if self.model is not None:
            inverse_transformed_Q = self.model.scaler.unscale_y(torch.Tensor(self.previous_Qs).unsqueeze(-1))
            ACICalibration.plot_parameter_vs_time_aux(self, inverse_transformed_Q, starting_time, ending_time,
                                                      'Q', save_dir=save_dir)


class ACIOnlineWithDCP(ACICalibration):

    def __init__(self, desired_coverage_level, gamma, model: TSQR, calibration_size, extrapolate_quantiles,
                 **kwargs):
        super(ACIOnlineWithDCP, self).__init__(desired_coverage_level, gamma, calibration_size=calibration_size,
                                               **kwargs)
        self.model = model
        self.previous_scores = []
        self.tau_t = 1 - self.desired_coverage_level
        self.calibration_method = ('ext. ' if extrapolate_quantiles else '') + 'ACI+DCP'
        self.extrapolate_quantiles = extrapolate_quantiles
        self.previous_tau_t = []

    def fit(self, x_cal=None, y_cal=None, **kwargs):
        if x_cal is not None and y_cal is not None:
            scores = self.calc_score(x_cal, y_cal)
        else:
            scores = self.previous_scores
        if len(scores) < 500:
            self.tau_t = 1 - self.desired_coverage_level
        else:
            calibration_size = len(scores)
            level_adjusted = (1.0 - self.alpha_t) * (1.0 + 1.0 / float(calibration_size))
            level_adjusted = np.clip(level_adjusted, 0, 1)
            self.tau_t = 1 - 2 * np.ma.getdata(mquantiles(np.array(scores), prob=level_adjusted))[0]
        self.previous_tau_t += [self.tau_t]

    def calibrate(self, x, y, predicted_interval, **kwargs):
        assert x.shape[0] == 1
        calibrated_interval = self.model._TSModel__construct_interval(x, y[:, :-1], 1 - self.tau_t, False,
                                                                      true_y=None,
                                                                      extrapolate_quantiles=self.extrapolate_quantiles) \
            .intervals
        return calibrated_interval

    def calc_score(self, x_t, y_t):
        cdf_function, _ = self.model.get_quantile_function(x_t, y_t[:, :-1],
                                                           extrapolate_quantiles=self.extrapolate_quantiles)
        cdf_values = torch.clip(cdf_function(y_t[:, -1].detach()), 0, 1).detach().cpu()
        scores = np.abs(np.array(cdf_values) - 1 / 2).squeeze()
        return scores

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        score = self.calc_score(x_t, y_t)
        self.previous_scores += [score]
        self.previous_scores = self.previous_scores[-self.calibration_size:]
        if len(self.previous_scores) >= 500:
            super(ACIOnlineWithDCP, self).update(x_t, y_t, uncalibrated_interval_t, calibrated_interval_t)

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        ACICalibration.plot_parameter_vs_time_aux(self, self.previous_tau_t, starting_time, ending_time, 'tau',
                                                  save_dir=save_dir)
