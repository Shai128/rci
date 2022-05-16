import numpy as np
import torch
from utils.Calibration.ACICalibration import ACICalibration, get_is_in_interval
from utils.Model import Model
from utils.TSCVAE import QuantileRegions, TSCVAE
from utils.TSModel import PredictionIntervals
from utils.TSQR import TSQR


class RCIInYScale(ACICalibration):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: Model = None, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.Q_t = 0
        self.alpha = 1 - desired_coverage_level
        self.calibration_method = 'RCI_Y'
        self.previous_Q_t = []
        self.model = model

    # should be called after each time stamp
    def fit(self, **kwargs):
        pass

    def calibrate(self, x, y, predicted_interval : PredictionIntervals, **kwargs):
        predicted_interval = predicted_interval.intervals
        if len(predicted_interval.shape) == 1:
            predicted_interval = predicted_interval.unsqueeze(0)
        calibrated_interval = predicted_interval.clone()
        calibrated_interval[:, 0] -= self.Q_t
        calibrated_interval[:, 1] += self.Q_t
        calibrated_interval = PredictionIntervals(calibrated_interval)
        return calibrated_interval

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        y_t = y_t[:, -1, :].squeeze()
        with torch.no_grad():
            not_in_interval = 1 - get_is_in_interval(y_t, calibrated_interval_t)
            self.Q_t = self.Q_t + self.gamma * (not_in_interval - self.alpha)
        self.previous_Q_t += [self.Q_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        if self.model is not None:
            inverse_transformed_Q = self.model.scaler.unscale_y(torch.Tensor(self.previous_Q_t).unsqueeze(-1))
            ACICalibration.plot_parameter_vs_time_aux(self, inverse_transformed_Q, starting_time, ending_time,
                                                      'Q', save_dir=save_dir)


class RCIInStretchedYScale(RCIInYScale):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: Model = None, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.Q_t = 0
        self.theta_t = 0
        self.alpha = 1 - desired_coverage_level
        self.calibration_method = 'RCI_Stretched_Y'
        self.previous_Q_t = []
        self.previous_theta_t = []
        self.model = model
        power = lambda x: ((4 * x) ** 3)
        root = lambda x: (np.cbrt(x)) / 4

        def f(x):  # theta->Q
            linear_idx = (-0.1 <= power(x)) & (power(x) <= 0.1)
            return linear_idx * x + (~linear_idx) * power(x)

        def inv_f(x):  # Q->theta
            linear_idx = (-0.1 <= x) & (x <= 0.1)
            return linear_idx * x + (~linear_idx) * root(x)

        self.f = f
        self.inv_f = inv_f

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        y_t = y_t[:, -1, :].squeeze()
        with torch.no_grad():
            not_in_interval = 1 - get_is_in_interval(y_t, calibrated_interval_t)
            self.theta_t = self.theta_t + self.gamma * (not_in_interval - self.alpha)

        self.Q_t = self.f(np.array([self.theta_t])).item()
        self.previous_Q_t += [self.Q_t]
        self.previous_theta_t += [self.theta_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        if self.model is not None:
            inverse_transformed_Q = self.model.scaler.unscale_y(torch.Tensor(self.previous_Q_t).unsqueeze(-1))
            ACICalibration.plot_parameter_vs_time_aux(self, inverse_transformed_Q, starting_time, ending_time,
                                                      'Q', save_dir=save_dir)
            ACICalibration.plot_parameter_vs_time_aux(self, self.previous_theta_t, starting_time, ending_time,
                                                      '$theta$', save_dir=save_dir)


class RCIInExpStretchedYScale(RCIInStretchedYScale):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: Model = None, base=np.e, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.Q_t = 0
        self.theta_t = 0
        self.alpha = 1 - desired_coverage_level
        base_name = 'e' if base == np.e else base
        self.calibration_method = f'RCI_Stretched_Exp_{base_name}_Y'
        self.previous_Q_t = []
        self.model = model
        exp = lambda x: ((np.nan_to_num(np.power(base, x) - 1)) * (x > 0) - (np.nan_to_num(np.power(base, -x) - 1)) * (
                x < 0))

        def f(x):
            linear_idx = (-0.1 <= exp(x)) & (exp(x) <= 0.1)
            return linear_idx * x + (~linear_idx) * exp(x)

        self.f = f


class RCInTauScale(ACICalibration):

    def __init__(self, desired_coverage_level, gamma, model: TSQR, calibration_size, extrapolate_quantiles,
                 **kwargs):
        super().__init__(desired_coverage_level, gamma, calibration_size=calibration_size, **kwargs)
        self.model = model
        self.alpha = self.tau_t = 1 - desired_coverage_level
        self.previous_tau_t = []
        self.calibration_method = ('ext._' if extrapolate_quantiles else '') + f'RCI_Tau'
        self.extrapolate_quantiles = extrapolate_quantiles

    def fit(self, **kwargs):
        pass

    def calibrate(self, x, y, predicted_interval, **kwargs):
        calibrated_interval = self.model._TSModel__construct_interval(x, y[:, :-1], 1 - self.tau_t, False,
                                                                      true_y=None,
                                                                      extrapolate_quantiles=self.extrapolate_quantiles)
        return calibrated_interval

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        self.tau_t = ACICalibration.get_updated_alpha(y_t, calibrated_interval_t, self.alpha, self.gamma,
                                                      self.tau_t)
        self.previous_tau_t += [self.tau_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        ACICalibration.plot_parameter_vs_time_aux(self, self.previous_tau_t, starting_time, ending_time,
                                                  r'tau', save_dir=save_dir)


class RCIInStretchedTauScale(RCInTauScale):

    def __init__(self, desired_coverage_level, gamma, model: TSQR, calibration_size, extrapolate_quantiles,
                 **kwargs):
        super().__init__(desired_coverage_level, gamma, calibration_size=calibration_size, model=model,
                         extrapolate_quantiles=extrapolate_quantiles, **kwargs)
        alpha = self.theta_t = 1 - desired_coverage_level
        self.previous_tau_t = []
        self.calibration_method = ('ext._' if extrapolate_quantiles else '') + 'RCI_Stretched_Tau'

        bounded_by_one = lambda x: (np.tanh(3 * x) + 1) / 2
        intermediate = lambda x: (x + 0.5) ** 2 - 0.25 + alpha
        bounded_by_zero = lambda x: (np.tanh(x * 20 + 0.4) + 1) / 8
        thetas = np.linspace(0, 2, 1000)
        intersect_idx = np.argmin((bounded_by_one(thetas) - intermediate(thetas)) ** 2)
        c1 = (thetas[intersect_idx], intermediate(thetas[intersect_idx]))

        thetas = np.linspace(-0.08, 0, 1000)
        intersect_idx = np.argmin((bounded_by_zero(thetas) - intermediate(thetas)) ** 2)
        c2 = (thetas[intersect_idx], intermediate(thetas[intersect_idx]))

        def f(x):  # theta-> tau
            res = (x > c1[0]) * bounded_by_one(x) + ((c2[0] <= x) & (x <= c1[0])) * intermediate(x) + (
                    c2[0] > x) * bounded_by_zero(x)
            res = np.maximum(res, 1e-15)
            return res

        self.f = f

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, **kwargs):
        self.theta_t = ACICalibration.get_updated_alpha(y_t, calibrated_interval_t, self.alpha, self.gamma,
                                                        self.theta_t)
        self.tau_t = self.f(np.array([self.theta_t])).item()
        self.previous_tau_t += [self.tau_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        ACICalibration.plot_parameter_vs_time_aux(self, self.previous_tau_t, starting_time, ending_time,
                                                  'tau', save_dir=save_dir)


class RCIInMultivariateYScale(ACICalibration):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: Model = None, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.r_t = 1
        self.alpha = 1 - desired_coverage_level
        self.calibration_method = 'RCI_r'
        self.previous_r_t = []
        self.model = model

    # should be called after each time stamp
    def fit(self, **kwargs):
        pass

    def calibrate(self, x, y, predicted_interval, **kwargs):
        calibrated_interval = self.model._TSModel__construct_interval(x, y[:, :-1], 1 - self.alpha, False,
                                                                      true_y=None,
                                                                      y_stretching=self.r_t
                                                                      )
        return calibrated_interval

    def update(self, x_t, y_t, uncalibrated_interval_t: QuantileRegions, calibrated_interval_t: QuantileRegions,
               **kwargs):
        y_t = y_t[:, -1, :]
        with torch.no_grad():
            not_in_interval = 1 - calibrated_interval_t.is_in_region(y_t).float().item()
            self.r_t = self.r_t + self.gamma * (not_in_interval - self.alpha)
        self.previous_r_t += [self.r_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        pass


class RCIInStretchedMultivariateYScale(RCIInMultivariateYScale):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSCVAE = None, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.theta_t = 0
        self.r_t = 1
        self.alpha = 1 - desired_coverage_level
        self.calibration_method = 'RCI_stretched_r'
        self.previous_theta_t = []
        self.model = model

        const = 1
        func = lambda x: (2 * x) ** 3 + const
        linear = lambda x: x + const
        thetas = np.linspace(-5, -0.001, 10000)
        intersect_idx = np.argmin((func(thetas) - linear(thetas)) ** 2)
        c1 = thetas[intersect_idx]
        thetas = np.linspace(0.001, 5, 10000)
        intersect_idx = np.argmin((func(thetas) - linear(thetas)) ** 2)
        c2 = thetas[intersect_idx]

        def f(theta):
            return ((theta <= c2) | (theta >= c1)) * func(theta) + ((c2 < theta) & (theta < c1)) * linear(theta)

        self.f = f

    def update(self, x_t, y_t, uncalibrated_interval_t: QuantileRegions, calibrated_interval_t: QuantileRegions,
               **kwargs):
        y_t = y_t[:, -1, :]
        with torch.no_grad():
            not_in_interval = 1 - calibrated_interval_t.is_in_region(y_t).float().item()
            self.theta_t = self.theta_t + self.gamma * (not_in_interval - self.alpha)
        self.r_t = self.f(self.theta_t)
        self.previous_theta_t += [self.theta_t]



class RCIInStretchedExpMultivariateYScale(RCIInStretchedMultivariateYScale):
    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSCVAE = None, base=np.e, **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.base = base
        self.theta_t = 0
        self.r_t = 1
        self.alpha = 1 - desired_coverage_level
        self.previous_theta_t = []
        self.model = model
        base_name = 'e' if base == np.e else base
        self.calibration_method = f'RCI_Stretched_Exp_{base_name}_Y'
        self.model = model

        exp = lambda x: ((np.nan_to_num(np.power(base, x) - 1)) * (x > 0) - (np.nan_to_num(np.power(base, -x) - 1)) * (
                x < 0))

        def f(theta):
            return exp(theta)

        self.f = f
