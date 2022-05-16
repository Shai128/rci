import itertools
import torch
import helper
from utils.BaseModel import BaseModel
from utils.LSTMModel import LSTMModel
from utils.TSModel import TSModel, PredictionIntervals, PredictionIntervalSet


def batch_pinball_loss(quantile_level, quantile, y):
    diff = quantile - y.squeeze()
    mask = (diff.ge(0).float() - quantile_level).detach()

    return (mask * diff).mean(dim=0)


class TSQR(TSModel):
    def plot_losses(self):
        pass

    def get_uncertainty_quantification_set_class(self):
        return PredictionIntervalSet

    def update_models(self, models):
        self.models = models
        self.quantile_estimator, self.x_feature_extractor = self.models
        params = list(itertools.chain(*[list(model.parameters()) for model in self.models]))
        self.optimizers = [torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)]

    def __init__(self, x_dim, y_dim, device, lstm_hidden_size, lstm_layers, lstm_in_layers,
                 lstm_out_layers, tau, args, non_linearity='lrelu', dropout=0.1, lr=1e-3, wd=0.,
                 dataset=None):
        super().__init__(dataset, args)
        self.lr = lr
        self.wd = wd
        tau_dim = 1
        self.x_feature_extractor = LSTMModel(x_dim, y_dim=y_dim, out_dim=lstm_out_layers[-1],
                                             lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers,
                                             lstm_in_layers=lstm_in_layers, lstm_out_layers=lstm_out_layers[:-1],
                                             dropout=dropout, non_linearity=non_linearity, args=args).to(device)

        self.quantile_estimator = BaseModel(lstm_out_layers[-1] + tau_dim, y_dim, [32], dropout=dropout,
                                            batch_norm=False, non_linearity=non_linearity).to(device)

        self.models = [self.quantile_estimator, self.x_feature_extractor]
        params = list(itertools.chain(*[list(model.parameters()) for model in self.models]))
        self.optimizers = [torch.optim.Adam(params, lr=lr, weight_decay=wd)]

        self.backward_size = args.backward_size
        self.train_all_q = args.train_all_q
        self.device = device
        self.tau = tau
        self.lr = lr
        self.wd = wd

    def forward(self, x, previous_ys, quantile_levels):
        if previous_ys.shape[0] == 1 and x.shape[0] > 1:
            previous_ys = previous_ys.repeat(x.shape[0], 1, 1)
        x_and_time_extraction = self.x_feature_extractor(x, previous_ys)
        y_rec = self.quantile_estimator(torch.cat([quantile_levels.unsqueeze(-1), x_and_time_extraction], dim=-1))

        return y_rec

    def pinball_loss(self, x, y, previous_ys, desired_coverage):
        if self.train_all_q:
            quantile_levels = self.get_learned_quantile_levels()
        else:
            alpha = 1 - desired_coverage
            quantile_levels = torch.Tensor([alpha / 2, 1 - alpha / 2]).to(x.device)

        quantiles = self.estimate_quantiles(x, previous_ys, quantile_levels=quantile_levels)
        quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(x.shape[0], 1).flatten(0, 1)
        y_rep = y.squeeze().unsqueeze(1).repeat(1, quantile_levels.shape[0]).flatten(0, 1)
        pinball_loss = batch_pinball_loss(quantile_levels_rep, quantiles.flatten(0, 1), y_rep)
        return pinball_loss

    def get_learned_quantile_levels(self):
        if self.train_all_q:
            return torch.arange(0.02, 0.99, 0.005, device=self.device)
        else:
            return torch.Tensor([self.tau / 2, 1 - self.tau / 2]).to(self.device)

    def estimate_quantiles(self, x, previous_ys, quantile_levels):
        quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(x.shape[0], 1).flatten(0, 1)
        x_rep = x.unsqueeze(1).repeat(1, quantile_levels.shape[0], 1, 1).flatten(0, 1)
        previous_ys_rep = previous_ys.unsqueeze(1).repeat(1, quantile_levels.shape[0], 1, 1).flatten(0, 1)
        unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(x.shape[0], quantile_levels.shape[0]))
        quantiles = unflatten(self.forward(x_rep, previous_ys_rep, quantile_levels_rep)).squeeze(-1)
        return quantiles

    def _TSModel__construct_interval_aux(self, x, previous_ys, desired_coverage, true_y=None,
                                         extrapolate_quantiles=False) -> PredictionIntervals:
        alpha = 1 - desired_coverage
        alpha_rep = torch.ones((x.shape[0], 1), device=x.device) * alpha
        if self.train_all_q:
            _, inverse_cdf = self.get_quantile_function(x, previous_ys, extrapolate_quantiles=extrapolate_quantiles)
            q_low = inverse_cdf(alpha_rep / 2)
            q_high = inverse_cdf(1 - alpha_rep / 2)
            intervals = torch.stack([q_low, q_high]).T
            return PredictionIntervals(intervals.squeeze())
        else:
            intervals = self.estimate_quantiles(x, previous_ys,
                                                torch.Tensor([alpha / 2, 1 - alpha / 2]).to(x.device))
            return PredictionIntervals(intervals)

    def get_quantile_function(self, x, previous_ys, extrapolate_quantiles=False):
        quantile_levels, _ = self.get_learned_quantile_levels().sort()
        quantiles = self.estimate_quantiles(x, previous_ys, quantile_levels)
        quantile_levels = quantile_levels.detach().squeeze()
        quantiles = quantiles.detach()
        quantile_functions = helper.batch_estim_dist(quantiles, quantile_levels, self.dataset.y_scaled_min,
                                                     self.dataset.y_scaled_max,
                                                     smooth_tails=True, tau=0.01,
                                                     extrapolate_quantiles=extrapolate_quantiles)
        return quantile_functions

    def loss_aux(self, all_pre_x, all_pre_y, desired_coverage):
        batch_size = min(all_pre_x.shape[0], 256)
        loss = self.pinball_loss(all_pre_x[-batch_size:], all_pre_y[-batch_size:, -1],
                                 all_pre_y[-batch_size:, :-1], desired_coverage)
        return loss
