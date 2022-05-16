import itertools
import torch
from scipy.stats import chi2
import helper
from utils.BaseModel import BaseModel
from utils.DataScaler import DataScaler
from utils.LSTMModel import LSTMModel
from utils.TSModel import TSModel, UncertaintyEstimationSet, UncertaintyEstimation
import torch.nn.functional as F
import matplotlib.pyplot as plt

z_dim_to_quantile_region_sample_size = {
    1: 1e3,
    2: 5e3,
    3: 5e3,
    4: 1e4,
}


class QuantileRegions(UncertaintyEstimation):

    def __init__(self, quantile_region_sample, unscaled_y_train, scaler: DataScaler):
        super().__init__()
        self.quantile_region_sample = quantile_region_sample
        self.in_region_distances = helper.get_min_distance(quantile_region_sample, quantile_region_sample,
                                                           ignore_zero_distance=True,
                                                           y_batch_size=10000,
                                                           points_batch_size=10000)
        self.unscaled_y_train = unscaled_y_train
        self.initial_in_region_threshold = torch.quantile(self.in_region_distances, q=0.8).item()
        self.scaler = scaler
        self.area = self.calc_area()

    def is_in_region(self, points: torch.Tensor, in_region_threshold: float = None):
        if in_region_threshold is None:
            in_region_threshold = self.initial_in_region_threshold
        if len(points.shape) == 2:
            points = points.unsqueeze(1)
            squeeze_dim_1 = True
        else:
            squeeze_dim_1 = False
        res = helper.get_min_distance(points, self.quantile_region_sample,
                                      ignore_zero_distance=False,
                                      y_batch_size=50,
                                      points_batch_size=10000) < in_region_threshold
        if squeeze_dim_1:
            res = res.squeeze(1)
        return res

    def calc_area(self):
        border_max = torch.quantile(self.unscaled_y_train, dim=0, q=0.95) * (11 / 10)
        border_min = torch.quantile(self.unscaled_y_train, dim=0, q=0.05) * (9 / 10)
        y_dim = self.unscaled_y_train.shape[-1]
        n_points_to_sample = z_dim_to_quantile_region_sample_size[y_dim]
        stride = (border_max - border_min) / (n_points_to_sample ** (1 / y_dim))
        device = self.quantile_region_sample.device
        unscaled_y_grid = helper.get_grid_from_borders(border_max, border_min, stride, device)
        y_grid_area = (border_max - border_min).prod(dim=-1)
        scaled_y_grid = self.scaler.scale_y(unscaled_y_grid)
        is_in_region = self.is_in_region(scaled_y_grid.unsqueeze(0).repeat(self.quantile_region_sample.shape[0], 1, 1))
        area = is_in_region.float().mean().item() * y_grid_area
        return area

    def get_area(self):
        return self.area


class QuantileRegionSet(UncertaintyEstimationSet):
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, test_size: int, scaler: DataScaler):
        super().__init__(x_train, y_train, test_size, scaler)
        self.quantile_region_samples = []
        self.quantile_regions = []
        self._areas = []
        self.initial_in_region_threshold = []

    @property
    def areas(self) -> torch.Tensor:
        return torch.stack(self._areas)

    def add_prediction_intervals(self, new_prediction_intervals: QuantileRegions, idx: list, x: torch.Tensor):
        self.quantile_region_samples += [new_prediction_intervals.quantile_region_sample]
        self.quantile_regions += [new_prediction_intervals]
        self._areas += [new_prediction_intervals.area]
        self.initial_in_region_threshold += [new_prediction_intervals.initial_in_region_threshold]

    def is_in_region(self, y_test, in_region_threshold=None, is_scaled=True) -> torch.Tensor:
        if not is_scaled:
            y_test = self.scaler.scale_y(y_test)
        if in_region_threshold is None:
            assert len(self.initial_in_region_threshold) == y_test.shape[0]
            in_region_threshold = torch.Tensor(self.initial_in_region_threshold).to(y_test.device)
        quantile_region_samples = torch.stack([qr.quantile_region_sample for qr in self.quantile_regions]).flatten(0, 1)

        # noinspection PyTypeChecker
        return helper.get_min_distance(y_test.unsqueeze(1), quantile_region_samples,
                                       ignore_zero_distance=False,
                                       y_batch_size=50,
                                       points_batch_size=10000).squeeze() < in_region_threshold


class TSCVAE(TSModel):

    def __init__(self, x_dim, y_dim, z_dim, device, tau, args, non_linearity='lrelu', dropout=0.1, lr=1e-3, wd=0.,
                 dataset=None):

        super().__init__(dataset, args)
        if x_dim <= 5:
            feature_extractor_in_dims = [32, 64]
            feature_extractor_out_dims = [64]
            hidden_dims = [32, 64, 64, 32]
        elif x_dim <= 25:
            feature_extractor_in_dims = [32, 64, 128]
            feature_extractor_out_dims = [64]
            hidden_dims = [32, 64, 128, 64, 32]
        elif x_dim <= 60:
            feature_extractor_in_dims = [64, 128]
            feature_extractor_out_dims = [64]
            hidden_dims = [64, 128, 128, 64, 32]
        else:
            feature_extractor_in_dims = [64, 128]
            feature_extractor_out_dims = [64]
            hidden_dims = [64, 128, 256, 128, 64, 32]

        encoded_dim = 8
        hidden_dims = hidden_dims[:-1]

        self.x_feature_extractor = LSTMModel(x_dim, y_dim=y_dim, out_dim=feature_extractor_out_dims[-1],
                                             lstm_hidden_size=64, lstm_layers=1,
                                             lstm_in_layers=feature_extractor_in_dims,
                                             lstm_out_layers=feature_extractor_out_dims[:-1],
                                             dropout=dropout, non_linearity=non_linearity, args=args).to(device)

        self.features_encoder = BaseModel(feature_extractor_out_dims[-1] + y_dim + x_dim, out_dim=encoded_dim,
                                          hidden_dims=hidden_dims,
                                          dropout=dropout, non_linearity=non_linearity).to(device)

        self.features_decoder = BaseModel(x_dim + encoded_dim + feature_extractor_out_dims[-1], out_dim=y_dim,
                                          hidden_dims=hidden_dims,
                                          dropout=dropout, non_linearity=non_linearity).to(device)

        self.mu_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.sigma_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.decoder = torch.nn.Linear(z_dim, encoded_dim).to(device)

        params = []
        self.models = [self.x_feature_extractor, self.features_encoder, self.features_decoder, self.mu_linear,
                       self.sigma_linear, self.decoder]
        for model in self.models:
            params += list(model.parameters())

        self.optimizers = [torch.optim.Adam(params, lr=lr, weight_decay=wd)]
        self.z_dim = z_dim
        self.lr = lr
        self.wd = wd

    def encode_and_reconstruct(self, y, x, previous_ys):
        z = self.encode(y, x, previous_ys)
        y_rec = self.decode(z, x, previous_ys)
        return y_rec, z

    def encode(self, y, x, previous_ys):
        z, _, _ = self.get_encode_parameters(y, x, previous_ys=previous_ys)
        return z

    def get_encode_parameters(self, y, x, previous_ys):
        x_extracted = self.x_feature_extractor(x, previous_ys)
        #  Sample a latent vector z given an input x from the posterior q(Z|x,y).
        encoded = self.features_encoder(torch.cat([x[:, -1], x_extracted, y], dim=-1))

        mu = self.mu_linear(encoded)
        log_sigma2 = self.sigma_linear(encoded)
        log_sigma2 = torch.minimum(log_sigma2, torch.ones_like(log_sigma2) * 10)

        std = torch.exp(0.5 * log_sigma2)
        z = mu + torch.randn_like(std) * std

        return z, mu, log_sigma2

    def decode(self, z, x, previous_ys):
        if previous_ys.shape[0] == 1 and x.shape[0] > 1:
            previous_ys = previous_ys.repeat(x.shape[0], 1, 1)
        x_and_time_extraction = self.x_feature_extractor(x, previous_ys)
        z_decoded = self.decoder(z)
        y_rec = self.features_decoder(torch.cat([x[:, -1], z_decoded, x_and_time_extraction], dim=-1))

        return y_rec

    def loss_aux(self, all_pre_x, all_pre_y, desired_coverage_level):
        batch_size = min(all_pre_x.shape[0], 64)
        y = all_pre_y[-batch_size:, -1]
        x = all_pre_x[-batch_size:]
        previous_ys = all_pre_y[-batch_size:, :-1]

        z, mu, log_sigma2 = self.get_encode_parameters(x=x, y=y, previous_ys=previous_ys)
        y_rec = self.decode(z=z, x=x, previous_ys=previous_ys)
        kl_weight = 0.01
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma2 - mu ** 2 - log_sigma2.exp(), dim=1), dim=0)
        kl_loss = kl_loss * kl_weight

        reconstruction_loss = F.mse_loss(y_rec, y)
        loss = kl_loss + reconstruction_loss

        return loss

    @staticmethod
    def generate_z_in_radius(radius, z_dim, z_sample_size, device):
        z = torch.randn(int(5 * z_sample_size), z_dim).to(device)
        z = z[z.norm(dim=-1) < radius][:z_sample_size]

        while z.shape[0] < z_sample_size:
            z = torch.randn(int(5 * z_sample_size), z_dim).to(device)
            z = z[z.norm(dim=-1) < radius][:z_sample_size]

        return z

    @staticmethod
    def stretch_samples(samples, center, stretch_strength):
        center = center.unsqueeze(1).repeat(1, samples.shape[1], 1)
        samples = samples - center
        samples *= stretch_strength
        samples += center
        return samples

    def decode_z_batch(self, x, previous_ys, z):
        z_sample_size = z.shape[0]
        x_rep = x.unsqueeze(1).repeat(1, z_sample_size, 1, 1).flatten(0, 1)
        previous_ys_rep = previous_ys.unsqueeze(1).repeat(1, z_sample_size, 1, 1).flatten(0, 1)
        z_rep = z.unsqueeze(0).repeat(x.shape[0], 1, 1).flatten(0, 1)
        unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(x.shape[0], z_sample_size))
        decoded = unflatten(self.decode(z_rep, x_rep, previous_ys_rep))
        return decoded

    def compute_quantile_region_sample(self, x, previous_ys, desired_coverage, y_stretching=1):
        radius_in_z_space = torch.sqrt(torch.Tensor([chi2.ppf(desired_coverage, df=self.z_dim)]).to(x.device))

        z_sample_size = int(z_dim_to_quantile_region_sample_size[self.z_dim])
        z = TSCVAE.generate_z_in_radius(radius_in_z_space, self.z_dim, z_sample_size, x.device)
        z = torch.cat([torch.zeros(1, z.shape[-1]).to(z.device), z], dim=0)

        quantile_region_sample = self.decode_z_batch(x, previous_ys, z)
        center = quantile_region_sample[:, 0]
        quantile_region_sample = TSCVAE.stretch_samples(quantile_region_sample, center, y_stretching)
        return quantile_region_sample

    def _TSModel__construct_interval_aux(self, x, previous_ys, desired_coverage, true_y=None, y_stretching=1):
        quantile_region_sample = self.compute_quantile_region_sample(x, previous_ys, desired_coverage, y_stretching)
        return QuantileRegions(quantile_region_sample, self.dataset.y_train, self.scaler)

    def get_uncertainty_quantification_set_class(self):
        return QuantileRegionSet

    def plot_losses(self):
        pass

    def update_models(self, models):
        self.models = models
        self.x_feature_extractor, self.features_encoder, self.features_decoder, self.mu_linear, \
        self.sigma_linear, self.decoder = self.models
        params = list(itertools.chain(*[list(model.parameters()) for model in self.models]))
        self.optimizers = [torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)]

    def plot_quantile_region(self, x, previous_ys, desired_coverage):

        for stretching in reversed([1, 2, 5, 10]):
            z_dim_to_quantile_region_sample_size[2] *= 10
            pts = self.compute_quantile_region_sample(x, previous_ys, desired_coverage,
                                                      y_stretching=stretching).squeeze().cpu()
            plt.scatter(pts[:, 0], pts[:, 1], label=f"s={stretching}")
            plt.xlabel("$Y_0$")
            plt.ylabel("$Y_1$")
            plt.legend()
            z_dim_to_quantile_region_sample_size[2] /= 10

        plt.show()
