import os
import torch
import numpy as np
import random
from scipy.stats import chi2

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_current_seed():
    return torch.initial_seed()


def pearsons_corr(x, y):
    """
    computes the correlation between to samples of empirical samples
    Parameters
    ----------
    x - a vector if n samples drawn from X
    y - a vector if n samples drawn from Y
    Returns
    -------
    The empirical correlation between X and Y
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def is_in_normal_quantile_region(samples, mu, sigma_squared_vec, desired_coverage):
    x_minus_mu = (samples - mu)
    inv_sigma_squared_vec = sigma_squared_vec ** (-1)
    inv_sigma_squared_mat = torch.diag_embed(inv_sigma_squared_vec).repeat(x_minus_mu.shape[0], 1, 1)
    x_minus_mu = x_minus_mu.unsqueeze(-1)
    is_in_qr = torch.bmm(torch.bmm(x_minus_mu.transpose(1, 2), inv_sigma_squared_mat), x_minus_mu) <= chi2.ppf(
        desired_coverage, df=mu.shape[-1])
    return is_in_qr.squeeze()


def get_grid_from_borders(border_max, border_min, stride, device):
    if len(border_max.shape) == 0:
        grid = torch.arange(
            border_min, border_max, step=stride, dtype=torch.float32, device=device).unsqueeze(-1)
        return grid

    [torch.arange(
        0, 1, step=stride[..., i], dtype=torch.float32, device=device
    ) for i in range(border_max.shape[-1])]
    shifts = [torch.arange(
        border_min[..., i], border_max[..., i], step=stride[..., i], dtype=torch.float32, device=device
    ) for i in range(border_max.shape[-1])]

    grid = torch.cartesian_prod(*shifts)
    return grid


def get_normal_distr_quantile_region(mu, log_sigma2, desired_coverage, n_points_to_sample=1e5):
    sigma_vec = torch.exp(0.5 * log_sigma2)
    radius = 3 * sigma_vec.squeeze()
    border_max = mu.squeeze() + radius
    border_min = mu.squeeze() - radius

    stride = (border_max - border_min) / (n_points_to_sample ** (1 / mu.shape[-1]))
    samples = get_grid_from_borders(border_max, border_min, stride, device)

    if desired_coverage == 1:
        return samples
    is_in_qr = is_in_normal_quantile_region(samples, mu, sigma_vec ** 2, desired_coverage)
    return samples[is_in_qr]


def get_min_distance(y, points, ignore_zero_distance=False, y_batch_size=50, points_batch_size=10000) -> torch.Tensor:
    assert len(points.shape) == len(y.shape)
    if len(y.shape) == 2:
        return get_min_distance_aux(y, points, ignore_zero_distance, y_batch_size, points_batch_size)

    assert len(y.shape) == 3 and y.shape[0] == points.shape[0]
    res = []
    for i in range(y.shape[0]):
        res += [get_min_distance_aux(y[i], points[i], ignore_zero_distance, y_batch_size, points_batch_size)]
    return torch.stack(res)


def get_min_distance_aux(y, points, ignore_zero_distance=False, y_batch_size=50, points_batch_size=10000):
    min_dists_from_points = []
    for i in range(0, y.shape[0], y_batch_size):
        yi = y[i: min(i + y_batch_size, y.shape[0])]
        yi_min_dists_from_points = []
        for j in range(0, points.shape[0], points_batch_size):
            pts = points[j: min(j + points_batch_size, points.shape[0])]
            dist_from_pts = (yi - pts.unsqueeze(1).repeat(1, yi.shape[0], 1)).norm(dim=-1)
            if ignore_zero_distance:
                dist_from_pts[dist_from_pts == 0] = np.inf
            min_dist_from_pts = dist_from_pts.min(dim=0)[0]
            yi_min_dists_from_points += [min_dist_from_pts]

        if len(yi_min_dists_from_points) > 0:
            min_dists_from_points += [torch.stack(yi_min_dists_from_points)]

    if len(min_dists_from_points) == 0:
        return torch.Tensor([np.inf]).repeat(len(y)).to(y.device)
    else:
        return torch.cat(min_dists_from_points, dim=1).min(dim=0)[0]


def interp1d_func(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def batch_interp1d(x: torch.Tensor, y: torch.Tensor, a: float = None, b: float = None):
    if a is None or b is None:
        fill_value = 'extrapolate'
    else:
        fill_value = (a, b)

    def interp(desired_x):
        # desired_x = np.random.rand(3, 100) * 30 - 5
        desired_x = desired_x.to(x.device)
        if len(desired_x.shape) != 2 or desired_x.shape[0] != x.shape[0]:
            raise Exception(f"the shape of the input vector should be ({x.shape[0]},m), but got {desired_x.shape}")
        desired_x, _ = desired_x.sort()
        desired_x_rep = desired_x.unsqueeze(-1).repeat(1, 1, x.shape[-1] - 1)
        x_rep = x.unsqueeze(1).repeat(1, desired_x.shape[1], 1)
        relevant_idx = torch.stack(
            ((x_rep[:, :, :-1] <= desired_x_rep) & (desired_x_rep <= x_rep[:, :, 1:])).nonzero(as_tuple=True))

        x0 = x[relevant_idx[0], relevant_idx[2]]
        y0 = y[relevant_idx[0], relevant_idx[2]]
        x1 = x[relevant_idx[0], relevant_idx[2] + 1]
        y1 = y[relevant_idx[0], relevant_idx[2] + 1]
        desired_x_in_interpolation_range = desired_x[relevant_idx[0], relevant_idx[1]]
        res = torch.zeros_like(desired_x)
        res[relevant_idx[0], relevant_idx[1]] = interp1d_func(desired_x_in_interpolation_range, x0, x1, y0, y1)
        if fill_value == 'extrapolate':
            idx = (desired_x < x[:, 0, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], 0], x[idx[0], 1]
            y0, y1 = y[idx[0], 0], y[idx[0], 1]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

            idx = (desired_x > x[:, -1, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], -1], x[idx[0], -2]
            y0, y1 = y[idx[0], -1], y[idx[0], -2]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

        else:
            a, b = fill_value
            res[desired_x < x[:, 0, None]] = a
            res[desired_x > x[:, -1, None]] = b
        return res

    return interp


def batch_estim_dist(quantiles: torch.Tensor, percentiles: torch.Tensor, y_min, y_max, smooth_tails, tau,
                     extrapolate_quantiles=False):
    """ Estimate CDF from list of quantiles, with smoothing """
    device = quantiles.device
    noise = torch.rand_like(quantiles) * 1e-8
    noise_monotone, _ = torch.sort(noise)
    quantiles = quantiles + noise_monotone
    assert len(percentiles.shape) == 1 and len(quantiles.shape) == 2 and quantiles.shape[1] == percentiles.shape[0]
    percentiles = percentiles.unsqueeze(0).repeat(quantiles.shape[0], 1)

    # Smooth tails
    cdf = batch_interp1d(quantiles, percentiles, 0.0, 1.0)
    if extrapolate_quantiles:
        inv_cdf = batch_interp1d(percentiles, quantiles)
        return cdf, inv_cdf
    inv_cdf = batch_interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = torch.ones(quantiles.shape[0], 1, device=device) * tau
        tau_hi = torch.ones(quantiles.shape[0], 1, device=device) * (1 - tau)
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = torch.where(percentiles < tau_lo)[0]
        idx_hi = torch.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = torch.linspace(quantiles[0], q_lo, steps=len(idx_lo), device=device)
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = torch.linspace(q_hi, quantiles[-1], steps=len(idx_hi), device=device)

        cdf = batch_interp1d(quantiles_smooth, percentiles, 0.0, 1.0)

    # Standardize
    breaks = torch.linspace(y_min, y_max, steps=1000, device=device).unsqueeze(0).repeat(quantiles.shape[0], 1)
    cdf_hat = cdf(breaks)
    f_hat = torch.diff(cdf_hat)
    f_hat = (f_hat + 1e-10) / (torch.sum(f_hat + 1e-10, dim=-1)).reshape((f_hat.shape[0], 1))
    cumsum = torch.cumsum(f_hat, dim=-1)
    cdf_hat = torch.cat([torch.zeros_like(cumsum)[:, 0:1], cumsum], dim=-1)
    cdf = batch_interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = batch_interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf


def get_binary_vec_of_streaks(vec_len, avg_streak_size):
    binary_vec = torch.zeros(vec_len).int()
    curr_idx = 0
    curr_sign = 0
    while curr_idx < len(binary_vec) - 1:
        next_idx = curr_idx + (avg_streak_size + torch.randn(1) * 8).int().item()
        next_idx = max(next_idx, curr_idx + 50)
        next_idx = min(next_idx, len(binary_vec))

        binary_vec[curr_idx:next_idx] = 1 - curr_sign

        curr_sign = 1 - curr_sign
        curr_idx = next_idx

    return binary_vec.bool()


def get_soft_binary_vec_of_streaks(vec_len, avg_streak_size):
    binary_vec = torch.zeros(vec_len).int()
    curr_idx = 0
    curr_sign = 0
    while curr_idx < len(binary_vec) - 1:
        next_idx = curr_idx + (avg_streak_size + torch.randn(1) * 8).int().item()
        next_idx = max(next_idx, curr_idx + 50)
        next_idx = min(next_idx, len(binary_vec))

        while next_idx < len(binary_vec) - 1 and next_idx - curr_idx < 10:
            binary_vec[next_idx] = binary_vec[next_idx - 1] * 0.8
            next_idx += 1

        curr_sign = 1 - curr_sign
        curr_idx = next_idx

    return binary_vec.bool()
