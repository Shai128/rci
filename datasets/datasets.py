"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import abc
from abc import ABC

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

import helper
from helper import set_seeds, get_current_seed
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

from utils.DataScaler import DataScaler

sys.path.insert(1, '..')


class DataGeneratorFactory:
    possible_syn_dataset_names = ['simple_invertible_data', 'simple_non_invertible_data', 'simple_continuous_data',
                                  'indp_x_continuous_data', 'simple_cyclic_data',
                                  'miscoverage_streak', 'asymmetric_miscoverage_streak',
                                  'window']

    @staticmethod
    def get_data_generator(dataset_name, is_real, args):
        if is_real:
            return DataGeneratorFactory.get_real_data_generator(dataset_name, args)
        else:
            return DataGeneratorFactory.get_syn_data_generator(dataset_name, args)

    @staticmethod
    def get_real_data_generator(dataset_name, args):
        return RealDataGenerator(dataset_name, args)

    @staticmethod
    def get_syn_data_generator(dataset_name, args):

        assert any(
            possible_dataset in dataset_name for possible_dataset in DataGeneratorFactory.possible_syn_dataset_names)
        if 'x_dim' in dataset_name:
            x_dim = int(re.search(r'\d+', re.search(r'x_dim_\d+', dataset_name).group()).group())
        else:
            x_dim = 1
        if 'z_dim' in dataset_name:
            data_z_dim = int(re.search(r'\d+', re.search(r'z_dim_\d+', dataset_name).group()).group())
        else:
            data_z_dim = 1

        if 'window' in dataset_name:
            window_length = float(re.search(r'\d+', re.search(r'len_\d+', dataset_name).group()).group())
            return WindowSynDataGenerator(x_dim, data_z_dim, window_length)
        else:
            assert False


class DataGenerator(ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_data(self, T, x=None, previous_data_info=None):
        pass


def get_features_to_use_as_y(dataset_name) -> list:
    if dataset_name == 'tetuan_power':
        return [0, 1]
    else:
        raise NotImplementedError()


class RealDataGenerator(DataGenerator):
    def __init__(self, dataset_name, args):
        super().__init__()
        self.dataset_name = dataset_name
        self.x, self.y = GetDataset(dataset_name, 'datasets/real_data/')
        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y)
        if args.method_type == 'mqr':
            idx = get_features_to_use_as_y(dataset_name)
            self.y = torch.cat([self.y.squeeze().unsqueeze(-1), self.x[:, idx]], dim=-1)
            self.x = self.x[:, list(set(range(self.x.shape[-1])) - set(idx))]
        # self.x = torch.cat([self.x, self.y], dim=-1)
        self.max_data_size = self.x.shape[0]

    def generate_data(self, T, x=None, previous_data_info=None, device='cpu'):

        if previous_data_info is None:
            starting_time = 0
        else:
            starting_time = previous_data_info['ending_time'] + 1

        current_process_info = {'ending_time': starting_time + T - 1}
        return self.x[starting_time: starting_time + T].cpu().to(device), \
               self.y[starting_time:starting_time + T].cpu().to(device), current_process_info


class SynDataGenerator(ABC):
    def __init__(self):
        self.max_data_size = np.inf

    @abc.abstractmethod
    def generate_data(self, T, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, use_constant_seed=True):
        pass

    @abc.abstractmethod
    def get_y_given_x_and_uncertainty(self, x, uncertainty, previous_data_info=None):
        pass

    @abc.abstractmethod
    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):
        pass


class SimpleSynDataGenerator(SynDataGenerator):
    def __init__(self, x_dim, data_z_dim, z_rho=0.9):
        super().__init__()
        self.x_dim, self.data_z_dim = x_dim, data_z_dim
        self.beta = torch.rand(x_dim)
        self.beta /= self.beta.norm(p=1)

        self.beta2 = torch.rand(data_z_dim)
        self.beta2 /= self.beta2.norm(p=1)
        self.z_rho = z_rho

    @staticmethod
    def generate_beta(dim):
        beta = torch.rand(dim)
        beta /= beta.norm(p=1)
        return beta

    @abc.abstractmethod
    def generate_y_given_x_z(self, x, z, device='cpu', previous_data_info=None, current_process_info=None):
        pass

    def generate_x(self, z, n_samples, T, device, previous_data_info=None):
        x = torch.rand(n_samples, T, self.x_dim, device=device)
        return x

    def generate_data_given_z(self, z, x=None, n_samples=1, device='cpu', previous_data_info=None,
                              current_process_info=None):
        if x is None:
            x = self.generate_x(z, n_samples, z.shape[1], device, previous_data_info=previous_data_info)
        return x, self.generate_y_given_x_z(x, z, device, previous_data_info=previous_data_info,
                                            current_process_info=current_process_info)

    def generate_z(self, T, n_samples, device, previous_data_info=None, current_process_info=None):
        rho = self.z_rho
        z = torch.zeros(n_samples, T, self.data_z_dim, device=device)
        if previous_data_info is not None:
            pre_z = previous_data_info['z']
        else:
            pre_z = None

        if pre_z is not None and current_process_info is None:
            assert len(pre_z.shape) == 3 and pre_z.shape[2] == self.data_z_dim and pre_z.shape[0] >= 1
            pre_z = pre_z[:, -1:]
            initial_new_index = len(pre_z)
            z = torch.cat([pre_z.repeat(n_samples, 1, 1), z], dim=1)
        elif pre_z is not None and current_process_info is not None:
            initial_new_index = 1
            z[:, 0] = rho * pre_z[:, - 1] + torch.randn(n_samples, self.data_z_dim, device=device) * (
                    (1 - rho ** 2) ** 0.5)
        else:
            initial_new_index = 1
            z[:, 0] = torch.randn(n_samples, self.data_z_dim, device=device)

        if current_process_info is not None:
            current_process_z = current_process_info['z'].repeat(n_samples, 1, 1)
        else:
            current_process_z = z

        for t in range(initial_new_index, z.shape[1]):
            z[:, t] = rho * current_process_z[:, t - 1] + torch.randn(n_samples, self.data_z_dim, device=device) * (
                    (1 - rho ** 2) ** 0.5)

        if pre_z is not None and current_process_info is None:
            z = z[:, pre_z.shape[0]:]
        return z

    def generate_data(self, T, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, get_z=False, use_constant_seed=True):
        if use_constant_seed:
            initial_seed = get_current_seed()
            set_seeds(0)
        z = self.generate_z(T, n_samples=n_samples, previous_data_info=previous_data_info,
                            current_process_info=current_process_info, device=device)
        x, y = self.generate_data_given_z(z, x=x, n_samples=n_samples, device=device,
                                          previous_data_info=previous_data_info,
                                          current_process_info=current_process_info)
        initial_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        curr_data_info = {'z': z, 'y': y, 'ending_time': initial_time + T - 1}
        if n_samples == 1:
            z, x, y = z.squeeze(0), x.squeeze(0), y.squeeze(0)

        if use_constant_seed:
            set_seeds(initial_seed)

        if get_z:
            return z, x, y, curr_data_info
        else:
            return x, y, curr_data_info

    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):

        device = x_test.device
        n_points_to_sample = 2000
        z_set = torch.randn(n_points_to_sample, self.data_z_dim, device=device)
        z_set_size = z_set.shape[0]
        z_set = z_set.reshape(z_set_size, self.data_z_dim)

        unflattened_y_set = self.generate_y_given_x_z(x_test.unsqueeze(0).repeat(z_set_size, 1, 1),
                                                      z_set.unsqueeze(1).repeat(1, x_test.shape[0], 1), device=device,
                                                      current_process_info=current_process_info,
                                                      previous_data_info=previous_data_info)

        y_upper = unflattened_y_set.quantile(dim=0, q=1 - alpha / 2)
        y_lower = unflattened_y_set.quantile(dim=0, q=alpha / 2)
        return y_lower, y_upper

    def get_y_given_x_and_uncertainty(self, x, z, previous_data_info=None):
        return self.generate_y_given_x_z(x, z, device=x.device, previous_data_info=previous_data_info)

    def reduce_x_and_z(self, x, z):
        self.beta = self.beta.to(x.device)
        self.beta2 = self.beta2.to(x.device)
        x = x @ self.beta
        z = z @ self.beta2
        return x, z


class WindowSynDataGenerator(SimpleSynDataGenerator):

    def __init__(self, x_dim, data_z_dim, window_length):
        super().__init__(x_dim, data_z_dim, z_rho=0.)
        self.window_length = window_length
        max_data_size = 50000
        self.y_rho = 0.5
        distr_num = torch.zeros(max_data_size).int()
        curr_idx = 0
        curr_count = 0
        while curr_idx < len(distr_num) - 1:
            next_idx = curr_idx + (window_length + torch.randn(1) * min(10, window_length // 20)).int().item()
            next_idx = max(next_idx, curr_idx)
            next_idx = min(next_idx, len(distr_num))

            distr_num[curr_idx:next_idx] = curr_count

            curr_count = curr_count + 1
            curr_idx = next_idx

        self.data_type_idx = distr_num.int()
        n_data_types = self.data_type_idx.max()
        self.x_betas = [WindowSynDataGenerator.generate_beta(x_dim) for _ in range(n_data_types)]
        self.noise_levels = [(torch.rand(1).abs().item() * 10 + 20) for _ in range(n_data_types)]

    def generate_y_given_x_z(self, x, z, device='cpu', previous_data_info=None, current_process_info=None):

        x, z = x.to(device), z.to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
        initial_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        assert z.shape[0] == x.shape[0] or x.shape[0] == 1 or z.shape[0] == 1
        if x.shape[0] == 1 and z.shape[0] != 1:
            x = x.repeat(z.shape[0], 1, 1)
        if x.shape[0] != 1 and z.shape[0] == 1:
            z = z.repeat(x.shape[0], 1, 1)

        if previous_data_info is not None:
            pre_y = previous_data_info['y']
            pre_y = pre_y[:, -1:].to(device).repeat(x.shape[0], 1)

        else:
            pre_y = torch.ones(x.shape[0], 1, device=device)

        y = torch.zeros(x.shape[0], x.shape[1], device=device)
        if current_process_info is None:
            pre_y_and_curr_y = torch.cat([pre_y, y], dim=1)
            for t in range(1, pre_y_and_curr_y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t - 1], z[:, t - 1], t - 1 + initial_time)
                pre_y_and_curr_y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1] + curr_time_factor
            y = pre_y_and_curr_y[:, 1:]

        else:
            current_y_process = current_process_info['y'].repeat(x.shape[0], 1)
            pre_y_and_curr_y = torch.cat([pre_y, current_y_process], dim=1)
            for t in range(0, y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t], z[:, t], t + initial_time)
                y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1 + pre_y.shape[1]] + curr_time_factor
        return y

    def curr_time_factor(self, x, z, absolute_time):
        reduced_x = x @ self.x_betas[self.data_type_idx[absolute_time]].to(x.device)
        reduced_x = reduced_x.abs()
        small_uncertainty = torch.sin(2 * x[..., 1] * z[..., 0]) * 2

        if int(self.data_type_idx[absolute_time].item()) % 2 == 0:
            noise_level = self.noise_levels[self.data_type_idx[absolute_time]] ** 2
        else:
            noise_level = 1

        large_uncertainty = noise_level * reduced_x
        return small_uncertainty + large_uncertainty.abs()


class DataSet:

    def __init__(self, data_generator, is_real_data, device, T, test_ratio):
        validation_ratio = 0
        train_ratio = 1 - test_ratio - validation_ratio
        self.data_generator = data_generator
        self.is_real_data = is_real_data
        self.x_train, self.y_train, self.training_data_info = data_generator.generate_data(int(train_ratio * T),
                                                                                           device=device)
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.unsqueeze(-1)

        self.data_scaler = DataScaler()
        self.data_scaler.initialize_scalers(self.x_train, self.y_train)
        # self.y_train = self.data_scaler.scale_y(self.y_train)

        if validation_ratio is not None and validation_ratio > 0:
            self.x_val, self.y_val, self.pre_test_data_info = data_generator.generate_data(int(validation_ratio * T),
                                                                                           device=device,
                                                                                           previous_data_info=self.training_data_info)
            self.y_val = self.y_val.unsqueeze(1)
            self.starting_test_time = self.x_train.shape[0] + self.x_val.shape[0]
            # self.y_val = self.data_scaler.scale_y(self.y_val)

        else:
            self.y_val = None
            self.pre_test_data_info = self.training_data_info
            self.starting_test_time = self.x_train.shape[0]

        self.x_test, self.y_test, self.test_data_info = data_generator.generate_data(int(test_ratio * T), device=device,
                                                                                     previous_data_info=self.pre_test_data_info)
        if len(self.y_test.shape) == 1:
            self.y_test = self.y_test.unsqueeze(-1)

        if self.y_val is not None:
            all_y = torch.cat([self.y_train, self.y_val, self.y_test], dim=0)
        else:
            all_y = torch.cat([self.y_train, self.y_test], dim=0)

        all_y_scaled = self.data_scaler.scale_y(all_y)

        self.y_scaled_min = all_y_scaled.min().item()
        self.y_scaled_max = all_y_scaled.max().item()

        # self.y_test = self.data_scaler.scale_y(self.y_test)


def GetDataset(name, base_path):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""
    if name == 'energy':
        df = pd.read_csv(base_path + 'energy.csv')
        y = np.array(df['Appliances'])
        X = df.drop(['Appliances', 'date'], axis=1)
        date = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'tetuan_power':
        df = pd.read_csv(base_path + 'tetuan_power.csv')
        y = np.array(df['Zone 1 Power Consumption'])
        X = df.drop(['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'],
                    axis=1)
        date = pd.to_datetime(df['DateTime'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%m/%d/%Y %H:%M')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'traffic':
        df = pd.read_csv(base_path + 'traffic.csv')
        df['holiday'].replace(df['holiday'].unique(),
                              list(range(len(df['holiday'].unique()))), inplace=True)
        df['weather_description'].replace(df['weather_description'].unique(),
                                          list(range(len(df['weather_description'].unique()))), inplace=True)
        df['weather_main'].replace(['Clear', 'Haze', 'Mist', 'Fog', 'Clouds', 'Smoke', 'Drizzle', 'Rain', 'Squall',
                                    'Thunderstorm', 'Snow'],
                                   list(range(len(df['weather_main'].unique()))), inplace=True)
        y = np.array(df['traffic_volume'])
        X = df.drop(['date_time', 'traffic_volume'], axis=1)
        date = pd.to_datetime(df['date_time'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        # X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek

        X = np.array(X)

    if name == 'wind':
        df = pd.read_csv(base_path + 'wind_power.csv')
        date = pd.to_datetime(df['dt'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['dt', 'MW'], axis=1)
        y = np.array(df['MW'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['minute'] = date.dt.minute
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    if name == 'prices':
        df = pd.read_csv(base_path + 'Prices_2016_2019_extract.csv')
        # 15/01/2016  4:00:00
        date = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['Date', 'Spot', 'hour'], axis=1)
        y = np.array(df['Spot'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    except Exception as e:
        raise Exception("invalid dataset")

    return X, y


"""
░░░░░█░░█░█▀▀█░█▀▀▄░█▀▀▄░░░░░
░░░░░█▀▀█░█▀▀█░█▀▀█░█░░█░░░░░
░░░░░▀░░▀░▀░░▀░▀░░▀░▀▀▀░░░░░░
▀█▀░█░█░▀█▀░█▀█░█▀▀▄░█░█▀▀█░█░░
░█░░█░█░░█░░█░█░█▀▀█░█░█▀▀█░█░░
░▀░░▀▀▀░░▀░░▀▀▀░▀░░▀░▀░▀░░▀░▀▀░


━━━━-╮
╰┃ ┣▇━▇
 ┃ ┃  ╰━▅╮
 ╰┳╯ ╰━━┳╯E Z A F
  ╰╮ ┳━━╯T Y 4
 ▕▔▋ ╰╮╭━╮ T U T O R I A L
╱▔╲▋╰━┻┻╮╲╱▔▔▔╲
▏  ▔▔▔▔▔▔▔  O O┃
╲╱▔╲▂▂▂▂╱▔╲▂▂▂╱
 ▏╳▕▇▇▕ ▏╳▕▇▇▕
 ╲▂╱╲▂╱ ╲▂╱╲▂╱
"""
