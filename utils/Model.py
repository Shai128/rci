import torch
from torch import nn
from abc import abstractmethod

from utils.DataScaler import DataScaler
from utils.utils import optimizer_to


class Model(nn.Module):

    def __init__(self, dataset):
        super().__init__()
        self.scaler = DataScaler()
        self.dataset = dataset

    def initialize_scalers(self, x_train, y_train):
        self.scaler.initialize_scalers(x_train, y_train)
        scaled_y_train = self.scaler.scale_y(y_train)
        self.y_min = torch.min(scaled_y_train).item()
        self.y_max = torch.max(scaled_y_train).item()

    @abstractmethod
    def plot_losses(self):
        pass

    @property
    def models(self):
        return self._model

    @models.setter
    def models(self, value):
        self._model = value

    @property
    def optimizers(self):
        return self._optimizer

    @optimizers.setter
    def optimizers(self, value):
        self._optimizer = value

    def train(self, mode: bool = True):
        super(Model, self).train(mode)
        for model in self.models:
            model.train(mode)

    def eval(self):
        super(Model, self).eval()
        for model in self.models:
            model.eval()


    def to(self, device):
        super().to(device)
        for optimizer in self.optimizers:
            optimizer_to(optimizer, device)
        return self

    @abstractmethod
    def update_models(self, models):
        pass
