import torch
from sklearn.preprocessing import StandardScaler


class DataScaler():

    def __init__(self):

        self.s_tr_x = StandardScaler()
        self.s_tr_y = StandardScaler()

    def initialize_scalers(self, x_train, y_train):
        self.s_tr_x = self.s_tr_x.fit(x_train.cpu())
        self.s_tr_y = self.s_tr_y.fit(y_train.cpu())

    def scale_x(self, *x_list):
        scaled_x = []
        for x in x_list:
            scaled_x += [torch.Tensor(self.s_tr_x.transform(x.detach().cpu())).to(x.device)]
        if len(scaled_x) == 1:
            scaled_x = scaled_x[0]
        return scaled_x

    def scale_y(self, *y_list):
        scaled_y = []
        for y in y_list:
            y = y.unsqueeze(-1) if len(y.shape) == 1 else y
            scaled_y += [torch.Tensor(self.s_tr_y.transform(y.detach().cpu())).to(y.device)]
        if len(scaled_y) == 1:
            scaled_y = scaled_y[0]
        return scaled_y

    def unscale_y(self, y):
        res = torch.Tensor(self.s_tr_y.inverse_transform(y.detach().cpu())).to(y.device).squeeze()
        return res

    def scale_x_y(self, x, y):
        x, y = self.scale_x(x), self.scale_y(y)
        return x, y
