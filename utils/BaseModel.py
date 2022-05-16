from torch import nn
from utils.utils import get_non_linearity


class BaseModel(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims=None, n_layers=3, dropout=0., bias=True, non_linearity='lrelu',
                 batch_norm=False, last_layer=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64] * n_layers

        n_layers = len(hidden_dims)
        if n_layers == 0:
            self.intermediate_layer = nn.Sequential()
            modules = [nn.Linear(in_dim, out_dim, bias=bias)]

        else:
            modules = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
            if dropout > 0:
                modules += [nn.Dropout(dropout)]
            if batch_norm:
                modules += [nn.BatchNorm1d(hidden_dims[0])]
            modules += [get_non_linearity(non_linearity)()]

            for i in range(n_layers - 1):
                modules += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias)]
                if batch_norm:
                    modules += [nn.BatchNorm1d(hidden_dims[i + 1])]
                modules += [get_non_linearity(non_linearity)()]
                if dropout > 0:
                    modules += [nn.Dropout(dropout)]

            self.intermediate_layer = nn.Sequential(*modules)

            modules += [nn.Linear(hidden_dims[-1], out_dim, bias=bias)]

        if last_layer is not None:
            modules += [last_layer()]

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

    def change_requires_grad(self, new_val):
        for p in self.parameters():
            p.requires_grad = new_val

    def freeze(self):
        self.change_requires_grad(True)

    def unfreeze(self):
        self.change_requires_grad(False)
