import torch
from torch import nn
from utils.BaseModel import BaseModel


class LSTMModel(nn.Module):

    def __init__(self, in_dim, y_dim, out_dim, args, lstm_hidden_size=64, lstm_layers=3,
                 lstm_in_layers=[32], use_previous_y=True,
                 lstm_out_layers=[64], dropout=0., non_linearity='lrelu'):
        super().__init__()

        if len(lstm_in_layers) > 0:
            lstm_in_dim = lstm_hidden_size
            self.lstm_in_model = BaseModel(in_dim + (y_dim if use_previous_y else 0), lstm_hidden_size, hidden_dims=lstm_in_layers,
                                            dropout=dropout, non_linearity=non_linearity)
        else:
            lstm_in_dim = in_dim+y_dim

        if lstm_layers == 0 or lstm_hidden_size == 0:
            self.lstm = None
            lstm_out_dim = in_dim
        else:
            lstm_out_dim = in_dim + lstm_hidden_size
            self.lstm = nn.LSTM(input_size=lstm_in_dim, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                                batch_first=True,
                                dropout=dropout)

        self.use_previous_y = use_previous_y
        self.lstm_out_layers = BaseModel(lstm_out_dim, out_dim,
                                         hidden_dims=lstm_out_layers,
                                         dropout=dropout, non_linearity=non_linearity)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_in_layers = lstm_in_layers

    def forward(self, x, previous_ys):

        if len(self.lstm_in_layers) > 0 and self.lstm is not None:
            pre_lstm_in = torch.cat([x[:, :-1, :], previous_ys], dim=-1) if self.use_previous_y else x[:, :-1, :]
            lstm_in = self.lstm_in_model(pre_lstm_in)
        else:
            lstm_in = torch.cat([x[:, :-1, :], previous_ys], dim=-1)

        if self.lstm is not None:
            lstm_out, _ = self.lstm(lstm_in)
            lstm_out = lstm_out[:, -1, :]
            lstm_out_layers_in = torch.cat([lstm_out, x[:, -1, :]], dim=-1)
        else:
            lstm_out_layers_in = x[:, -1, :]

        return self.lstm_out_layers(lstm_out_layers_in)  # take only the output of the last element in the sequence
