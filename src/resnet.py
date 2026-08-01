import torch
import torch.nn as nn

from src.fnn import FNN, get_activation, get_initializer
from src.model import LDNN


class ActivationModule(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class ResidualBlock1d(nn.Module):
    def __init__(self, input_dim, output_dim, activation, kernel_initializer, dropout=0.0):
        super().__init__()
        self.activation1 = get_activation(activation)
        self.activation2 = get_activation(activation)
        self.init_func = get_initializer(kernel_initializer)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.init_params()

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                self.init_func(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        y = self.activation1(self.linear1(x))
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.linear2(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return self.activation2(y + self.shortcut(x))


class ResNN(nn.Module):
    def __init__(self, layer_sizes, activation, kernel_initializer, dropout=0.0):
        super().__init__()
        self.hidden_depth = len(layer_sizes) - 2
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.hidden_dim = layer_sizes[1]
        self.activation = get_activation(activation)
        self.init_func = get_initializer(kernel_initializer)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.num_residual_blocks = (self.hidden_depth - 1) // 2
        self.num_remaining_hidden_layers = (self.hidden_depth - 1) % 2
        self.residual_blocks = nn.ModuleList(
            ResidualBlock1d(self.hidden_dim, self.hidden_dim, activation, kernel_initializer, dropout)
            for _ in range(self.num_residual_blocks)
        )
        if self.num_remaining_hidden_layers == 1:
            layers = [nn.Linear(self.hidden_dim, self.hidden_dim), ActivationModule(get_activation(activation))]
            if self.dropout is not None:
                layers.insert(1, self.dropout)
            self.remaining_hidden_layers = nn.ModuleList(layers)
        else:
            self.remaining_hidden_layers = nn.ModuleList()
        self.input_hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_params()

    def init_params(self):
        for block in self.residual_blocks:
            block.init_params()
        for layer in self.remaining_hidden_layers:
            if isinstance(layer, nn.Linear):
                self.init_func(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.init_func(self.input_hidden_linear.weight)
        if self.input_hidden_linear.bias is not None:
            nn.init.zeros_(self.input_hidden_linear.bias)
        self.init_func(self.hidden_output_linear.weight)
        if self.hidden_output_linear.bias is not None:
            nn.init.zeros_(self.hidden_output_linear.bias)

    def forward(self, x):
        x = self.activation(self.input_hidden_linear(x))
        if self.dropout is not None:
            x = self.dropout(x)
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.remaining_hidden_layers:
            x = layer(x)
        return self.hidden_output_linear(x)


class ResLDNN(LDNN):
    def __init__(self, layer_sizes_dyn, layer_sizes_rec, activation, kernel_initializer, dropout=0):
        if isinstance(dropout, list):
            dyn_dropout, rec_dropout = dropout
        else:
            dyn_dropout = rec_dropout = dropout
        dyn = FNN(layer_sizes_dyn, activation, kernel_initializer, dropout=dyn_dropout)
        rec = ResNN(layer_sizes_rec, activation, kernel_initializer, dropout=rec_dropout)
        super().__init__(dyn, rec)

