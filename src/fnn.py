import torch.nn as nn
import torch.nn.init as init


def get_activation(name):
    if name in ["sigmoid", "Sigmoid"]:
        return nn.Sigmoid()
    if name in ["tanh", "Tanh"]:
        return nn.Tanh()
    if name in ["relu", "ReLU"]:
        return nn.ReLU()
    if name in ["leakyrelu", "LeakyReLU", "leaky_relu"]:
        return nn.LeakyReLU()
    if name in ["prelu", "PReLU"]:
        return nn.PReLU()
    if name in ["rrelu", "RReLU"]:
        return nn.RReLU()
    if name in ["elu", "ELU"]:
        return nn.ELU()
    if name in ["selu", "SELU"]:
        return nn.SELU()
    if name in ["celu", "CELU"]:
        return nn.CELU()
    if name in ["gelu", "GELU"]:
        return nn.GELU()
    if name in ["silu", "SiLU"]:
        return nn.SiLU()
    if name in ["mish", "Mish"]:
        return nn.Mish()
    if name in ["glu", "GLU"]:
        return nn.GLU()
    raise NotImplementedError(f"Unsupported activation: {name}")


def get_initializer(name):
    if name in ["xavier_uniform", "Glorot uniform"]:
        return init.xavier_uniform_
    if name in ["xvaier_normal", "Glorot normal"]:
        return init.xavier_normal_
    if name == "kaiming_uniform":
        return init.kaiming_uniform_
    if name == "kaiming_normal":
        return init.kaiming_normal_
    if name == "uniform":
        return init.uniform_
    if name == "normal":
        return init.normal_
    if name == "trunc_normal":
        return init.trunc_normal_
    raise NotImplementedError(f"Unsupported initializer: {name}")


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation=None, kernel_initializer=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        )
        self.activation = activation if callable(activation) else get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.init = get_initializer(kernel_initializer) if kernel_initializer is not None else None
        if self.init is not None:
            self.apply_kernel_initializer()

    def apply_kernel_initializer(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return self.layers[-1](x)

