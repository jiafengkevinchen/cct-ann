from torch import nn
from torch.nn import Sequential


def feedforward_network(
    input_dim,
    depth,
    width,
    output_dim=1,
    hidden_activation=nn.ReLU,
    output_activation=None,
    bias=True,
):
    """
    Depth is the number of hidden layers. (Depth=0 is a generalized linear model)
    Can also think of depth as the number of calls to the hidden activation function
    """
    layer_lst = (
        [nn.Linear(input_dim, width), hidden_activation()]
        + [nn.Linear(width, width), hidden_activation()] * (depth - 1)
        + [nn.Linear(width, output_dim, bias=bias)]
    )
    if output_activation is not None:
        layer_lst.append(output_activation())
    return Sequential(*layer_lst)
