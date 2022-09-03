import torch
from torch import nn
from .mlp import feedforward_network
from pipeline.pipeline import compute_inverse_design
from .chen07 import Chen07Nonparametric


class SingleLayerSigmoid(nn.Module):
    """
    A single layer sigmoid network with a fast jacobian option
    """

    def __init__(self, input_size, width):
        super().__init__()
        self.width = width
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, width)
        self.l2 = nn.Linear(width, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        self.after_l1 = self.l1(x)
        self.after_sigmoid = self.act(self.after_l1)
        self.after_l2 = self.l2(self.after_sigmoid)
        return self.after_l2

    def jacobian(self, x):
        if x is not None:
            output = self.forward(x)
        with torch.no_grad():
            self.grad_l2_weight = self.after_sigmoid.detach()  # n x h
            self.grad_l2_bias = torch.ones(len(x))

            t = ((self.after_sigmoid) * (1 - self.after_sigmoid)).detach()
            self.grad_l1_weight = (
                (x[:, :, None] * t[:, None, :]) * self.l2.weight
            ).transpose(1, 2)
            self.grad_l1_bias = (t[:, None, :] * self.l2.weight).squeeze(1)

            grad_l1_weight_flat = self.grad_l1_weight.reshape((len(x), -1))
            self.flattened_grad = torch.cat(
                [
                    grad_l1_weight_flat,
                    self.grad_l1_bias,
                    self.grad_l2_weight,
                    self.grad_l2_bias.unsqueeze(1),
                ],
                dim=1,
            )
        return (
            self.flattened_grad,
            (
                self.grad_l1_weight,
                self.grad_l1_bias,
                self.grad_l2_weight,
                self.grad_l2_bias,
            ),
        )


class NonparametricWithJacobian(Chen07Nonparametric):
    def jacobian(self, w):
        """
        Get the jacobian df(x; weights)/dweights
        """
        grads = []
        t = self(w)
        for i in range(len(w)):
            self.zero_grad()
            t[i].backward(retain_graph=True)
            grads.append(
                nn.utils.convert_parameters.parameters_to_vector(
                    [p.grad for p in self.parameters()]
                )
            )
        return torch.stack(grads)
