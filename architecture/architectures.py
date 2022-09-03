import numpy as np
import torch
from pipeline.pipeline import compute_inverse_design, project
from torch import nn

from .mlp import feedforward_network


def _get_partially_linear_standard_error(
    endogenous_of_interest,
    transformed_endogenous,
    transformed_instrument,
    inverse_variance,
    projected_residual_variance=None,
    inverse_design_instrument=None,
    return_beta=False,
):
    with torch.no_grad():
        if projected_residual_variance is None:
            projected_residual_variance = 1 / inverse_variance

        if inverse_design_instrument is None:
            inverse_design_instrument = compute_inverse_design(transformed_instrument)

        endogenous_projected = project(
            inverse_design_instrument, transformed_instrument, transformed_endogenous,
        )

        endogenous_interest_projected = project(
            inverse_design_instrument, transformed_instrument, endogenous_of_interest,
        )

        denominator = (
            endogenous_projected.T
            @ (inverse_variance ** 0.5 * endogenous_projected)
            / len(endogenous_projected)
        )
        numerator = (
            endogenous_projected.T
            @ (inverse_variance ** 0.5 * endogenous_interest_projected)
            / len(endogenous_projected)
        )

        # Get sieve coefficients for w
        beta = torch.pinverse(denominator, rcond=1e-6) @ numerator

        if return_beta:
            return beta

        w_star = transformed_endogenous @ beta
        dw_star = -project(
            inverse_design_instrument,
            transformed_instrument,
            endogenous_of_interest - w_star,
        )

        g_matrix = (dw_star ** 2 * inverse_variance).mean().item()
        omega_matrix = (
            ((dw_star ** 2) * (inverse_variance ** 2) * projected_residual_variance)
            .mean()
            .item()
        )

    return (omega_matrix / (g_matrix ** 2) / len(inverse_variance)) ** 0.5


class PartiallyLinear(nn.Module):
    def __init__(self, bootstrap_weights=None, *mlp_args, **mlp_kwargs):
        """
        Arguments
        def feedforward_network(
            input_dim,
            depth,
            width,
            output_dim=1,
            hidden_activation=<class 'torch.nn.modules.activation.ReLU'>,
            output_activation=None,
        )
        """
        super().__init__()
        self.linear_param = nn.Parameter(torch.tensor([0.0]))
        self.mlp = feedforward_network(*mlp_args, **mlp_kwargs)
        self.bootstrap_weights = bootstrap_weights

    def forward(self, endogenous):
        nonlinear = self.mlp(endogenous[:, 1:])
        linear = self.linear_param * endogenous[:, [0]]
        return linear + nonlinear

    def get_parameter_of_interest(self, *args):
        return self.linear_param.item()

    def get_standard_error(
        self,
        endogenous_of_interest,
        transformed_endogenous,
        transformed_instrument,
        inverse_variance,
        projected_residual_variance=None,
        inverse_design_instrument=None,
        **kwargs,
    ):
        """
        Return the standard error of the partially linear parameter

        Parameters
        ----------
        endogenous_of_interest : torch.Tensor
            Endogenous variable that correspond to the linear part
        transformed_endogenous : torch.Tensor
            Linear basis of all other endogenous variables
            for estimating the standard error
        transformed_instrument : torch.Tensor
            Linear basis of instruments, can be torchvec["transformed_instrument"]
        inverse_variance : torch.Tensor
            n x 1 vector of weights for efficient weighting
        projected_residual_variance : torch.Tensor, optional
            an estimate of E[residual^2 | instrument], in efficiency weighting
            this is 1/inverse_variance, by default 1/inverse_variance
        inverse_design_instrument : torch.Tensor, optional
            (Z.T @ Z / n)^-1 where Z is the transformed instrument, by default None

        Returns
        -------
        se : float
            Standard error of the variable
        """
        # raise NotImplementedError
        return _get_partially_linear_standard_error(
            endogenous_of_interest,
            transformed_endogenous,
            transformed_instrument,
            inverse_variance,
            projected_residual_variance,
            inverse_design_instrument,
            **kwargs,
        )


class Nonparametric(nn.Module):
    def __init__(
        self, moment_function=None, bootstrap_weights=None, *mlp_args, **mlp_kwargs
    ):
        """
        Arguments
        def feedforward_network(
            input_dim
            depth,
            width,
            output_dim=1,
            hidden_activation=<class 'torch.nn.modules.activation.ReLU'>,
            output_activation=None,
        )
        """
        super().__init__()
        self.mlp = feedforward_network(*mlp_args, **mlp_kwargs)
        self.bootstrap_weights = bootstrap_weights
        if moment_function is None:
            self.moment_function = (
                lambda y, yhat: y - yhat
                if self.bootstrap_weights is None
                else (y - yhat) * self.bootstrap_weights
            )
        else:
            self.moment_function = moment_function

    def forward(self, endogenous):
        return self.mlp(endogenous)

    def get_parameter_of_interest(self, endogenous):
        derivatives = self.get_derivatives(endogenous)
        return derivatives.mean().item()

    def get_derivatives(self, endogenous, index=0):
        device = next(self.parameters()).device
        vec = torch.zeros((len(endogenous), 1), requires_grad=True, device=device)
        x = endogenous.clone()
        x[:, [index]] += vec
        self.forward(x).sum().backward()
        return vec.grad.clone()

    def forward_filter_residuals(
        self,
        endogenous,
        response,
        inefficient_derivative,
        inefficient_prediction,
        weights,
        basis,
        inverse_design,
    ):
        moment_function = self.moment_function

        if inverse_design is None:
            inverse_design = compute_inverse_design(basis)
        efficient_derivative = self.get_derivatives(endogenous)
        residuals = moment_function(response, inefficient_prediction)
        unprojected_Gamma = (
            (efficient_derivative - inefficient_derivative.mean())
            * (residuals - residuals.mean())
            * weights
        )
        Gamma = basis @ inverse_design @ (basis.T @ unprojected_Gamma) / len(basis)
        filtered = efficient_derivative - Gamma * moment_function(
            response, self.forward(endogenous).detach()
        )
        return filtered, Gamma

    def _forward_filter_residuals(
        self, endogenous, response, inverse_design, transformed_instrument
    ):
        residuals = response - self.forward(endogenous)
        derivatives = self.get_derivatives(endogenous)
        projected_covariance = project(
            inverse_design,
            transformed_instrument,
            (derivatives - derivatives.mean()) * (residuals - residuals.mean()),
        )
        inverse_variance = 1 / project(
            inverse_design, transformed_instrument, residuals ** 2
        ).clamp(min=0.1)
        Gamma = inverse_variance * projected_covariance
        filtered = derivatives - Gamma * residuals

        return filtered.mean().item()

    def get_parameter_of_interest_with_correction(
        self,
        endogenous,
        response,
        inefficient_derivative,
        inefficient_prediction,
        weights,
        basis,
        inverse_design,
        return_standard_error=False,
    ):
        filtered, Gamma = self.forward_filter_residuals(
            endogenous,
            response,
            inefficient_derivative,
            inefficient_prediction,
            weights,
            basis,
            inverse_design,
        )
        if return_standard_error:
            return (
                filtered.mean().item(),
                filtered.std().item() / (len(filtered) ** 0.5),
            )
        else:
            return filtered.mean().item()

    def get_standard_error_nonparametric(
        self,
        filtered,  # assume that filtered is the derivatives
        Gamma,
        transformed_endogenous,
        transformed_endogenous_gradient,
        transformed_instrument,
        inverse_projected_variance,
        inverse_design_instrument=None,
        return_beta=False,
        rcond=1e-6,
        reg=0,
        weighting=True,
        residuals=None,
    ):
        filtered_variance = filtered.std() ** 2
        n = len(transformed_endogenous)
        if not weighting:
            Gamma = 0
            filtered_variance = 1
            inverse_projected_variance = 1

        with torch.no_grad():
            if inverse_design_instrument is None:
                inverse_design_instrument = compute_inverse_design(
                    transformed_instrument
                )

            dm1_dh = (
                transformed_endogenous_gradient + Gamma * transformed_endogenous
            ).mean(0, True)

            weight_1 = 1 / filtered_variance

            project_middle = (
                inverse_design_instrument
                @ transformed_instrument.T
                @ (inverse_projected_variance * transformed_instrument)
                @ inverse_design_instrument
                / n ** 2
            )

            sandwich = (
                transformed_endogenous.T
                @ transformed_instrument
                @ project_middle
                @ transformed_instrument.T
                @ transformed_endogenous
                / n
            )

            beta = -torch.pinverse(
                sandwich
                + weight_1 * dm1_dh.T @ dm1_dh
                + reg * torch.eye(len(sandwich)),
                rcond=rcond,
            ) @ (dm1_dh.T * weight_1)

            if return_beta:
                return beta

            if weighting:
                term1 = (1 + dm1_dh @ beta) ** 2 * weight_1
                term2 = (
                    project(
                        inverse_design_instrument,
                        transformed_instrument,
                        transformed_endogenous @ beta,
                    )
                    ** 2
                    * inverse_projected_variance
                ).mean()

                return ((1 / (term1 + term2) / n) ** 0.5).item()
            else:
                es_values = (
                    filtered
                    + project(
                        inverse_design_instrument,
                        transformed_instrument,
                        transformed_endogenous
                        @ beta
                        / (1 + transformed_endogenous_gradient @ beta).mean(),
                    )
                    * residuals
                )
                return es_values.std().item() / (n ** 0.5)

    def get_standard_error(
        self,
        endogenous_of_interest,
        transformed_endogenous,
        transformed_instrument,
        inverse_variance,
        projected_residual_variance=None,
        inverse_design_instrument=None,
        **kwargs,
    ):
        return _get_partially_linear_standard_error(
            endogenous_of_interest,
            transformed_endogenous,
            transformed_instrument,
            inverse_variance,
            projected_residual_variance,
            inverse_design_instrument,
            **kwargs,
        )


class PartiallyAdditive(nn.Module):
    def __init__(self, bootstrap_weights=None, *mlp_args, **mlp_kwargs):
        """
        Arguments
        def feedforward_network(
            input_dim,
            depth,
            width,
            output_dim=1,
            hidden_activation=<class 'torch.nn.modules.activation.ReLU'>,
            output_activation=None,
        )
        """
        super().__init__()
        self.linear_param = nn.Parameter(torch.tensor([0.0]))

        dim = mlp_kwargs["input_dim"]
        self.mlp1 = feedforward_network(
            *mlp_args, **{**mlp_kwargs, "input_dim": 1, "width": 3}, bias=False
        )
        self.mlp2 = feedforward_network(
            *mlp_args, **{**mlp_kwargs, "input_dim": 1, "width": 3}, bias=False
        )

        if dim > 2:
            self.mlp3 = feedforward_network(
                *mlp_args, **{**mlp_kwargs, "input_dim": dim - 2}
            )

        self.bootstrap_weights = bootstrap_weights

    def forward(self, endogenous):

        nonlin1 = self.mlp1(endogenous[:, [1]])
        nonlin2 = self.mlp2(endogenous[:, [2]])

        summed = nonlin1 + nonlin2
        if endogenous.shape[1] >= 4:
            summed += self.mlp3(endogenous[:, 3:])

        linear = self.linear_param * endogenous[:, [0]]
        return linear + summed

    def get_parameter_of_interest(self, *args):
        return self.linear_param.item()

    def get_standard_error(
        self,
        endogenous_of_interest,
        transformed_endogenous,
        transformed_instrument,
        inverse_variance,
        projected_residual_variance=None,
        inverse_design_instrument=None,
    ):
        return _get_partially_linear_standard_error(
            endogenous_of_interest,
            transformed_endogenous,
            transformed_instrument,
            inverse_variance,
            projected_residual_variance,
            inverse_design_instrument,
        )


class PartiallyAdditiveWithSpline(nn.Module):
    def __init__(
        self,
        bootstrap_weights=None,
        spline_deg=3,
        knots=2,
        knot_locs=None,
        *mlp_args,
        **mlp_kwargs,
    ):
        """
        Arguments
        def feedforward_network(
            input_dim,
            depth,
            width,
            output_dim=1,
            hidden_activation=<class 'torch.nn.modules.activation.ReLU'>,
            output_activation=None,
        )
        """
        super().__init__()
        self.linear_param = nn.Parameter(torch.tensor([0.0]))
        dim = mlp_kwargs["input_dim"]

        self.spline_deg = spline_deg
        self.knots = knots
        self.knot_locs = knot_locs
        self.spline1 = nn.Parameter(torch.zeros(spline_deg + knots - 1))
        self.spline2 = nn.Parameter(torch.zeros(spline_deg + knots - 1))

        if dim > 2:
            self.mlp3 = feedforward_network(
                *mlp_args, **{**mlp_kwargs, "input_dim": dim - 2}
            )

        self.bootstrap_weights = bootstrap_weights

    def forward(self, endogenous):
        x1 = endogenous[:, 1].numpy()
        x2 = endogenous[:, 2].numpy()

        knots1 = (
            np.quantile(x1, [i / (self.knots + 1) for i in range(1, self.knots + 1)])
            if self.knots != 2 or self.knot_locs is None
            else self.knot_locs[1, :]
        )

        knots2 = np.quantile(
            x2,
            [i / (self.knots + 1) for i in range(1, self.knots + 1)]
            if self.knots != 2 or self.knot_locs is None
            else self.knot_locs[1, :],
        )

        splx1 = torch.tensor(self.spl(x1, self.spline_deg, knots1)).float()
        splx2 = torch.tensor(self.spl(x2, self.spline_deg, knots2)).float()

        nonlin1 = (splx1 @ self.spline1).unsqueeze(-1)
        nonlin2 = (splx2 @ self.spline2).unsqueeze(-1)

        summed = nonlin1 + nonlin2
        if endogenous.shape[1] >= 4:
            summed += self.mlp3(endogenous[:, 3:])

        linear = self.linear_param * endogenous[:, [0]]
        return linear + summed

    def get_parameter_of_interest(self, *args):
        return self.linear_param.item()

    def spl(self, x, r, knots):
        poly = [x ** k for k in range(1, r)]
        other = [np.maximum(x - k, 0) ** (r - 1) for k in knots]
        return np.array(poly + other).T

    def get_standard_error(
        self,
        endogenous_of_interest,
        transformed_endogenous,
        transformed_instrument,
        inverse_variance,
        projected_residual_variance=None,
        inverse_design_instrument=None,
        **kwargs,
    ):
        return _get_partially_linear_standard_error(
            endogenous_of_interest,
            transformed_endogenous,
            transformed_instrument,
            inverse_variance,
            projected_residual_variance,
            inverse_design_instrument,
            **kwargs,
        )

