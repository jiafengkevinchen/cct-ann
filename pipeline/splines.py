import numpy as np
from statsmodels.tools import add_constant


def splines(var, deg, knots):
    spl_basis = spl(var, deg, knots)[:, 1:]
    dspl_basis = dspl(var, deg, knots)[:, 1:]
    d2spl_basis = d2spl(var, deg, knots)[:, 1:]
    return (spl_basis, dspl_basis, d2spl_basis)


def generate_endogenous_basis(npvec, knots, deg, pl=False, interact=True, quad_interact=False):
    """
    Returns the basis expansion of endogenous and its first derivative
    [psi_1(y1), ..., psi_K(y1)] and [\\nabla_1 psi_1(y1), ..., \\nabla_1 psi_K(y1)]

    For each variable y1i, the basis are Spl(deg, knots) for the variable, where
    Spl(deg, knots) is a degree (deg - 1) piecewise polynomial that is continuous in
    deg - 2 derivatives at locations prescribed by knots.

    Interactions (if interact=True) are of the form y1i * y1j
    unless quad_interact is set to True,
    in which case the endogenous variable of interest (y11) is interacted with y1j^2
    for j > 1 to better approximate nonlinear cross-partial behavior

    pl: partially linear
    """
    endo_of_interest = npvec["endogenous"][:, 0]
    endo = npvec["endogenous"][:, 1:]

    if not pl:
        endo0_basis, endo0_dbasis, endo0_d2basis = splines(endo_of_interest, deg, knots[0, :])

    else:
        endo0_basis = endo_of_interest[:, None]
        endo0_dbasis = np.ones_like(endo0_basis)

    basis = [add_constant(endo0_basis)]
    dbasisd0 = [np.c_[np.zeros(len(endo0_basis)), endo0_dbasis]]
    n, p = npvec["endogenous"].shape

    for i in range(endo.shape[1]):
        var = endo[:, i]
        sp, dsp, d2sp = splines(var, deg, knots[i + 1, :])

        basis.append(sp)
        dbasisd0.append(np.zeros_like(dsp))

        varj = endo[:, (i + 1) :]
        if interact:
            basis.append(varj * var[:, None])
            dbasisd0.append(np.zeros_like(varj))

    if not pl and interact:
        basis.append(endo_of_interest[:, None] * endo)
        dbasisd0.append(endo)

        if quad_interact:
            basis.append(endo_of_interest[:, None] * endo**2 / 2)
            dbasisd0.append(endo**2 / 2)

    return np.hstack(basis), np.hstack(dbasisd0)


def _instrument_basis(inst_mat, deg, knots_inst, interact):
    if interact == "full":
        interactions = [
            inst_mat[:, [i]] * inst_mat[:, [j]]
            for i in range(inst_mat.shape[1])
            for j in range(inst_mat.shape[1])
            if j > i
        ]
    elif interact == "light":
        interactions = [
            inst_mat[:, [i]] * inst_mat[:, [i + 1]] for i in range(inst_mat.shape[1] - 1)
        ]
    else:
        interactions = []

    transformed_instrument = add_constant(
        np.hstack(
            [
                splines(
                    inst_mat[:, i],
                    deg,
                    knots_inst[i, :],
                )[0]
                for i in range(inst_mat.shape[1])
            ]
            + interactions
        )
    )
    return transformed_instrument


def instrument_basis(npvec, deg, knots_inst, interact="full"):
    return _instrument_basis(npvec["instrument"], deg, knots_inst, interact)


def spl_experiment(
    npvec,
    deg,
    knots_inst,
    knots_endo,
    pl=False,
    full_return=False,
    interact="full",
    endogenous_interact=True,
    se=False,
    bootstrap_weights=None,
    rcond=1e-15,
    quad_interact=False,
):
    """
    Returns the spline NPIV estimate.

    npvec: numpy object for estimation (dict that contains endogenous,
        instrument, response as fields)
    deg: degree of the instrument spline (see generate_endogenous_basis for definition).
        The degree of the spline for endogenous variables is one fewer than that for the
        instruments.
    knots_*: knots of the instruments and endogenous variables, which is a K x J array if
        there are K knots and J variables
    pl: partially linaer
    full_return: return the basis expansions along with the estimates
    se: standard error estimates
    bootstrap_weights: mean 1 variance 1 random weights passed to the estimation for
        bootstrapping purposes
    rcond: regularization term when inverting matrices, see np.linalg.pinv
    interact, endogenous_interact, quad_interact: arguments passed to basis construction
    """

    # Get the instrument basis
    transformed_instrument = instrument_basis(npvec, deg, knots_inst, interact=interact)

    # If certain instruments are discrete, resulting in co-linear spline bases,
    # then remove them
    transformed_instrument = transformed_instrument[:, ~(transformed_instrument == 0).all(axis=0)]
    n = len(transformed_instrument)

    # Basis for endogenous variables
    b, dbd0 = generate_endogenous_basis(
        npvec,
        deg=max(1, deg - 1),
        pl=pl,
        knots=knots_endo,
        interact=endogenous_interact,
        quad_interact=quad_interact,
    )

    if bootstrap_weights is None:
        bootstrap_weights = np.ones(n)

    # Compute a TSLS estimate with endogenous basis and spline basis
    y = npvec["response"].flatten()
    invzz = np.linalg.pinv(transformed_instrument.T @ transformed_instrument, rcond=rcond)
    zx = transformed_instrument.T @ (bootstrap_weights[:, None] * b)
    zy = transformed_instrument.T @ (bootstrap_weights * y)
    coef_vector = (np.linalg.pinv(zx.T @ invzz @ zx, rcond=rcond) @ (zx.T @ invzz @ zy)).flatten()

    if se:
        # Estimation of standard errors
        if not pl:
            pzx = transformed_instrument @ invzz @ (transformed_instrument.T @ b)
            residual = y.flatten() - b @ coef_vector

            # Influence function for the coefficient vector
            ifs = (np.linalg.pinv(pzx.T @ pzx / n, rcond=rcond) @ (residual[:, None] * pzx).T).T

            # Influence function for the average derivative
            if_avg_deriv = (dbd0 - dbd0.mean(0, keepdims=True)) @ coef_vector + ifs @ dbd0.mean(0)

            se_ = if_avg_deriv.std() / (n**0.5)
        else:
            # Return the TSLS standard errors of the covariate of interest if
            # we are in partially linear setting
            residual = y.flatten() - b @ coef_vector
            zsigmaz = transformed_instrument.T @ ((residual**2)[:, None] * transformed_instrument)

            meat = zx.T @ invzz @ zsigmaz @ invzz @ zx
            bread = np.linalg.pinv(zx.T @ invzz @ zx, rcond=rcond)
            se_ = (bread @ meat @ bread)[1, 1] ** 0.5

    if full_return and not se:
        return (b, dbd0, transformed_instrument, coef_vector)
    elif full_return and se:
        return (b, dbd0, transformed_instrument, coef_vector, se_)

    if not pl:
        return (dbd0 @ coef_vector).mean() if not se else ((dbd0 @ coef_vector).mean(), se_)
    else:
        return coef_vector[1] if not se else (coef_vector[1], se_)


def optimally_weighted_spline_experiment(
    npvec,
    deg,
    knots_inst,
    knots_endo,
    pl=False,
    return_initial=False,
    bootstrap_weights=None,
    rcond=1e-15,
):
    # Get preliminary estimate from spl_experiment
    b, dbd0, tf_inst, coef_vec = spl_experiment(
        npvec,
        deg,
        knots_inst,
        knots_endo,
        pl=pl,
        full_return=True,
        bootstrap_weights=bootstrap_weights,
        rcond=rcond,
    )

    if bootstrap_weights is None:
        bootstrap_weights = np.ones(len(b))

    y = npvec["response"].flatten()
    residuals = y - b @ coef_vec
    initial_estimator = (dbd0 @ coef_vec).mean()
    zx = tf_inst.T @ (bootstrap_weights[:, None] * b)
    zy = tf_inst.T @ (bootstrap_weights * y)

    # Estimate the inverse variance weighting weight matrix
    weight_matrix = np.linalg.pinv(tf_inst.T @ ((residuals**2)[:, None] * tf_inst), rcond=rcond)

    # Optimally weighted coefficient
    optimal_coef = np.linalg.pinv(zx.T @ weight_matrix @ zx, rcond=rcond) @ (
        zx.T @ weight_matrix @ zy
    )

    # Estimate the covariance between the moments
    deriv = dbd0 @ optimal_coef
    optimal_residual = y - b @ coef_vec
    zz_inv_z = np.linalg.pinv(tf_inst.T @ tf_inst, rcond=rcond) @ tf_inst.T
    gamma_coef = zz_inv_z @ (
        (deriv - initial_estimator) * (optimal_residual - optimal_residual.mean()) ** 2
    )
    sigma_coef = zz_inv_z @ (optimal_residual - optimal_residual.mean()) ** 2

    # Project again to obtain the projection coefficient between the moments
    gamma = tf_inst @ zz_inv_z @ (tf_inst @ gamma_coef / (tf_inst @ sigma_coef).clip(min=0.1))

    # Return deriv - gamma * optimal_residual
    if not return_initial:
        return (deriv - gamma * optimal_residual).mean()
    else:
        return (deriv - gamma * optimal_residual).mean(), initial_estimator


def spl(x, r, knots):
    """Spline up to degree r-1, inclusive, with knots at the given locations."""
    if len(set(x)) == 2:
        return add_constant(x)
    poly = [x**k for k in range(0, r)]
    other = [np.maximum(x - k, 0) ** (r - 1) for k in knots]
    return np.array(poly + other).T


def dspl(x, r, knots):
    if len(set(x)) == 2:
        return np.c_[np.zeros((len(x), 1)), np.ones_like(x)]
    poly = [k * x ** (k - 1) for k in range(0, r)]
    poly[0] = np.zeros(len(x))
    other = [(x > k) * (r - 1) * (x - k) ** (r - 2) for k in knots]
    return np.array(poly + other).T


def d2spl(x, r, knots):
    if len(set(x)) == 2:
        return np.c_[np.zeros((len(x), 1)), np.zeros_like(x)]
    poly = [k * (k - 1) * x ** (k - 2) for k in range(0, r)]
    other = [(x > k) * ((r - 1) * (r - 2) * np.maximum((x - k), 0) ** (r - 3)) for k in knots]
    return np.array(poly + other).T
