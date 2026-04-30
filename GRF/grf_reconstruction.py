"""
c0_inverse_pipeline.py
======================
Inverse optimization pipeline to infer c0(x) from strain fields.

Stages:
  1. Correlation Structure Initialization (KL/GRF basis)
  2. Field Reconstruction (spectral envelope + spatial modulation)
  3. L-BFGS-B Optimization against COMSOL strain output

Usage:
    grf_reconstruction.py
"""


# Imports

import mph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.optimize import minimize


# COMSOL client (global)

client = mph.start()


# Coordinate / strain utilities

def coord_extract(model, t=-1):
    x = model.evaluate('X')
    y = model.evaluate('Y')
    return x[t], y[t]


def standardize_coords(x_coords, y_coords):
    coords        = np.column_stack([x_coords, y_coords])
    mu            = coords.mean(axis=0)
    sigma         = coords.std(axis=0, ddof=1)
    coords_scaled = (coords - mu) / sigma
    return coords_scaled, mu, sigma


def standardize_strain(strain):
    """Zero mean, unit variance — preserves amplitude structure in alpha0."""
    mu    = float(np.mean(strain))
    sigma = float(np.std(strain, ddof=1))
    out   = (strain - mu) / (sigma + 1e-12)
    print(f"  Strain mean={mu:.4e}  std={sigma:.4e}  "
          f"stdized range=[{out.min():.3f}, {out.max():.3f}]")
    return out, mu, sigma


# COMSOL model runner

def run_model_with_parameters(**kwargs):
    """Load, configure, build, solve COMSOL model; return strain fields at t=0."""
    model = client.load('c0_model.mph')

    try:
        func = model.java.func('int3')
        func.set('filename', 'c0_optimized.csv')
        func.importData()
        print("Updated interpolation function 'int3' with file: c0_optimized.csv")
    except Exception as e:
        print(f"Failed to update 'int3': {e}")

    for param_name, param_value in kwargs.items():
        model.parameter(param_name, str(param_value))
        print(f"Set {param_name} = {param_value}")

    model.build()
    model.solve()

    time_values = np.array(model.evaluate('t')).ravel()
    c           = np.array(model.evaluate('c'))
    exx         = np.array(model.evaluate('solid.eXX'))
    eyy         = np.array(model.evaluate('solid.eYY'))
    exy         = np.array(model.evaluate('solid.eXY'))

    interp_times = np.array([0.0])
    if (np.any(interp_times < time_values.min()) or
            np.any(interp_times > time_values.max())):
        raise ValueError("Interpolation times must be within the range of time_values")

    def interp_field(field):
        return np.array([
            np.interp(interp_times, time_values, field[:, i])
            for i in range(field.shape[1])
        ]).T

    return interp_field(c), interp_field(exx), interp_field(eyy), interp_field(exy)


# Strain-field MSE loss

def mse_loss_strain(exx_0, eyy_0, exy_0,
                    exx_0_exp, eyy_0_exp, exy_0_exp,
                    alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    strains_sim = [exx_0,     eyy_0,     exy_0    ]
    strains_exp = [exx_0_exp, eyy_0_exp, exy_0_exp]
    weights     = [alpha,     beta,      gamma    ]
    labels      = ['exx',     'eyy',     'exy'    ]

    mse_terms = {
        label: w * np.mean((np.array(s) - np.array(e))**2)
        for label, w, s, e in zip(labels, weights, strains_sim, strains_exp)
    }

    mse_strain = sum(mse_terms.values())

    avg_constraint = delta * np.sum([
        (np.mean(s) - np.mean(e))**2
        for s, e in zip(strains_sim, strains_exp)
    ])

    return mse_strain + avg_constraint


# Stage 1: Correlation Structure Initialization

def compute_initial_correlation(coords_scaled, n_neighbors=6, epsilon=1e-6):
    nbrs         = NearestNeighbors(n_neighbors=n_neighbors).fit(coords_scaled)
    nn_dist, _   = nbrs.kneighbors(coords_scaled)
    length_scale = float(np.mean(nn_dist[:, -1]))
    distances    = cdist(coords_scaled, coords_scaled)
    C0           = np.exp(-distances / length_scale)
    C0          += epsilon * np.eye(len(C0))
    return C0, length_scale


def compute_eigendecomposition(C0, n_modes=500, theta=0.99):
    evals_all, evecs_all = eigh(C0)
    idx       = np.argsort(evals_all)[::-1]
    evals_all = evals_all[idx]
    evecs_all = evecs_all[:, idx]
    cumvar    = np.cumsum(evals_all) / np.sum(evals_all)
    Q         = int(np.searchsorted(cumvar, theta)) + 1
    Q         = min(Q, n_modes, len(evals_all))
    print(f"  Modes available : {len(evals_all)}")
    print(f"  Retained Q      : {Q}  (theta={theta}, "
          f"variance captured={cumvar[Q-1]:.4f})")
    print(f"  Top 5 evals     : {evals_all[:5].round(4)}")
    print(f"  Spectrum shape  : log10(lambda) range "
          f"[{np.log10(evals_all[0]):.2f}, "
          f"{np.log10(evals_all[Q-1]):.2f}]")
    return evals_all[:Q], evecs_all[:, :Q], Q


def compute_baseline_weights(C0, strain_std, epsilon=1e-6):
    A      = C0 + epsilon * np.eye(len(C0))
    alpha0 = np.linalg.solve(A, strain_std)
    print(f"  alpha0 range : [{alpha0.min():.4e}, {alpha0.max():.4e}]  "
          f"mean={np.mean(alpha0):.4e}  std={np.std(alpha0):.4e}")
    return alpha0


def save_stage1_outputs(eigenvalues, eigenvectors, alpha0,
                        coords_scaled, length_scale, Q, mu, sigma,
                        strain_mu, strain_sigma, prefix="stage1"):
    pd.DataFrame({
        'mode'           : np.arange(1, Q + 1),
        'eigenvalue'     : eigenvalues,
        'norm_eigenvalue': eigenvalues / eigenvalues[0],
    }).to_csv(f"{prefix}_eigenvalues.csv", index=False)
    print(f"  Saved {prefix}_eigenvalues.csv       ({Q} modes)")

    pd.DataFrame(eigenvectors,
                 columns=[f"phi_{k+1}" for k in range(Q)]
    ).to_csv(f"{prefix}_eigenvectors.csv", index=False)
    print(f"  Saved {prefix}_eigenvectors.csv      "
          f"({eigenvectors.shape[0]} x {Q})")

    pd.DataFrame({'alpha0': alpha0}
    ).to_csv(f"{prefix}_alpha0.csv", index=False)
    print(f"  Saved {prefix}_alpha0.csv            ({len(alpha0)} weights)")

    pd.DataFrame({'x_scaled': coords_scaled[:, 0],
                  'y_scaled': coords_scaled[:, 1]}
    ).to_csv(f"{prefix}_coords_scaled.csv", index=False)
    print(f"  Saved {prefix}_coords_scaled.csv     ({len(coords_scaled)} pts)")

    pd.DataFrame({
        'length_scale' : [length_scale],
        'Q'            : [Q],
        'mu_x'         : [mu[0]],    'mu_y'    : [mu[1]],
        'sigma_x'      : [sigma[0]], 'sigma_y' : [sigma[1]],
        'strain_mu'    : [strain_mu],
        'strain_sigma' : [strain_sigma],
    }).to_csv(f"{prefix}_meta.csv", index=False)
    print(f"  Saved {prefix}_meta.csv")


def load_stage1_outputs(prefix="stage1"):
    df_eig    = pd.read_csv(f"{prefix}_eigenvalues.csv")
    df_phi    = pd.read_csv(f"{prefix}_eigenvectors.csv")
    df_alpha  = pd.read_csv(f"{prefix}_alpha0.csv")
    df_coords = pd.read_csv(f"{prefix}_coords_scaled.csv")
    m         = pd.read_csv(f"{prefix}_meta.csv").iloc[0]
    return {
        'eigenvalues'  : df_eig['eigenvalue'].values,
        'eigenvectors' : df_phi.values,
        'alpha0'       : df_alpha['alpha0'].values,
        'coords_scaled': df_coords[['x_scaled', 'y_scaled']].values,
        'length_scale' : float(m['length_scale']),
        'Q'            : int(m['Q']),
        'mu'           : np.array([m['mu_x'],    m['mu_y']]),
        'sigma'        : np.array([m['sigma_x'], m['sigma_y']]),
        'strain_mu'    : float(m['strain_mu']),
        'strain_sigma' : float(m['strain_sigma']),
    }


def run_stage1(x_coords, y_coords, strain,
               n_modes=500, theta=0.99, n_neighbors=6,
               epsilon_corr=1e-6, epsilon_weights=1e-6,
               prefix="stage1"):
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    strain   = np.asarray(strain,   dtype=float)

    print("=" * 60)
    print("STAGE 1: CORRELATION STRUCTURE INITIALIZATION")
    print("=" * 60)
    print(f"  N={len(strain)}  raw strain range "
          f"[{strain.min():.4e}, {strain.max():.4e}]")

    print("\n[1/5] Standardizing coordinates...")
    coords_scaled, mu, sigma = standardize_coords(x_coords, y_coords)

    print("\n[2/5] Standardizing strain...")
    strain_std, strain_mu, strain_sigma = standardize_strain(strain)

    print("\n[3/5] Building correlation matrix C(0)...")
    C0, length_scale = compute_initial_correlation(
        coords_scaled, n_neighbors=n_neighbors, epsilon=epsilon_corr)
    print(f"  Length scale l(0)={length_scale:.6f}")

    print("\n[4/5] Eigendecomposition...")
    eigenvalues, eigenvectors, Q = compute_eigendecomposition(
        C0, n_modes=n_modes, theta=theta)

    print("\n[5/5] Baseline weights alpha(0)...")
    alpha0 = compute_baseline_weights(C0, strain_std, epsilon=epsilon_weights)

    print("\nSaving...")
    save_stage1_outputs(eigenvalues, eigenvectors, alpha0,
                        coords_scaled, length_scale, Q, mu, sigma,
                        strain_mu, strain_sigma, prefix=prefix)
    print("\nStage 1 complete.")

    return {
        'eigenvalues'  : eigenvalues,  'eigenvectors' : eigenvectors,
        'alpha0'       : alpha0,       'coords_scaled': coords_scaled,
        'length_scale' : length_scale, 'Q'            : Q,
        'mu'           : mu,           'sigma'        : sigma,
        'strain_mu'    : strain_mu,    'strain_sigma' : strain_sigma,
    }


# Stage 2: Spectral envelope + spatial modulation (field reconstruction)

def spectral_envelope(lam0, a, gamma, b):
    """
    Apply smooth multiplicative envelope to Stage 1 eigenvalues.

        lambda_k = lambda_k^(0) * (a * exp(-gamma * k_norm) + b)

    Parameters
    ----------
    lam0  : (Q,)  Stage 1 eigenvalues (fixed, descending order)
    a     : float Amplitude of exponential decay component (> 0)
    gamma : float Decay rate across mode index              (>= 0)
    b     : float Floor scaling                             (> 0)
    """
    Q      = len(lam0)
    k_norm = np.arange(Q) / Q
    f      = a * np.exp(-gamma * k_norm) + b
    return lam0 * np.abs(f)


def reconstruct_c0(a, gamma, b, s, bias, c1, c2, c3,
                   lam0, eigenvectors, alpha0, coords_scaled,
                   coords_true=None, output_csv=None):
    """
    Reconstruct c0(x) from 3 spectral + 5 spatial parameters (8 total).

    Parameters
    ----------
    a, gamma, b    : float  Spectral envelope parameters.
    s, bias        : float  Global amplitude scale and offset for alpha.
    c1, c2, c3     : float  Spatial modulation: radial, x-tilt, y-tilt.
    lam0           : (Q,)   Stage 1 eigenvalues (fixed).
    eigenvectors   : (N, Q) Fixed Phi from Stage 1.
    alpha0         : (N,)   Fixed baseline weights from Stage 1.
    coords_scaled  : (N, 2) Standardized coordinates.
    coords_true    : (N, 2) Original coordinates (for CSV output only).
    output_csv     : str    Path to save output field (no header).

    Returns
    -------
    c0_act : (N,) reconstructed field (shift-activated to be strictly positive).
    """
    lam        = spectral_envelope(lam0, a, gamma, b)
    r          = np.linalg.norm(coords_scaled, axis=1)
    x          = coords_scaled[:, 0]
    y          = coords_scaled[:, 1]
    modulation = 1.0 + c1 * r + c2 * x + c3 * y
    alpha      = (s * alpha0 + bias) * modulation
    scores     = eigenvectors.T @ alpha        # (Q,)
    c0         = eigenvectors @ (lam * scores) # (N,)
    c0_act     = c0 - c0.min() + 1e-6

    if output_csv is not None:
        assert coords_true is not None
        pd.DataFrame({
            'x' : coords_true[:, 0],
            'y' : coords_true[:, 1],
            'c0': c0_act,
        }).to_csv(output_csv, index=False, header=False)

    return c0_act


# Stage 3: trust-constr optimization

from scipy.optimize import minimize, Bounds


def pack(a, gamma, b, s, bias, c1, c2, c3):
    """Map physical parameters to log-space optimization vector."""
    return np.array([np.log(a), gamma, np.log(b),
                     np.log(s), bias, c1, c2, c3])


def unpack(p):
    """Map log-space optimization vector back to physical parameters."""
    a     = np.exp(p[0])
    gamma = p[1]
    b     = np.exp(p[2])
    s     = np.exp(p[3])
    bias  = p[4]
    c1, c2, c3 = p[5], p[6], p[7]
    return a, gamma, b, s, bias, c1, c2, c3


def run_stage3(stage1, initial_exx, initial_eyy, initial_exy,
               n_restarts=50, prefix="stage1"):
    """
    Run trust-constr optimization against experimental strain fields.

    Parameters
    ----------
    stage1       : dict   Output of run_stage1 (or load_stage1_outputs).
    initial_exx/eyy/exy : (N,) experimental strain fields at t=0.
    n_restarts   : int    Maximum number of trust-constr restarts.
    prefix       : str    File prefix (used only for reporting).
    """
    _lam0          = stage1['eigenvalues']
    _eigenvectors  = stage1['eigenvectors']
    _alpha0        = stage1['alpha0']
    _coords_scaled = stage1['coords_scaled']
    _mu            = stage1['mu']
    _sigma         = stage1['sigma']
    _coords_true   = _coords_scaled * _sigma + _mu
    Q              = stage1['Q']

    print(f"\nStage 1 loaded : {len(_alpha0)} points, {Q} modes")
    print(f"Parameters     : 3 spectral (a, gamma, b) + 5 spatial = 8 total")
    print(f"alpha0 range   : [{_alpha0.min():.4e}, {_alpha0.max():.4e}]")
    print(f"Experimental strains loaded: {len(initial_exx)} points")

    # --- Analytical warm start for s and bias
    print("\nComputing blind warm start for s and bias...")
    _base_scores  = _eigenvectors.T @ _alpha0
    _base_field   = _eigenvectors @ (_lam0 * _base_scores)
    base_mean     = float(np.mean(_base_field))
    base_std      = float(np.std(_base_field, ddof=1)) + 1e-12
    s_init        = float(np.clip(1.0 / base_std,        0.01, 20.0))
    bias_init     = float(np.clip(-base_mean / base_std, -10.0, 10.0))
    c0_init_check = s_init * _base_field + bias_init
    print(f"  base_field : mean={base_mean:.4f}  std={base_std:.4f}")
    print(f"  s*={s_init:.6f}   bias*={bias_init:.6f}")
    print(f"  Initial reconstruction: mean={c0_init_check.mean():.4f}  "
          f"std={c0_init_check.std():.4f}  (should be ~0, ~1)")

    # --- Bounds (trust-constr uses scipy.optimize.Bounds, not list of tuples)
    lb = np.array([np.log(0.001),  0.0, np.log(0.001),
                   np.log(0.001), -10.0, -3.0, -3.0, -3.0])
    ub = np.array([np.log(50.0),  10.0, np.log(10.0),
                   np.log(50.0),  10.0,  3.0,  3.0,  3.0])
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=True)

    # --- State
    iter_count = [0]
    best_loss  = [np.inf]
    mse_log    = open("mse_history.txt", "w")
    mse_log.write("iter,mse,a,gamma,b,s,bias,c1,c2,c3\n")

    # --- Objective
    def objective(params):
        a, gamma, b, s, bias, c1, c2, c3 = unpack(params)

        reconstruct_c0(a, gamma, b, s, bias, c1, c2, c3,
                       lam0=_lam0, eigenvectors=_eigenvectors,
                       alpha0=_alpha0, coords_scaled=_coords_scaled,
                       coords_true=_coords_true,
                       output_csv="c0_optimized.csv")

        try:
            _, exx_interp, eyy_interp, exy_interp = run_model_with_parameters()
        except Exception as e:
            print(f"\n  COMSOL run failed: {e}  — returning large loss")
            return 1e10

        exx_0 = exx_interp[0]
        eyy_0 = eyy_interp[0]
        exy_0 = exy_interp[0]

        loss = mse_loss_strain(exx_0, eyy_0, exy_0,
                               initial_exx, initial_eyy, initial_exy,
                               alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

        iter_count[0] += 1

        if loss < best_loss[0]:
            best_loss[0] = loss
            mse_log.write(
                f"{iter_count[0]},{loss:.10e},"
                f"{a:.6f},{gamma:.6f},{b:.6f},"
                f"{s:.6f},{bias:.6f},{c1:.6f},{c2:.6f},{c3:.6f}\n")
            mse_log.flush()
            pd.DataFrame(params).to_csv(
                "c0_optimized_params.csv", index=False, header=False)

        print(f"\r  iter {iter_count[0]:5d}  |  "
              f"loss={loss:.6e}  best={best_loss[0]:.6e}  |  "
              f"a={a:.3f}  g={gamma:.3f}  b={b:.3f}  "
              f"s={s:.3f}  bias={bias:.4f}  "
              f"c1={c1:.3f}  c2={c2:.3f}  c3={c3:.3f}",
              end="", flush=True)

        return loss

    # --- Initial starting point (warm or cold)
    try:
        prev   = pd.read_csv("c0_optimized_params.csv",
                             header=None).values.flatten()
        assert len(prev) == 8, (
            f"Param file has {len(prev)} params but expected 8. "
            "Delete c0_optimized_params.csv and rerun.")
        x_init = prev
        print(f"\n  Warm start: loaded from c0_optimized_params.csv")
    except FileNotFoundError:
        x_init = pack(1.0, 0.0, 1e-4, s_init, bias_init, 0.0, 0.0, 0.0)
        print(f"\n  Cold start: a=1, gamma=0, s={s_init:.4f}, "
              f"bias={bias_init:.4f}")

    # --- Auto-restarting trust-constr
    #
    # trust-constr uses a barrier/augmented-Lagrangian interior-point method.
    # Convergence criterion: optimality (grad of Lagrangian) < gtol AND
    # constraint violation < gtol. finite_diff_jac_options controls the
    # step size for numerical Jacobian estimation — tightening eps improves
    # gradient accuracy at the cost of 2 extra function evaluations per step.
    #
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        print(f"\nRestart {restart+1}/{n_restarts}  best={best_loss[0]:.6e}")
        iter_count[0] = 0

        result = minimize(
            objective,
            x_init,
            method  = 'trust-constr',
            bounds  = bounds,
            options = {
                'maxiter'                  : 10000,
                'gtol'                     : 1e-9,   # gradient-of-Lagrangian tolerance
                'xtol'                     : 1e-12,  # step-size tolerance
                'barrier_tol'              : 1e-9,   # interior-point barrier tolerance
                'initial_tr_radius'        : 1.0,    # initial trust-region radius
                'initial_constr_penalty'   : 1.0,    # initial augmented-Lagrangian penalty
                'verbose'                  : 0,      # suppress scipy internal output
                'finite_diff_jac_options'  : {'rel_step': 1e-5},
            }
        )

        # trust-constr sets result.status:
        #   0 = max iterations reached
        #   1 = gradient norm < gtol
        #   2 = trust-region radius < xtol
        #   3 = objective change < gtol (interpreted as converged)
        converged = result.status in (1, 2, 3)
        print(f"\n  loss={result.fun:.6e}  status={result.status}  "
              f"converged={converged}  "
              f"optimality={result.optimality:.3e}  "
              f"constr_nfev={result.nfev}")

        if converged:
            print(f"  Converged at restart {restart+1}")
            break

        # Perturb spectral params; keep best spatial fixed
        try:
            best_vec      = pd.read_csv("c0_optimized_params.csv",
                                        header=None).values.flatten()
        except FileNotFoundError:
            best_vec = x_init

        spectral_best = best_vec[:3]
        spatial_best  = best_vec[3:]
        noise         = rng.uniform(-0.3, 0.3, 3)
        new_spectral  = np.clip(spectral_best + noise, lb[:3], ub[:3])
        x_init        = np.concatenate([new_spectral, spatial_best])
        print(f"  Perturbing spectral params, keeping spatial fixed")

    else:
        print(f"\n  Did not converge after {n_restarts} restarts")

    mse_log.close()

    # --- Final report
    best_vec = pd.read_csv("c0_optimized_params.csv",
                           header=None).values.flatten()
    a_opt, gm_opt, b_opt, s_opt, bs_opt, c1_opt, c2_opt, c3_opt = unpack(best_vec)
    lam_opt  = spectral_envelope(_lam0, a_opt, gm_opt, b_opt)

    print(f"\n{'='*55}")
    print(f"Final loss : {best_loss[0]:.6e}")
    print(f"\nLearned spectral envelope:")
    print(f"  a     = {a_opt:.6f}  gamma = {gm_opt:.6f}  b = {b_opt:.6f}")
    print(f"\nLearned spatial parameters:")
    print(f"  s={s_opt:.6f}  bias={bs_opt:.6f}  "
          f"c1={c1_opt:.6f}  c2={c2_opt:.6f}  c3={c3_opt:.6f}")
    print(f"\nSpectral envelope effect:")
    for k in [0, 1, 4, 9, 24, 49, min(99, Q-1), Q-1]:
        print(f"  mode {k+1:4d}: lam0={_lam0[k]:.4f}  "
              f"lam_opt={lam_opt[k]:.4f}  ratio={lam_opt[k]/_lam0[k]:.4f}")

    c0_final = reconstruct_c0(
        a_opt, gm_opt, b_opt, s_opt, bs_opt, c1_opt, c2_opt, c3_opt,
        lam0=_lam0, eigenvectors=_eigenvectors, alpha0=_alpha0,
        coords_scaled=_coords_scaled, coords_true=_coords_true,
        output_csv="c0_optimized.csv")

    _, exx_final, eyy_final, exy_final = run_model_with_parameters()
    exx_0 = exx_final[0]
    eyy_0 = eyy_final[0]
    exy_0 = exy_final[0]

    residual_exx = exx_0 - initial_exx
    residual_eyy = eyy_0 - initial_eyy
    residual_exy = exy_0 - initial_exy

    print(f"\nFinal field saved to c0_optimized.csv")
    print(f"  MSE exx : {np.mean(residual_exx**2):.6e}")
    print(f"  MSE eyy : {np.mean(residual_eyy**2):.6e}")
    print(f"  MSE exy : {np.mean(residual_exy**2):.6e}")
    print(f"  R2  exx : {1 - np.var(residual_exx)/np.var(initial_exx):.6f}")
    print(f"  R2  eyy : {1 - np.var(residual_eyy)/np.var(initial_eyy):.6f}")
    print(f"  R2  exy : {1 - np.var(residual_exy)/np.var(initial_exy):.6f}")

    return c0_final, best_vec


# Entry point

if __name__ == "__main__":

    # --- Extract coordinates from COMSOL model
    _model    = client.load('c0_model.mph')
    x_coords, y_coords = coord_extract(_model, t=-1)

    # --- Load experimental strain data
    data_xx     = np.loadtxt('exx_data.csv', delimiter=',')
    data_yy     = np.loadtxt('eyy_data.csv', delimiter=',')
    data_xy     = np.loadtxt('exy_data.csv', delimiter=',')
    initial_exx = data_xx[:, 2]
    initial_eyy = data_yy[:, 2]
    initial_exy = data_xy[:, 2]
    ave_strain  = (initial_exx + initial_eyy + initial_exy) / 3

    # --- Stage 1: Build correlation basis
    stage1 = run_stage1(
        x_coords    = x_coords,
        y_coords    = y_coords,
        strain      = ave_strain,
        n_modes     = 500,
        theta       = 0.95,
        n_neighbors = 6,
        prefix      = "stage1"
    )

    # --- Stage 3: Optimize c0 against COMSOL strain output
    c0_final, best_params = run_stage3(
        stage1      = stage1,
        initial_exx = initial_exx,
        initial_eyy = initial_eyy,
        initial_exy = initial_exy,
        n_restarts  = 50,
        prefix      = "stage1"
    )