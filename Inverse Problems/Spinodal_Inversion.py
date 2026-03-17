import mph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.special import legendre
import datetime
import pathlib


class CahnHilliardInversion:
    """
    Inverse identification of:
      - a = [a0..a4] : Legendre coefficients of the cheemical potential interaction term μ_h(c)
      - b = [b0..b4] : Legendre coefficients of the diffusivity D(c)

    Constraints:
      - Double-well: d²G/dc² ≤ -DELTA on c ∈ [0.15, 0.85]  (phase-separating μ)
      - Positivity : D(c; b)  ≥  EPSILON  for all c ∈ [0, 1]

    Objective:
      L(a, b) = (1/2) Σ_i mean_x ‖ε_sim(x,t_i) − ε_exp(x,t_i)‖²
    """

    INTERP_TIMES = np.array([0.001, 0.1, 0.15, 0.30, 1.0])
    N_TERMS      = 5
    N_CHECK      = 200
    EPSILON      = 1e-6   # diffusivity lower bound
    DELTA        = 1e-4   # spinodal concavity margin

    def __init__(
        self,
        model_path: str      = 'CH_M.mph',
        data_dir: str        = '.',
        chem_file: str       = 'chem.csv',
        log_file: str        = 'optimization_iterations.txt',
        hessian_file: str    = 'hessian_eigenvalues.txt',
        spinodal_range: tuple = (0.15, 0.85),
    ):
        self.model_path    = model_path
        self.data_dir      = pathlib.Path(data_dir)
        self.chem_file     = chem_file
        self.log_file      = log_file
        self.hessian_file  = hessian_file
        self.spinodal_lo   = spinodal_range[0]
        self.spinodal_hi   = spinodal_range[1]
        self.result        = None
        self._iter         = 0

        # ── Single COMSOL client ──────────────────────────────────────────────
        self._client = mph.start()
        self._model  = self._client.load(self.model_path)
        print(f"[CahnHilliardInversion] Loaded: {self.model_path}")
        print(f"  Parameters : {self._model.parameters()}")
        print(f"  Physics    : {self._model.physics()}")

        # ── Pre-build Legendre basis matrices at collocation points ───────────
        self._c_check  = np.linspace(0.0, 1.0, self.N_CHECK)
        self._xi_check = 2.0 * self._c_check - 1.0
        self._B        = self._build_basis(self._xi_check)          # (N_CHECK, 5)
        self._dBdc     = self._build_dbasis_dc(self._xi_check)      # (N_CHECK, 5)
        self._d2Bdc2   = self._build_d2basis_dc2(self._dBdc)        # (N_CHECK, 5)

        spinodal_mask      = ((self._c_check >= self.spinodal_lo) &
                              (self._c_check <= self.spinodal_hi))
        self._d2G_spinodal = self._d2Bdc2[spinodal_mask]            # (M, 5)

        # ── Load experimental data ────────────────────────────────────────────
        self.eps_exp = self._load_experimental_data()   # (eps11, eps22, eps33, eps12)
        self.n_timesteps = len(self.INTERP_TIMES)

    # ═══════════════════════════════════════════════════════════════════════════
    # Legendre basis construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_basis(self, xi: np.ndarray) -> np.ndarray:
        return np.stack([legendre(k)(xi) for k in range(self.N_TERMS)], axis=-1)

    def _build_dbasis_dc(self, xi: np.ndarray) -> np.ndarray:
        dBdc = np.zeros((len(xi), self.N_TERMS))
        for k in range(1, self.N_TERMS):
            Pk   = legendre(k)(xi)
            Pk_1 = legendre(k - 1)(xi)
            denom = 1.0 - xi**2
            safe  = np.abs(xi) < 1.0 - 1e-10
            dPdxi = np.where(
                safe,
                k * (Pk_1 - xi * Pk) / np.where(safe, denom, 1.0),
                k * (k + 1) / 2.0 * xi**(k - 1)
            )
            dBdc[:, k] = 2.0 * dPdxi
        return dBdc

    def _build_d2basis_dc2(self, dBdc: np.ndarray) -> np.ndarray:
        dc = 1.0 / (self.N_CHECK - 1)
        d2Bdc2 = np.zeros_like(dBdc)
        d2Bdc2[1:-1] = (dBdc[2:] - dBdc[:-2]) / (2 * dc)
        d2Bdc2[0]    = d2Bdc2[1]
        d2Bdc2[-1]   = d2Bdc2[-2]
        return d2Bdc2

    # ═══════════════════════════════════════════════════════════════════════════
    # Chemical potential / diffusivity
    # ═══════════════════════════════════════════════════════════════════════════

    def _interaction(self, c: np.ndarray, params: np.ndarray) -> np.ndarray:
        """μ_h(c; a) = Σ a_k P_k(2c−1)."""
        return self._build_basis(2.0 * c - 1.0) @ params

    def generate_chem_potential(self, a: np.ndarray, filename: str = None) -> pd.DataFrame:
        """
        μ(c) = μ_h(c; a) + log(c/(1−c)),  valid on [0.01, 0.99].
        Writes CSV for COMSOL interpolation function.
        """
        filename = filename or self.chem_file
        c_range  = np.linspace(-3.0, 3.0, 300)
        mu       = np.zeros_like(c_range)

        for i, c in enumerate(c_range):
            if 0.01 <= c <= 0.99:
                mu[i] = self._interaction(np.array([c]), a)[0] + np.log(c / (1.0 - c))

        data = pd.DataFrame({'concentration': c_range, 'chemical_potential': mu})
        data.to_csv(filename, index=False, float_format='%.8f')
        return data

    def diffusivity(self, c: np.ndarray, b: np.ndarray) -> np.ndarray:
        """D(c; b) = Σ b_k P_k(2c−1)."""
        return self._build_basis(2.0 * c - 1.0) @ b

    # ═══════════════════════════════════════════════════════════════════════════
    # COMSOL forward solve
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_forward(self, a: np.ndarray, b: np.ndarray):
        """Generate chem.csv, push to COMSOL, solve, return interpolated strains."""
        self.generate_chem_potential(a)

        # Update interpolation function in COMSOL
        try:
            func = self._model.java.func('int1')
            func.set('filename', self.chem_file)
            func.importData()
        except Exception as e:
            print(f"[Warning] Could not update int1: {e}")

        # Set diffusivity parameters
        b_names = ['b0', 'b1', 'b2', 'b3', 'b4']
        for name, val in zip(b_names, b):
            self._model.parameter(name, str(val))

        self._model.build()
        self._model.solve()

        time_values = np.array(self._model.evaluate('t')).ravel()
        eps11_full  = np.array(self._model.evaluate('solid.eXX'))
        eps22_full  = np.array(self._model.evaluate('solid.eYY'))
        eps33_full  = np.array(self._model.evaluate('solid.eZZ'))
        eps12_full  = np.array(self._model.evaluate('solid.eXY'))

        if (np.any(self.INTERP_TIMES < time_values.min()) or
                np.any(self.INTERP_TIMES > time_values.max())):
            raise ValueError("INTERP_TIMES outside solved time range.")

        n_pts = eps11_full.shape[1]
        def interp(eps):
            return np.array([
                np.interp(self.INTERP_TIMES, time_values, eps[:, i])
                for i in range(n_pts)
            ]).T   # (n_times, n_pts)

        return interp(eps11_full), interp(eps22_full), interp(eps33_full), interp(eps12_full)

    # ═══════════════════════════════════════════════════════════════════════════
    # Loss
    # ═══════════════════════════════════════════════════════════════════════════

    def _mse_loss(self, eps_sim: tuple, eps_exp: tuple) -> float:
        total = 0.0
        for comp_sim, comp_exp in zip(eps_sim, eps_exp):
            for i in range(self.n_timesteps):
                diff   = np.asarray(comp_sim[i]) - np.asarray(comp_exp[i])
                total += np.mean(diff ** 2)
        return 0.5 * total

    # ═══════════════════════════════════════════════════════════════════════════
    # Constraints
    # ═══════════════════════════════════════════════════════════════════════════

    def _c1_values(self, params: np.ndarray) -> np.ndarray:
        """d²G/dc² on spinodal — must be ≤ −DELTA."""
        return self._d2G_spinodal @ params[:5]

    def _c1_jac(self, params: np.ndarray) -> np.ndarray:
        J = np.zeros((self._d2G_spinodal.shape[0], 10))
        J[:, :5] = self._d2G_spinodal
        return J

    def _c2_values(self, params: np.ndarray) -> np.ndarray:
        """D(c; b) on [0,1] — must be ≥ EPSILON."""
        return self._B @ params[5:]

    def _c2_jac(self, params: np.ndarray) -> np.ndarray:
        J = np.zeros((self.N_CHECK, 10))
        J[:, 5:] = self._B
        return J

    def _build_constraints(self):
        return [
            NonlinearConstraint(self._c1_values, -np.inf, -self.DELTA, jac=self._c1_jac),
            NonlinearConstraint(self._c2_values,  self.EPSILON, np.inf, jac=self._c2_jac),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Objective
    # ═══════════════════════════════════════════════════════════════════════════

    def _objective(self, params: np.ndarray) -> float:
        a, b     = params[:5], params[5:]
        eps_sim  = self._run_forward(a, b)
        loss     = self._mse_loss(eps_sim, self.eps_exp)
        self._iter += 1

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param_str = ", ".join(f"{p:.6f}" for p in params)
        msg = (
            f"{timestamp} | iter {self._iter:04d} "
            f"| params: [{param_str}] "
            f"| D_min: {self._c2_values(params).min():.3e} "
            f"| d²G_max: {self._c1_values(params).max():.3e} "
            f"| MSE: {loss:.6e}"
        )
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
        return loss

    # ═══════════════════════════════════════════════════════════════════════════
    # Data loader
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_experimental_data(self):
        tags = [5, 25, 50, 75, 100]
        load = lambda comp, t: np.loadtxt(self.data_dir / f"{comp}_{t}.txt")
        return tuple(
            tuple(load(comp, t) for t in tags)
            for comp in ('eps11', 'eps22', 'eps33', 'eps12')
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        x0: np.ndarray = None,
        gtol: float    = 1e-8,
        xtol: float    = 1e-8,
        maxiter: int   = 5000,
        initial_tr_radius: float = 0.2,
    ):
        if x0 is None:
            x0 = np.array([-0.5, 0.0, -0.5, 0.0, 0.0,   # a: double-well via a2
                            0.5,  0.0,  0.0, 0.0, 0.0])  # b: flat positive D

        assert self._c2_values(x0).min() > self.EPSILON, \
            "x0 violates diffusivity positivity — increase x0[5]."
        assert self._c1_values(x0).max() < -self.DELTA, \
            "x0 does not satisfy double-well constraint — adjust a0/a2."

        self._iter  = 0
        self.result = minimize(
            self._objective, x0,
            method='trust-constr',
            constraints=self._build_constraints(),
            options={
                'verbose':           2,
                'gtol':              gtol,
                'xtol':              xtol,
                'barrier_tol':       1e-8,
                'initial_tr_radius': initial_tr_radius,
                'maxiter':           maxiter,
            }
        )

        print("\n── Result ─────────────────────────────────────")
        print(f"  Converged : {self.result.success}")
        print(f"  Message   : {self.result.message}")
        print(f"  a (G)     : {self.result.x[:5]}")
        print(f"  b (D)     : {self.result.x[5:]}")
        print(f"  D_min     : {self._c2_values(self.result.x).min():.3e}")
        print(f"  d²G_max   : {self._c1_values(self.result.x).max():.3e}")

        self._save_hessian()
        return self.result

    def _save_hessian(self):
        try:
            H = np.array(self.result.hess.todense())
        except AttributeError:
            H = np.array(self.result.hess)

        eigvals     = np.linalg.eigvalsh(H)
        eigvals_log = np.sign(eigvals) * np.log1p(np.abs(eigvals))

        with open(self.hessian_file, 'w') as f:
            f.write("Hessian at solution:\n")
            np.savetxt(f, H, fmt='%.8f')
            f.write("\nEigenvalues (log-scaled):\n")
            np.savetxt(f, eigvals_log.reshape(-1, 1), fmt='%.8f')
            f.write(f"\nAll eigenvalues positive (local minimum): "
                    f"{bool(np.all(eigvals > 0))}\n")

    def plot(self, params: np.ndarray = None):
        """Plot μ(c) and D(c) at given or optimized params."""
        if params is None:
            params = self.result.x
        a, b = params[:5], params[5:]
        c    = np.linspace(0.01, 0.99, 300)

        mu = self._interaction(c, a) + np.log(c / (1.0 - c))
        D  = self.diffusivity(c, b)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(c, mu, 'b', linewidth=2)
        axes[0].set(xlabel='c', ylabel='μ(c)', title='Chemical Potential')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(c, D, 'r', linewidth=2)
        axes[1].set(xlabel='c', ylabel='D(c)', title='Diffusivity')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('inversion_result.png', dpi=300)
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    inv    = CahnHilliardInversion(model_path='CH_M.mph', data_dir='.')
    result = inv.run()
    inv.plot()