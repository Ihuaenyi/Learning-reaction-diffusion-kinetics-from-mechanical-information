import mph
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.special import legendre
import datetime
import pathlib


class FickianInversion:
    """
    Physics-based inverse optimization for Fickian diffusivity identification.

    Identifies the Legendre polynomial expansion coefficients θ = [a0…a_{N-1}]
    of the concentration-dependent diffusivity D(c; θ) by minimizing the
    strain field residual:

        L(θ) = (1/2) Σ_{i=1}^{N} (1/|Ω|) Σ_x ‖ε(x,t_i;θ) − ε_data(x,t_i)‖²

    subject to D(c; θ) ≥ ε_pos for all c ∈ [0, 1]  (positivity constraint).
    """

    PARAM_NAMES = ["a0", "a1", "a2", "a3", "a4"]   # must match COMSOL parameter names

    def __init__(
        self,
        model_path: str,
        data_dir: str = ".",
        n_terms: int = 5,
        n_check: int = 200,
        epsilon_pos: float = 1e-6,
        timestep_indices: tuple = (1, 3, 10, 22, -1),
        log_file: str = "optimization_iterations.txt",
        hessian_file: str = "hessian_eigenvalues.txt",
    ):
        self.model_path      = model_path
        self.data_dir        = pathlib.Path(data_dir)
        self.n_terms         = n_terms
        self.n_check         = n_check
        self.epsilon_pos     = epsilon_pos
        self.timestep_indices = list(timestep_indices)
        self.n_timesteps     = len(timestep_indices)
        self.log_file        = log_file
        self.hessian_file    = hessian_file

        # ── Single COMSOL client (created once, reused across forward solves) ──
        self._client = mph.start()
        self._model  = self._client.load(self.model_path)
        print(f"[FickianInversion] Loaded model: {self.model_path}")
        print(f"  Parameters : {self._model.parameters()}")
        print(f"  Physics    : {self._model.physics()}")

        # ── Pre-build Legendre basis at collocation points (constant) ──────────
        self._x_check = np.linspace(0.0, 1.0, self.n_check)
        self._B_check = self._legendre_basis(self._x_check)   # (n_check, n_terms)

        # ── Load experimental data once ─────────────────────────────────────────
        self.eps_exp = self._load_experimental_data()   # tuple of 4 tuples

        # ── Iteration counter (instance-level, not global) ─────────────────────
        self._iter = 0
        self.result = None

    # ── Legendre basis ──────────────────────────────────────────────────────────

    def _legendre_basis(self, x: np.ndarray) -> np.ndarray:
        """B[j, k] = P_k(2*x[j] - 1).  Maps [0,1] → [-1,1] internally."""
        xi = 2.0 * x - 1.0
        return np.column_stack([legendre(k)(xi) for k in range(self.n_terms)])

    def diffusivity(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """D(x; θ) = Σ_k a_k * P_k(2x−1).  Output lives on [0,1]."""
        return self._legendre_basis(x) @ params

    # ── Positivity constraint ───────────────────────────────────────────────────

    def _positivity_values(self, params: np.ndarray) -> np.ndarray:
        return self._B_check @ params

    def _positivity_jac(self, params: np.ndarray) -> np.ndarray:
        return self._B_check   # constant — analytic, no finite differences

    def _build_positivity_constraint(self) -> NonlinearConstraint:
        return NonlinearConstraint(
            fun=self._positivity_values,
            lb=self.epsilon_pos,
            ub=np.inf,
            jac=self._positivity_jac,
        )

    # ── Forward COMSOL solve ────────────────────────────────────────────────────

    def _run_forward(self, params: np.ndarray):
        """Set Legendre coefficients in COMSOL, solve, return strain tuples."""
        for name, val in zip(self.PARAM_NAMES[:self.n_terms], params):
            self._model.parameter(name, str(val))

        self._model.build()
        self._model.solve()

        eps11_full = self._model.evaluate('solid.eXX')
        eps22_full = self._model.evaluate('solid.eYY')
        eps33_full = self._model.evaluate('solid.eZZ')
        eps12_full = self._model.evaluate('solid.eXY')

        idx = self.timestep_indices
        return (
            [eps11_full[i] for i in idx],
            [eps22_full[i] for i in idx],
            [eps33_full[i] for i in idx],
            [eps12_full[i] for i in idx],
        )

    # ── MSE loss ────────────────────────────────────────────────────────────────

    def _mse_loss(self, eps_sim: tuple) -> float:
        """
        L(θ) = (1/2) Σ_i mean_x ‖ε_sim(x,t_i) − ε_exp(x,t_i)‖²
        """
        eps11_sim, eps22_sim, eps33_sim, eps12_sim = eps_sim
        eps11_exp, eps22_exp, eps33_exp, eps12_exp = self.eps_exp
        total = 0.0

        for i in range(self.n_timesteps):
            e_sim = np.stack([
                np.asarray(eps11_sim[i]), np.asarray(eps22_sim[i]),
                np.asarray(eps33_sim[i]), np.asarray(eps12_sim[i]),
            ])   # (4, n_dof)
            e_exp = np.stack([
                np.asarray(eps11_exp[i]), np.asarray(eps22_exp[i]),
                np.asarray(eps33_exp[i]), np.asarray(eps12_exp[i]),
            ])
            total += np.mean(np.sum((e_sim - e_exp) ** 2, axis=0))

        return 0.5 * total

    # ── Objective ───────────────────────────────────────────────────────────────

    def _objective(self, params: np.ndarray) -> float:
        eps_sim = self._run_forward(params)
        loss    = self._mse_loss(eps_sim)
        self._iter += 1

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param_str = ", ".join(f"{p:.6f}" for p in params)
        d_min     = self._positivity_values(params).min()
        msg = (
            f"{timestamp} | iter {self._iter:04d} "
            f"| params: [{param_str}] "
            f"| D_min: {d_min:.3e} "
            f"| MSE: {loss:.6e}"
        )
        print(msg)
        with open(self.log_file, "a") as fh:
            fh.write(msg + "\n")

        return loss

    # ── Data loader ─────────────────────────────────────────────────────────────

    def _load_experimental_data(self):
        d    = self.data_dir
        tags = [5, 25, 50, 75, 100]
        load = lambda comp, t: np.loadtxt(d / f"{comp}_{t}.txt")
        return tuple(
            tuple(load(comp, t) for t in tags)
            for comp in ("eps11", "eps22", "eps33", "eps12")
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        x0: np.ndarray | None = None,
        gtol: float = 1e-10,
        xtol: float = 1e-8,
        maxiter: int = 5000,
        initial_tr_radius: float = 1.0,
    ):
        """
        Run the trust-constrained inversion.

        Parameters
        ----------
        x0 : initial guess for [a0…a_{N-1}].
             Defaults to a0=0.5, rest=0 (guaranteed feasible).
        """
        if x0 is None:
            x0 = np.zeros(self.n_terms)
            x0[0] = 0.5   # D(x) = 0.5 > epsilon_pos everywhere ✓

        assert self._positivity_values(x0).min() > self.epsilon_pos, (
            "Initial guess violates positivity constraint — increase x0[0]."
        )

        self._iter = 0   # reset counter for fresh run
        constraint = self._build_positivity_constraint()

        self.result = minimize(
            self._objective,
            x0,
            method="trust-constr",
            constraints=[constraint],
            options={
                "verbose":           2,
                "gtol":              gtol,
                "xtol":              xtol,
                "maxiter":           maxiter,
                "initial_tr_radius": initial_tr_radius,
            },
        )

        print("\n── Optimization complete ──────────────────────────")
        print(f"  Message  : {self.result.message}")
        print(f"  Converged: {self.result.success}")
        print(f"  Params   : {self.result.x}")
        print(f"  D_min    : {self._positivity_values(self.result.x).min():.3e}")

        self._save_hessian_diagnostics()
        return self.result

    def _save_hessian_diagnostics(self):
        try:
            H = np.array(self.result.hess.todense())
        except AttributeError:
            H = np.array(self.result.hess)

        eigvals = np.linalg.eigvalsh(H)
        eigvals_log = np.sign(eigvals) * np.log1p(np.abs(eigvals))

        with open(self.hessian_file, "w") as fh:
            fh.write("Hessian at solution:\n")
            np.savetxt(fh, H, fmt="%.8f")
            fh.write("\nEigenvalues (log-scaled):\n")
            np.savetxt(fh, eigvals_log.reshape(-1, 1), fmt="%.8f")
            fh.write(
                f"\nAll eigenvalues positive (local minimum): "
                f"{bool(np.all(eigvals > 0))}\n"
            )


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    inv = FickianInversion(
        model_path="Ficks_S.mph",
        data_dir=".",
    )
    result = inv.run()

    #Usage
    # inv = FickianInversion(model_path="Ficks_S.mph", data_dir=".")
    # result = inv.run()