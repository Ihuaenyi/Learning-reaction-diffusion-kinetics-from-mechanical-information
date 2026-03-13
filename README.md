# Learning Reaction-Diffusion Kinetics from Mechanical Information

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Computational%20Physics-blue)](https://doi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the computational codes accompanying the paper:

> **Learning reaction-diffusion kinetics from mechanical information**
> Royal C. Ihuaenyi, Hongbo Zhao, Ruqing Fang, Ruobing Bai, Martin Z. Bazant, Juner Zhu
> *Arxiv*, 2025

We present a PDE-constrained optimization framework for inferring chemical constitutive laws — including concentration-dependent diffusivity, chemical potential, exchange current density, and spatially heterogeneous reaction rates — exclusively from spatiotemporal mechanical strain field measurements. The framework operates without direct observation of the concentration field, solving a fundamentally cross-field inverse problem in which the chemical driving variable is itself hidden.

The methodology is validated on battery electrode materials across three physical regimes:
- Classical Fickian diffusion (graphite electrode)
- Spinodal decomposition with pattern formation (LFP electrode)
- Heterogeneous reaction kinetics from experimental STXM data (LFP nanoparticle)

---

## Repository Structure
```
├── inverse_problem/
│   ├── fickian_inversion.py         # Stage 1: Learn D(c) from strain fields
│   ├── spinodal_inversion.py        # Stage 2: Learn D(c) and mu(c) from strain fields
│   ├── c0_reconstruction.py         # GRF-based initial concentration field learning
│   ├── heterogeneous_inversion.py   # Stage 3: Learn j0(c), mu(c), k(x) jointly
│   └── optimization_utils.py        # Trust-region constrained optimization utilities
│
├── grf/
│   ├── grf_reconstruction.py        # GRF parameterization
|
│
├── data/
    ├── experimental/                # STXM experimental strain and concentration data
    ├── exx.csv                  # exx strain field (N_points x N_timesteps)
    ├── eyy.csv                  # eyy strain field
    ├── exy.csv                  # exy strain field
    └── c.csv                    # STXM concentration field (validation)


```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Core language |
| NumPy | ≥ 1.23 | Numerical arrays |
| SciPy | ≥ 1.10 | Optimization (`trust-constr`, `L-BFGS-B`, `linalg.eigh`) |
| pandas | ≥ 1.5 | Data I/O |
| matplotlib | ≥ 3.6 | Visualization |
| mph | ≥ 1.2 | Python–COMSOL interface |
| COMSOL Multiphysics | 6.2 | Forward PDE solver |

Install Python dependencies via:
```bash
pip install -r requirements.txt
```

---

## Forward Model

The coupled chemomechanical PDEs are solved using **COMSOL Multiphysics 6.2** with:
- Spatial discretization: quadratic Lagrange elements on unstructured triangular meshes
- Time integration: generalized-alpha method with adaptive time-stepping
- Linear solver: MUMPS direct solver
- Python–COMSOL interface: `mph` package

COMSOL `.mph` model files are provided in the `forward_model/` directory.

---

## Inverse Problem

The inverse problem minimizes the $L_2$ strain field discrepancy:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{N} \int_{\Omega} \left\| \boldsymbol{\varepsilon}(\mathbf{x}, t_i; \boldsymbol{\theta}) - \boldsymbol{\varepsilon}_{\text{data}}(\mathbf{x}, t_i) \right\|^2 d\mathbf{x}$$

Optimization is performed using SciPy's `trust-constr` algorithm with gradients computed via forward sensitivity analysis through the `mph` interface.

---

## Data

Experimental STXM imaging data of carbon-coated LiFePO$_4$ nanoparticles during 0.6C half-cycle discharge is available at:

> [https://doi.org/10.6084/m9.figshare.30037894.v1](https://doi.org/10.6084/m9.figshare.30037894.v1)

Place downloaded files in the `data/experimental/` directory before running the experimental inversion.

Synthetic ground truth data for the Fickian and spinodal test cases is included in `data/synthetic/`.

---

## Parametrization

Concentration-dependent constitutive functions are parametrized using Legendre polynomial expansions:

$$D(\bar{c}) = \sum_{i=1}^{M} a_i P_i(\bar{c}), \qquad \mu_h(\bar{c}) = \ln\frac{\bar{c}}{1-\bar{c}} + \sum_{i=1}^{M} b_i P_i(\bar{c}), \qquad j_0(\bar{c}) = \bar{c}(1-\bar{c})\sum_{i=1}^{M} c_i P_i(\bar{c})$$

The spatial heterogeneity field $k(\mathbf{x})$ is parametrized via a Karhunen-Loève Gaussian random field expansion with a low-dimensional spectral envelope, implemented in `grf/`.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{ihuaenyi2025learning,
  title   = {Learning reaction-diffusion kinetics from mechanical information},
  author  = {Ihuaenyi, Royal C. and Zhao, Hongbo and Fang, Ruqing and 
             Bai, Ruobing and Bazant, Martin Z. and Zhu, Juner},
  journal = {Arxiv},
  year    = {2025},
  doi     = {
https://doi.org/10.48550/arXiv.2508.17523
Focus to learn more
}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions regarding the code, please contact:

**Juner Zhu** — j.zhu@northeastern.edu  
Department of Mechanical and Industrial Engineering, Northeastern University
