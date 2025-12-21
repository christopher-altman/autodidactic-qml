from __future__ import annotations
import argparse
import numpy as np


def _require_tqdm():
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This experiment uses tqdm progress bars. Install it with:\n"
            "  pip install tqdm"
        ) from exc
    return tqdm

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction, QuarticAction
from matrix_models.sampler_langevin import LangevinSampler
from analysis.spectral_diagnostics import SpectralDiagnostics
from analysis.phase_transition_metrics import PhaseTransitionMetrics

def parse_args():
    p = argparse.ArgumentParser(description="Scan coupling parameter g for minimal controlled phase-like transitions.")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--T", type=float, default=1e-2)
    p.add_argument("--g-min", type=float, default=-0.25)
    p.add_argument("--g-max", type=float, default=0.25)
    p.add_argument("--points", type=int, default=17)
    p.add_argument("--stabilize", action="store_true", help="Add quartic stabilizer to avoid blowups.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    tqdm = _require_tqdm()
    rng = np.random.default_rng(args.seed)
    ens = HermitianEnsemble(dim=args.dim, scale=1.0, seed=args.seed)
    diag = SpectralDiagnostics()
    ptm = PhaseTransitionMetrics()

    gs = np.linspace(args.g_min, args.g_max, args.points)
    radii, stds, actions = [], [], []

    for g in tqdm(gs, desc="scan g", unit="g"):
        M0 = ens.sample(rng=rng)
        if args.stabilize:
            # Use cubic + quartic stabilizer by composing gradients
            cubic = CubicAction(g=float(g))
            quart = QuarticAction(a=0.0, b=0.02)
            class Combo:
                def action(self, M): return cubic.action(M) + quart.action(M)
                def grad(self, M): return cubic.grad(M) + quart.grad(M)
            act = Combo()
        else:
            act = CubicAction(g=float(g))

        sampler = LangevinSampler(action=act, dt=args.dt, temperature=args.T, seed=args.seed)
        M = sampler.run(M0, steps=args.steps, rng=rng)

        s = diag.summarize(M)
        radii.append(s["spectral_radius"])
        stds.append(s["eig_std"])
        actions.append(act.action(M) if hasattr(act, "action") else 0.0)

    radii = np.array(radii); stds = np.array(stds); actions = np.array(actions)
    m1 = ptm.metric(radii); m2 = ptm.metric(stds)
    print("g grid:", np.round(gs, 3))
    print("spectral_radius:", np.round(radii, 3))
    print("eig_std:", np.round(stds, 3))
    print("Heuristics:")
    print("  radius jump/kurtosis:", {k: round(v,4) for k,v in m1.items()})
    print("  std    jump/kurtosis:", {k: round(v,4) for k,v in m2.items()})

if __name__ == "__main__":
    main()
