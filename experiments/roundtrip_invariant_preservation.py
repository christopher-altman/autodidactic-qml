from __future__ import annotations
import argparse
import numpy as np
import torch

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from matrix_models.gauge_symmetries import invariants as gauge_invariants

from correspondence_maps.roundtrip import RoundTripMapper
from analysis.invariant_metrics import InvariantComparator
from analysis.spectral_diagnostics import SpectralDiagnostics

from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from autodidactic_protocols.mutual_information_drive import MutualInformationLearner
from autodidactic_protocols.renormalization_learning import RGLearner
from loop_dynamics_integration.retrocausal_feedback import TimeSymmetricGradientSmoother

def parse_args():
    p = argparse.ArgumentParser(description="Round-trip M -> model -> M_hat with invariant preservation metrics.")
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--sampler-steps", type=int, default=800)
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--g", type=float, default=0.12)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--T", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-mi", action="store_true")
    p.add_argument("--use-rg", action="store_true")
    p.add_argument("--use-time-symmetric", action="store_true",
                  help="Apply time-symmetric gradient smoothing steps.")
    return p.parse_args()

def pretty(d):
    return {k: (round(v, 6) if isinstance(v, (float, int)) else v) for k, v in d.items()}

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Sample a matrix from the matrix model distribution (minimal controlled cubic model)
    ens = HermitianEnsemble(dim=args.dim, scale=1.0, seed=args.seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=args.g)
    sampler = LangevinSampler(action=action, dt=args.dt, temperature=args.T, seed=args.seed)
    M = sampler.run(M0, steps=args.sampler_steps, rng=rng)

    inv0 = gauge_invariants(M, max_power=6)
    spec0 = SpectralDiagnostics().summarize(M)
    print("\n=== Source matrix M invariants ===")
    print(pretty(inv0))
    print("Spectral:", pretty(spec0))

    # 2) Build model from M
    rt = RoundTripMapper(width=args.width, seed=args.seed)
    core, model = rt.build_model(M)

    # 3) Round-trip before training
    Mhat0 = rt.extract_matrix(core)
    comp = InvariantComparator(max_power=6)
    pre = comp.compare(M, Mhat0)
    print("\n=== Round-trip error BEFORE training (M vs M_hat) ===")
    print(pretty(pre))

    # 4) Train with autodidactic protocols
    learner = SelfConsistencyLearner(lr=1e-3, steps_per_update=120, batch_size=128)
    mi = MutualInformationLearner(lr=1e-3, steps=120, batch_size=128, mi_weight=0.15)
    rg = RGLearner(lr=1e-3, steps=120, batch_size=128, factor=2)
    ts = TimeSymmetricGradientSmoother(horizon=6, sigma=2.0, lr=1e-3)

    def loss_fn(m, x):
        y = m(x)
        return torch.mean((y - x) ** 2)

    hist = []
    for ep in range(args.train_epochs):
        res = learner.update(model)
        res_mi = mi.update(model) if args.use_mi else {}
        res_rg = rg.update(model) if args.use_rg else {}
        res_ts = ts.step(model, loss_fn) if args.use_time_symmetric else {}

        Mhat = rt.extract_matrix(core)
        err = comp.compare(M, Mhat)
        err["epoch"] = float(ep)
        err["train_loss"] = float(res["loss"])
        if "mi_proxy" in res_mi:
            err["mi_proxy"] = float(res_mi["mi_proxy"])
        if "rg_loss" in res_rg:
            err["rg_loss"] = float(res_rg["rg_loss"])
        if "ts_loss_mean" in res_ts:
            err["ts_loss_mean"] = float(res_ts["ts_loss_mean"])
        hist.append(err)

        print(f"Epoch {ep:02d} | train_loss={res['loss']:.4f} | rel_tr_M2={err['rel_tr_M2']:.4f} | spec_l2_rel={err['spec_l2_rel']:.4f}")

    # 5) Final report
    MhatF = rt.extract_matrix(core)
    post = comp.compare(M, MhatF)
    print("\n=== Round-trip error AFTER training (M vs M_hat) ===")
    print(pretty(post))

    best = min(hist, key=lambda r: r["spec_l2_rel"])
    print("\n=== Best epoch by spec_l2_rel ===")
    print(pretty(best))

if __name__ == "__main__":
    main()
