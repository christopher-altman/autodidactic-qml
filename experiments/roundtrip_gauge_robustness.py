from __future__ import annotations
import argparse
import numpy as np
import torch

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from matrix_models.gauge_symmetries import invariants as gauge_invariants

from analysis.gauge_tools import random_orthogonal, gauge_transform
from analysis.invariant_metrics import InvariantComparator, fro_rel
from analysis.spectral_diagnostics import SpectralDiagnostics

from correspondence_maps.roundtrip import RoundTripMapper

from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from autodidactic_protocols.mutual_information_drive import MutualInformationLearner
from autodidactic_protocols.renormalization_learning import RGLearner
from loop_dynamics_integration.retrocausal_feedback import TimeSymmetricGradientSmoother

def parse_args():
    p = argparse.ArgumentParser(
        description="Gauge-robust round-trip: M' = U M U^T, build models, extract M_hat and M_hat', compare invariants."
    )
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
    p.add_argument("--use-time-symmetric", action="store_true")
    return p.parse_args()

def pretty(d):
    return {k: (round(v, 6) if isinstance(v, (float, int)) else v) for k, v in d.items()}

def train_model(model, seed: int, use_mi: bool, use_rg: bool, use_ts: bool, epochs: int):
    torch.manual_seed(seed)
    learner = SelfConsistencyLearner(lr=1e-3, steps_per_update=120, batch_size=128)
    mi = MutualInformationLearner(lr=1e-3, steps=120, batch_size=128, mi_weight=0.15)
    rg = RGLearner(lr=1e-3, steps=120, batch_size=128, factor=2)
    ts = TimeSymmetricGradientSmoother(horizon=6, sigma=2.0, lr=1e-3)

    def loss_fn(m, x):
        y = m(x)
        return torch.mean((y - x) ** 2)

    for ep in range(epochs):
        res = learner.update(model)
        if use_mi:
            mi.update(model)
        if use_rg:
            rg.update(model)
        if use_ts:
            ts.step(model, loss_fn)
    return

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1) Sample a symmetric matrix M from the minimal controlled model cubic matrix model.
    ens = HermitianEnsemble(dim=args.dim, scale=1.0, seed=args.seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=args.g)
    sampler = LangevinSampler(action=action, dt=args.dt, temperature=args.T, seed=args.seed)
    M = sampler.run(M0, steps=args.sampler_steps, rng=rng)

    # 2) Gauge transform: M' = U M U^T
    U = random_orthogonal(args.dim, rng=rng)
    Mprime = gauge_transform(M, U)

    # Sanity: invariants(M) == invariants(M')
    comp = InvariantComparator(max_power=6)
    sanity = comp.compare(M, Mprime)

    print("\n=== Sanity: invariants(M) vs invariants(M') (should be ~0) ===")
    print(pretty(sanity))

    invM = gauge_invariants(M, max_power=6)
    invMp = gauge_invariants(Mprime, max_power=6)
    print("\nRaw gauge invariants (matrix_models.gauge_symmetries):")
    print("M :", pretty(invM))
    print("M':", pretty(invMp))

    # 3) Build two models: one from M and one from M'
    rt = RoundTripMapper(width=args.width, seed=args.seed)
    core, model = rt.build_model(M)
    core_p, model_p = rt.build_model(Mprime)

    # 4) Extract before training
    Mhat0 = rt.extract_matrix(core)
    Mhatp0 = rt.extract_matrix(core_p)

    pre_M_Mhat = comp.compare(M, Mhat0)
    pre_Mp_Mhatp = comp.compare(Mprime, Mhatp0)
    pre_Mhat_Mhatp = comp.compare(Mhat0, Mhatp0)

    # Stronger equivariance test: does Mhat' ≈ U Mhat U^T ?
    equiv_pre = fro_rel(gauge_transform(Mhat0, U), Mhatp0)

    print("\n=== BEFORE training ===")
    print("inv(M) vs inv(Mhat):", pretty(pre_M_Mhat))
    print("inv(M') vs inv(Mhat'):", pretty(pre_Mp_Mhatp))
    print("inv(Mhat) vs inv(Mhat'):", pretty(pre_Mhat_Mhatp))
    print("equivariance fro_rel( U Mhat U^T , Mhat' ):", round(equiv_pre, 6))

    # 5) Train both models with the same protocol/settings.
    train_model(model, seed=args.seed, use_mi=args.use_mi, use_rg=args.use_rg, use_ts=args.use_time_symmetric, epochs=args.train_epochs)
    train_model(model_p, seed=args.seed, use_mi=args.use_mi, use_rg=args.use_rg, use_ts=args.use_time_symmetric, epochs=args.train_epochs)

    # 6) Extract after training
    Mhat = rt.extract_matrix(core)
    Mhatp = rt.extract_matrix(core_p)

    post_M_Mhat = comp.compare(M, Mhat)
    post_Mp_Mhatp = comp.compare(Mprime, Mhatp)
    post_Mhat_Mhatp = comp.compare(Mhat, Mhatp)
    equiv_post = fro_rel(gauge_transform(Mhat, U), Mhatp)

    print("\n=== AFTER training ===")
    print("inv(M) vs inv(Mhat):", pretty(post_M_Mhat))
    print("inv(M') vs inv(Mhat'):", pretty(post_Mp_Mhatp))
    print("inv(Mhat) vs inv(Mhat'):", pretty(post_Mhat_Mhatp))
    print("equivariance fro_rel( U Mhat U^T , Mhat' ):", round(equiv_post, 6))

    # 7) Optional: print spectral summaries for intuition
    sd = SpectralDiagnostics()
    print("\nSpectral summaries:")
    print("M   :", pretty(sd.summarize(M)))
    print("M'  :", pretty(sd.summarize(Mprime)))
    print("Mhat:", pretty(sd.summarize(Mhat)))
    print("Mhat':", pretty(sd.summarize(Mhatp)))

if __name__ == "__main__":
    main()
