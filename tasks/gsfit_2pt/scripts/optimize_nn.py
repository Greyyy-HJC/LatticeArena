"""Optimize gsfit_2pt configurations with a tiny NumPy MLP surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.gsfit_2pt.benchmark.core import benchmark_config, make_synthetic_cases
from tasks.gsfit_2pt.interface import GroundStateFitConfig, config_to_dict
from tasks.gsfit_2pt.operators.plain import PlainGroundStateFit


MAX_STATES = 3


class TinyMLPRegressor:
    """A small one-hidden-layer MLP regressor implemented in NumPy."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.w2 = rng.normal(scale=np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=np.float64)
        self.x_mean = np.zeros(input_dim, dtype=np.float64)
        self.x_std = np.ones(input_dim, dtype=np.float64)
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int = 2500, lr: float = 0.02) -> None:
        self.x_mean = x.mean(axis=0)
        self.x_std = np.maximum(x.std(axis=0), 1e-6)
        x_norm = (x - self.x_mean) / self.x_std

        self.y_mean = float(y.mean())
        self.y_std = float(max(y.std(), 1e-6))
        y_norm = ((y - self.y_mean) / self.y_std).reshape(-1, 1)

        for _ in range(epochs):
            hidden_pre = x_norm @ self.w1 + self.b1
            hidden = np.tanh(hidden_pre)
            pred = hidden @ self.w2 + self.b2

            error = pred - y_norm
            grad_pred = (2.0 / len(x_norm)) * error
            grad_w2 = hidden.T @ grad_pred
            grad_b2 = grad_pred.sum(axis=0)

            grad_hidden = grad_pred @ self.w2.T
            grad_hidden_pre = grad_hidden * (1.0 - hidden**2)
            grad_w1 = x_norm.T @ grad_hidden_pre
            grad_b1 = grad_hidden_pre.sum(axis=0)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.x_mean) / self.x_std
        hidden = np.tanh(x_norm @ self.w1 + self.b1)
        pred = hidden @ self.w2 + self.b2
        return pred[:, 0] * self.y_std + self.y_mean


def sample_random_config(rng: np.random.Generator, *, lt: int = 48) -> GroundStateFitConfig:
    """Sample a valid fit configuration from a simple search space."""

    n_states = int(rng.integers(1, MAX_STATES + 1))
    t_min = int(rng.integers(2, 9))
    t_span = int(rng.integers(7, 15))
    t_max = min(t_min + t_span, lt - 2)

    e0_prior = (float(rng.uniform(0.22, 0.55)), float(rng.uniform(0.06, 0.28)))
    delta_e_priors = [
        (float(rng.uniform(0.18, 0.85)), float(rng.uniform(0.08, 0.35)))
        for _ in range(n_states - 1)
    ]
    amplitude_priors = [
        (float(rng.uniform(0.08, 1.20)), float(rng.uniform(0.08, 0.65)))
        for _ in range(n_states)
    ]

    return GroundStateFitConfig(
        t_min=t_min,
        t_max=t_max,
        n_states=n_states,
        e0_prior=e0_prior,
        delta_e_priors=delta_e_priors,
        amplitude_priors=amplitude_priors,
    )


def featurize_config(config: GroundStateFitConfig, *, lt: int = 48) -> np.ndarray:
    """Turn a config into a fixed-width feature vector for the MLP."""

    features = [
        config.t_min / lt,
        config.t_max / lt,
        config.n_states / MAX_STATES,
        config.e0_prior[0],
        config.e0_prior[1],
    ]

    for idx in range(MAX_STATES - 1):
        if idx < len(config.delta_e_priors):
            features.extend(config.delta_e_priors[idx])
        else:
            features.extend([0.0, 0.0])

    for idx in range(MAX_STATES):
        if idx < len(config.amplitude_priors):
            features.extend(config.amplitude_priors[idx])
        else:
            features.extend([0.0, 0.0])

    return np.asarray(features, dtype=np.float64)


def evaluate_config(config: GroundStateFitConfig, *, num_samples: int, max_resamples: int) -> dict[str, object]:
    """Evaluate a config on the deterministic synthetic benchmark."""

    cases = make_synthetic_cases(num_samples=num_samples)
    summary = benchmark_config(config, cases=cases, max_resample_fits=max_resamples)
    return {
        "config": config,
        "score": float(summary["score"]),
        "metrics": summary,
    }


def optimize_with_nn(
    *,
    train_evals: int,
    proposal_samples: int,
    top_k: int,
    num_samples: int,
    max_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Run the surrogate-guided config search and return the best evaluated config."""

    rng = np.random.default_rng(seed)
    evaluations: list[dict[str, object]] = []

    plain_config = PlainGroundStateFit().config
    evaluations.append(evaluate_config(plain_config, num_samples=num_samples, max_resamples=max_resamples))

    for _ in range(max(train_evals - 1, 0)):
        config = sample_random_config(rng)
        evaluations.append(evaluate_config(config, num_samples=num_samples, max_resamples=max_resamples))

    x_train = np.stack([featurize_config(item["config"]) for item in evaluations])
    y_train = np.asarray([item["score"] for item in evaluations], dtype=np.float64)
    model = TinyMLPRegressor(input_dim=x_train.shape[1], hidden_dim=32, seed=seed)
    model.fit(x_train, y_train)

    proposals = [sample_random_config(rng) for _ in range(proposal_samples)]
    x_proposal = np.stack([featurize_config(config) for config in proposals])
    predicted_scores = model.predict(x_proposal)
    top_indices = np.argsort(predicted_scores)[::-1][:top_k]

    for idx in top_indices:
        evaluations.append(
            evaluate_config(proposals[int(idx)], num_samples=num_samples, max_resamples=max_resamples)
        )

    best = max(evaluations, key=lambda item: item["score"])
    return {
        "best_config": best["config"],
        "best_score": best["score"],
        "best_metrics": best["metrics"],
        "train_evals": train_evals,
        "proposal_samples": proposal_samples,
        "top_k": top_k,
        "seed": seed,
        "num_samples": num_samples,
        "max_resamples": max_resamples,
        "plain_score": evaluations[0]["score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize gsfit_2pt configs with a tiny NN surrogate")
    parser.add_argument("--train-evals", type=int, default=40, help="Number of true benchmark evaluations for training")
    parser.add_argument("--proposal-samples", type=int, default=256, help="Random proposal configs scored by the NN")
    parser.add_argument("--top-k", type=int, default=24, help="Top predicted configs to evaluate with the true benchmark")
    parser.add_argument("--num-samples", type=int, default=24, help="Synthetic samples per benchmark case")
    parser.add_argument("--max-resamples", type=int, default=12, help="Resample refits per case during search")
    parser.add_argument("--seed", type=int, default=20260415, help="Random seed for config search")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("tasks/gsfit_2pt/operators/nn_config.json"),
        help="Path to write the optimized config JSON",
    )
    args = parser.parse_args()

    result = optimize_with_nn(
        train_evals=args.train_evals,
        proposal_samples=args.proposal_samples,
        top_k=args.top_k,
        num_samples=args.num_samples,
        max_resamples=args.max_resamples,
        seed=args.seed,
    )

    payload = {
        "meta": {
            "name": "nn",
            "description": "Tiny-MLP surrogate tuned gsfit_2pt configuration",
            "authors": ["LatticeArena"],
        },
        "config": config_to_dict(result["best_config"]),
        "search": {
            "best_score": result["best_score"],
            "plain_score": result["plain_score"],
            "train_evals": result["train_evals"],
            "proposal_samples": result["proposal_samples"],
            "top_k": result["top_k"],
            "seed": result["seed"],
            "num_samples": result["num_samples"],
            "max_resamples": result["max_resamples"],
        },
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output_config": str(args.output_config), **payload["search"]}, indent=2))


if __name__ == "__main__":
    main()
