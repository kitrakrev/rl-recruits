"""
Metrics helpers: reward curve plots and JSON summary persistence.

plot_reward_curves(results, n_episodes)
    Saves training/reward_curves.png — three subplots:
    1. Reward curve (cumulative env reward per episode)
    2. Episode profit ($)
    3. Training loss (if present in results)

save_metrics(results)
    Saves training/metrics_summary.json — mean/max/min stats per policy/run.
"""
from __future__ import annotations

import json
from pathlib import Path


def plot_reward_curves(results: dict, n_episodes: int) -> None:
    """Plot and save reward curves for all policies / training runs.

    Parameters
    ----------
    results : dict mapping name → {"rewards": [...], "profits": [...], "losses": [...]}
    n_episodes : total episodes (used for x-axis labels)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] matplotlib not installed — skipping plot. Run: uv pip install matplotlib")
        return

    Path("training").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _COLORS = {
        "random":            "#e74c3c",
        "greedy":            "#f39c12",
        "optimal_heuristic": "#27ae60",
        "grpo_mc_training":  "#2980b9",
    }

    for policy_name, data in results.items():
        rewards = data.get("rewards", [])
        profits = data.get("profits", [])
        losses  = data.get("losses", [])
        color   = _COLORS.get(policy_name, "#8e44ad")

        def _smoothed(arr):
            if not arr:
                return [], []
            n = len(arr)
            window = max(1, n // 10)
            sm = np.convolve(arr, np.ones(window) / window, mode="valid")
            return range(n), range(len(sm)), arr, sm

        if rewards:
            x, xs, raw, sm = _smoothed(rewards)
            axes[0].plot(x, raw, alpha=0.2, color=color)
            axes[0].plot(xs, sm, color=color, linewidth=2, label=policy_name)

        if profits:
            x, xs, raw, sm = _smoothed(profits)
            axes[1].plot(x, raw, alpha=0.2, color=color)
            axes[1].plot(xs, sm, color=color, linewidth=2, label=policy_name)

        if losses:
            x, xs, raw, sm = _smoothed(losses)
            axes[2].plot(x, raw, alpha=0.2, color="#8e44ad")
            axes[2].plot(xs, sm, color="#8e44ad", linewidth=2, label="training loss")

    axes[0].set_title("Reward Curve (env cumulative)", fontsize=12)
    axes[0].set_ylabel("Cumulative Env Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Episode Profit ($)", fontsize=12)
    axes[1].set_ylabel("Net Profit ($)")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].set_title("Training Loss", fontsize=12)
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Staffing Agency RL — Training Progress", fontsize=14)
    plt.tight_layout()
    out_path = "training/reward_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path}")


def save_metrics(results: dict) -> None:
    """Persist per-policy mean/max/min stats to training/metrics_summary.json."""
    Path("training").mkdir(exist_ok=True)
    summary: dict = {}
    for policy, data in results.items():
        rewards = data.get("rewards", [0])
        profits = data.get("profits", [0])
        losses  = data.get("losses", [])
        summary[policy] = {
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "mean_profit": round(sum(profits) / len(profits), 2),
            "max_profit":  round(max(profits), 2),
            "min_profit":  round(min(profits), 2),
            "positive_profit_rate": round(sum(1 for p in profits if p > 0) / len(profits), 3),
            "mean_loss": round(sum(losses) / max(1, len(losses)), 4) if losses else 0.0,
        }
    out_path = "training/metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Saved: {out_path}")
    print("\nMetrics Summary:")
    for policy, m in summary.items():
        print(
            f"  {policy:25s}: mean_profit=${m['mean_profit']:>10,.0f}  "
            f"positive_rate={m['positive_profit_rate']:.1%}"
        )
