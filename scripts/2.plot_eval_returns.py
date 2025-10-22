"""Plot evaluation returns from trained policies.

This script loads the training results and plots the average evaluation
returns with percentile interval shading across multiple runs.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_evaluation_returns(results_dir="results", save_dir="figures"):
    """
    Plot evaluation returns for all algorithms and settings.

    Parameters
    ----------
    results_dir : str, optional
        Directory containing pickled results, by default "results".
    save_dir : str, optional
        Directory to save figures, by default "figures".
    """
    results_dir = Path(results_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    algorithms = ["MC", "SARSA", "Q-Learning"]
    slippery_settings = [True, False]

    # Create subplots for each slippery setting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = {
        "MC": "blue",
        "SARSA": "green",
        "Q-Learning": "red",
    }

    for idx, is_slippery in enumerate(slippery_settings):
        ax = axes[idx]
        slippery_str = "slippery" if is_slippery else "not_slippery"

        for algo_name in algorithms:
            # Load results
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                print(f"Warning: {results_path} not found, skipping...")
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            returns = np.array(results["returns"])  # shape: (n_runs, n_evals)
            timesteps = np.array(results["timesteps"][0])  # Use first run's timesteps

            # Calculate statistics
            mean_returns = np.mean(returns, axis=0)
            percentile_25 = np.percentile(returns, 25, axis=0)
            percentile_75 = np.percentile(returns, 75, axis=0)

            # Plot
            color = colors[algo_name]
            ax.plot(timesteps, mean_returns, label=algo_name, color=color, linewidth=2)
            ax.fill_between(
                timesteps, percentile_25, percentile_75, alpha=0.3, color=color
            )

        # Formatting
        title = "Slippery Environment" if is_slippery else "Non-Slippery Environment"
        ax.set_title(f"Evaluation Returns - {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / "evaluation_returns.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_path}")

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Create individual plots for each setting
    for is_slippery in slippery_settings:
        fig, ax = plt.subplots(figsize=(10, 6))
        slippery_str = "slippery" if is_slippery else "not_slippery"

        for algo_name in algorithms:
            # Load results
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            returns = np.array(results["returns"])
            timesteps = np.array(results["timesteps"][0])

            # Calculate statistics
            mean_returns = np.mean(returns, axis=0)
            percentile_25 = np.percentile(returns, 25, axis=0)
            percentile_75 = np.percentile(returns, 75, axis=0)

            # Plot
            color = colors[algo_name]
            ax.plot(timesteps, mean_returns, label=algo_name, color=color, linewidth=2)
            ax.fill_between(
                timesteps, percentile_25, percentile_75, alpha=0.3, color=color
            )

        # Formatting
        title = "Slippery Environment" if is_slippery else "Non-Slippery Environment"
        ax.set_title(f"Evaluation Returns - {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_path = save_dir / f"evaluation_returns_{slippery_str}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

        plt.show(block=False)
        plt.pause(3)
        plt.close()


def print_final_performance(results_dir="results"):
    """
    Print final performance statistics for all algorithms.

    Parameters
    ----------
    results_dir : str, optional
        Directory containing pickled results, by default "results".
    """
    results_dir = Path(results_dir)

    algorithms = ["MC", "SARSA", "Q-Learning"]
    slippery_settings = [True, False]

    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE STATISTICS")
    print("=" * 70)

    for is_slippery in slippery_settings:
        slippery_str = "slippery" if is_slippery else "not_slippery"
        title = "Slippery Environment" if is_slippery else "Non-Slippery Environment"

        print(f"\n{title}")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Mean Return':<15} {'Std Dev':<15} {'Best Run':<15}")
        print("-" * 70)

        for algo_name in algorithms:
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                print(f"{algo_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            returns = np.array(results["returns"])
            final_returns = returns[:, -1]  # Last evaluation for each run

            mean_return = np.mean(final_returns)
            std_return = np.std(final_returns)
            best_return = np.max(final_returns)

            print(
                f"{algo_name:<15} {mean_return:<15.4f} {std_return:<15.4f} {best_return:<15.4f}"
            )

    print("\n" + "=" * 70)


def main():
    """Main plotting function."""
    plot_evaluation_returns()
    print_final_performance()


if __name__ == "__main__":
    main()
