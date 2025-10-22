"""Plot policies and value functions from trained agents.

This script loads the trained agents and visualizes their learned
policies and value functions.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hw2.utils.render import plot_policy, plot_policy_and_value, plot_value_function


def plot_best_agents(results_dir="results", save_dir="figures"):
    """
    Plot policies and value functions for the best agent from each configuration.

    The best agent is selected based on the highest final evaluation return.

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

    algorithms = ["MC", "SARSA", "Q-Learning"]
    slippery_settings = [True, False]

    for is_slippery in slippery_settings:
        slippery_str = "slippery" if is_slippery else "not_slippery"
        title_suffix = "Slippery" if is_slippery else "Non-Slippery"

        print(f"\n{'=' * 60}")
        print(f"Plotting {title_suffix} Environment")
        print(f"{'=' * 60}")

        # Create figure with subplots for all algorithms
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))

        for idx, algo_name in enumerate(algorithms):
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                print(f"Warning: {results_path} not found, skipping...")
                continue

            # Load results
            with open(results_path, "rb") as f:
                results = pickle.load(f)

            # Find best agent (highest final evaluation return)
            returns = np.array(results["returns"])
            final_returns = returns[:, -1]
            best_run_idx = np.argmax(final_returns)
            best_agent = results["agents"][best_run_idx]

            print(f"\n{algo_name}:")
            print(f"  Best run: {best_run_idx + 1}")
            print(f"  Final return: {final_returns[best_run_idx]:.4f}")

            # Get policy and value function
            policy = best_agent.policy
            V = best_agent.get_value_function()

            # Plot policy
            ax_policy = axes[idx, 0]
            plot_policy(
                policy, grid_shape=(4, 4), title=f"{algo_name} - Policy", ax=ax_policy
            )

            # Plot value function
            ax_value = axes[idx, 1]
            plot_value_function(
                V, grid_shape=(4, 4), title=f"{algo_name} - Value Function", ax=ax_value
            )

        # Set overall title
        fig.suptitle(
            f"Policies and Value Functions - {title_suffix} Environment",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        # Save figure
        save_path = save_dir / f"policy_and_value_{slippery_str}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

        plt.show(block=False)
        plt.pause(3)
        plt.close()

        # Create individual figures for each algorithm
        for algo_name in algorithms:
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            returns = np.array(results["returns"])
            final_returns = returns[:, -1]
            best_run_idx = np.argmax(final_returns)
            best_agent = results["agents"][best_run_idx]

            policy = best_agent.policy
            V = best_agent.get_value_function()

            # Create individual figure
            title = f"{algo_name} - {title_suffix} Environment"
            save_path = save_dir / f"{algo_name}_{slippery_str}_policy_value.png"

            plot_policy_and_value(
                policy, V, grid_shape=(4, 4), title=title, save_path=save_path
            )
            print(f"Individual figure saved to {save_path}")

            plt.show(block=False)
            plt.pause(3)
            plt.close()


def plot_all_runs(results_dir="results", save_dir="figures"):
    """
    Plot value functions for all runs to show consistency.

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

    algorithms = ["MC", "SARSA", "Q-Learning"]
    slippery_settings = [True, False]

    for is_slippery in slippery_settings:
        slippery_str = "slippery" if is_slippery else "not_slippery"
        title_suffix = "Slippery" if is_slippery else "Non-Slippery"

        for algo_name in algorithms:
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            n_runs = len(results["agents"])

            # Create figure with subplots for all runs
            n_cols = 5
            n_rows = (n_runs + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axes = axes.flatten() if n_runs > 1 else [axes]

            for run_idx, agent in enumerate(results["agents"]):
                V = agent.get_value_function()
                ax = axes[run_idx]
                plot_value_function(
                    V, grid_shape=(4, 4), title=f"Run {run_idx + 1}", ax=ax
                )

            # Hide unused subplots
            for idx in range(n_runs, len(axes)):
                axes[idx].axis("off")

            fig.suptitle(
                f"{algo_name} - All Runs - {title_suffix} Environment",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            save_path = save_dir / f"{algo_name}_{slippery_str}_all_runs.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved {save_path}")

            plt.show(block=False)
            plt.pause(3)
            plt.close()


def compare_value_functions(results_dir="results", save_dir="figures"):
    """
    Compare average value functions across algorithms.

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

    algorithms = ["MC", "SARSA", "Q-Learning"]
    slippery_settings = [True, False]

    for is_slippery in slippery_settings:
        slippery_str = "slippery" if is_slippery else "not_slippery"
        title_suffix = "Slippery" if is_slippery else "Non-Slippery"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, algo_name in enumerate(algorithms):
            results_path = results_dir / f"{algo_name}_{slippery_str}.pkl"

            if not results_path.exists():
                continue

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            # Calculate average value function across all runs
            all_V = []
            for agent in results["agents"]:
                all_V.append(agent.get_value_function())

            avg_V = np.mean(all_V, axis=0)

            ax = axes[idx]
            plot_value_function(
                avg_V, grid_shape=(4, 4), title=f"{algo_name} - Average V", ax=ax
            )

        fig.suptitle(
            f"Average Value Functions - {title_suffix} Environment",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        save_path = save_dir / f"avg_value_functions_{slippery_str}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")

        plt.show(block=False)
        plt.pause(3)
        plt.close()


def main():
    """Main plotting function."""
    print("\nPlotting best agents...")
    plot_best_agents()

    print("\nComparing average value functions...")
    compare_value_functions()

    print("\nPlotting all runs...")
    plot_all_runs()

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
