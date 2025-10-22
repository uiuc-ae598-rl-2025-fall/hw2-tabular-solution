#!/usr/bin/env python
"""Run all scripts in sequence.

This convenience script runs:
1. Quick test to verify implementation
2. Full training (takes significant time)
3. Plot evaluation returns
4. Plot policies and value functions
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path, description):
    """
    Run a Python script and handle errors.

    Parameters
    ----------
    script_path : str
        Path to the script to run.
    description : str
        Description of what the script does.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70 + "\n")

    try:
        subprocess.run([sys.executable, script_path], check=True, text=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_path}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {description} interrupted by user")
        return False


def main():
    """Run all scripts in sequence."""
    print("=" * 70)
    print("HW2 TABULAR METHODS - FULL PIPELINE")
    print("=" * 70)

    scripts = [
        ("scripts/1.train_policies.py", "Training All Policies (this will take time)"),
        ("scripts/2.plot_eval_returns.py", "Plotting Evaluation Returns"),
        ("scripts/3.plot_policy_and_vf.py", "Plotting Policies and Value Functions"),
    ]

    # Check if we should skip training
    results_dir = Path("results")
    if results_dir.exists() and any(results_dir.glob("*.pkl")):
        print("\nFound existing results in 'results/' directory.")
        response = input("Skip training and use existing results? [y/N]: ")
        if response.lower() == "y":
            print("Skipping training step...")
            scripts = [scripts[0]] + scripts[2:]  # Skip training script

    # Run each script
    for script_path, description in scripts:
        if not Path(script_path).exists():
            print(f"\n✗ Script not found: {script_path}")
            continue

        success = run_script(script_path, description)

        if not success:
            print("\nPipeline stopped due to error or interruption.")
            response = input("Continue with remaining scripts? [y/N]: ")
            if response.lower() != "y":
                break

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - results/: Contains trained agents and evaluation data")
    print("  - figures/: Contains all plots and visualizations")


if __name__ == "__main__":
    main()
