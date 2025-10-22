"""Utility functions for rendering policies and value functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def plot_value_function(
    V, grid_shape=(4, 4), title="Value Function", save_path=None, ax=None
):
    """
    Plot the value function as a heatmap.

    Parameters
    ----------
    V : np.ndarray
        Value function of shape (n_states,).
    grid_shape : tuple, optional
        Shape of the grid (rows, cols), by default (4, 4).
    title : str, optional
        Title of the plot, by default "Value Function".
    save_path : str, optional
        Path to save the figure, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Reshape value function to grid
    V_grid = V.reshape(grid_shape)

    # Create heatmap
    im = ax.imshow(V_grid, cmap="viridis", aspect="auto")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax.text(
                j,
                i,
                f"{V_grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

    ax.set_title(title)
    ax.set_xticks(range(grid_shape[1]))
    ax.set_yticks(range(grid_shape[0]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_policy(policy, grid_shape=(4, 4), title="Policy", save_path=None, ax=None):
    """
    Plot the policy as arrows on a grid.

    Parameters
    ----------
    policy : np.ndarray
        Policy of shape (n_states,) with action indices.
    grid_shape : tuple, optional
        Shape of the grid (rows, cols), by default (4, 4).
    title : str, optional
        Title of the plot, by default "Policy".
    save_path : str, optional
        Path to save the figure, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Reshape policy to grid
    policy_grid = policy.reshape(grid_shape)

    # Action mapping: 0=Left, 1=Down, 2=Right, 3=Up
    action_arrows = {
        0: (-0.3, 0),  # Left
        1: (0, 0.3),  # Down
        2: (0.3, 0),  # Right
        3: (0, -0.3),  # Up
    }

    # Create background grid
    ax.imshow(np.zeros(grid_shape), cmap="gray_r", alpha=0.3, aspect="auto")

    # Draw arrows
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            action = policy_grid[i, j]
            dx, dy = action_arrows[action]

            arrow = FancyArrowPatch(
                (j, i),
                (j + dx, i + dy),
                arrowstyle="->",
                mutation_scale=20,
                color="blue",
                linewidth=2,
            )
            ax.add_patch(arrow)

    ax.set_title(title)
    ax.set_xticks(range(grid_shape[1]))
    ax.set_yticks(range(grid_shape[0]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, grid_shape[1] - 0.5)
    ax.set_ylim(grid_shape[0] - 0.5, -0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_policy_and_value(
    policy, V, grid_shape=(4, 4), title="Policy and Value Function", save_path=None
):
    """
    Plot both policy and value function side by side.

    Parameters
    ----------
    policy : np.ndarray
        Policy of shape (n_states,) with action indices.
    V : np.ndarray
        Value function of shape (n_states,).
    grid_shape : tuple, optional
        Shape of the grid (rows, cols), by default (4, 4).
    title : str, optional
        Title of the plot, by default "Policy and Value Function".
    save_path : str, optional
        Path to save the figure, by default None.

    Returns
    -------
    tuple
        Figure and axes objects.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_policy(policy, grid_shape, "Policy", ax=ax1)
    plot_value_function(V, grid_shape, "Value Function", ax=ax2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)
