"""HW2 - Tabular Methods for Reinforcement Learning.

This package implements three model-free RL algorithms:
- Monte Carlo Control
- SARSA (on-policy TD control)
- Q-Learning (off-policy TD control)

All algorithms are tested on the FrozenLake-v1 environment.
"""

__version__ = "0.1.0"

from hw2.algo.mc import MCControl
from hw2.algo.q import QLearning
from hw2.algo.sarsa import SARSA

__all__ = ["MCControl", "SARSA", "QLearning"]
