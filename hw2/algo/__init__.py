"""Tabular RL algorithms."""

from hw2.algo.mc import MCControl
from hw2.algo.q import QLearning
from hw2.algo.sarsa import SARSA

__all__ = ["MCControl", "SARSA", "QLearning"]
