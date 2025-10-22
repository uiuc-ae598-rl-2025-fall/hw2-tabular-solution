"""Monte Carlo Control algorithm for tabular reinforcement learning."""

from collections import defaultdict

import numpy as np


class MCControl:
    """
    On-policy first-visit Monte Carlo control algorithm.

    Implements the MC control algorithm from Chapter 5.4 of Sutton and Barto.
    Uses epsilon-greedy policy for exploration.

    Parameters
    ----------
    n_states : int
        Number of states in the environment.
    n_actions : int
        Number of actions in the environment.
    gamma : float, optional
        Discount factor, by default 0.95.
    epsilon : float, optional
        Exploration parameter for epsilon-greedy policy, by default 0.1.
    epsilon_decay : float, optional
        Decay rate for epsilon, by default 1.0 (no decay).
    epsilon_min : float, optional
        Minimum value for epsilon, by default 0.01.

    Attributes
    ----------
    Q : np.ndarray
        Action-value function Q(s, a).
    returns : dict
        Dictionary storing returns for each state-action pair.
    policy : np.ndarray
        Current policy (deterministic).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
    ):
        """Initialize MC Control agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-values and returns
        self.Q = np.zeros((n_states, n_actions))
        self.returns = defaultdict(list)

        # Initialize policy (greedy policy)
        self.policy = np.zeros(n_states, dtype=int)

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : int
            Current state.
        training : bool, optional
            If True, use epsilon-greedy policy; if False, use greedy policy,
            by default True.

        Returns
        -------
        int
            Selected action.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))

    def update(self, episode: list) -> None:
        """
        Update Q-values using first-visit Monte Carlo.

        Parameters
        ----------
        episode : list
            List of (state, action, reward) tuples from the episode.
        """
        # Calculate returns for each time step
        G = 0
        visited = set()

        # Iterate backwards through the episode
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            # First-visit MC: only update if this is the first visit to (s, a)
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[state, action] = np.mean(self.returns[(state, action)])

        # Update policy (greedy with respect to Q)
        for s in range(self.n_states):
            self.policy[s] = np.argmax(self.Q[s])

    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy policy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_value_function(self) -> np.ndarray:
        """
        Get state-value function from Q-values.

        Returns
        -------
        np.ndarray
            State-value function V(s) = max_a Q(s, a).
        """
        return np.max(self.Q, axis=1)
