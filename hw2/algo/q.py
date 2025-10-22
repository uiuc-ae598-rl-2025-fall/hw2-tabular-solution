"""Q-Learning algorithm for tabular reinforcement learning."""

import numpy as np


class QLearning:
    """
    Off-policy TD control (Q-Learning) algorithm.

    Implements the Q-Learning algorithm from Chapter 6.5 of Sutton and Barto.
    Uses epsilon-greedy behavior policy for exploration.

    Parameters
    ----------
    n_states : int
        Number of states in the environment.
    n_actions : int
        Number of actions in the environment.
    gamma : float, optional
        Discount factor, by default 0.95.
    alpha : float, optional
        Learning rate, by default 0.1.
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
    policy : np.ndarray
        Current policy (deterministic).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.95,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
    ):
        """Initialize Q-Learning agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-values
        self.Q = np.zeros((n_states, n_actions))

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

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """
        Update Q-values using Q-Learning update rule.

        Parameters
        ----------
        state : int
            Current state.
        action : int
            Action taken in current state.
        reward : float
            Reward received.
        next_state : int
            Next state.
        done : bool
            Whether the episode is done.
        """
        # Q-Learning update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        # Update policy (greedy with respect to Q)
        self.policy[state] = np.argmax(self.Q[state])

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
