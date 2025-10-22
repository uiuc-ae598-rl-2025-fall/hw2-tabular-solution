"""Train tabular RL policies on FrozenLake environment.

This script trains 10 policies for each algorithm (MC, SARSA, Q-Learning)
and environment setting (is_slippery True/False), saving evaluation returns
for later plotting.
"""

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from hw2.algo.mc import MCControl
from hw2.algo.q import QLearning
from hw2.algo.sarsa import SARSA


def evaluate_policy(agent, env, n_episodes=100, max_steps=100):
    """
    Evaluate policy performance.

    Parameters
    ----------
    agent : object
        Agent with select_action method.
    env : gym.Env
        Environment to evaluate in.
    n_episodes : int, optional
        Number of episodes to evaluate, by default 100.
    max_steps : int, optional
        Maximum steps per episode, by default 100.

    Returns
    -------
    float
        Average return over evaluation episodes.
    """
    returns = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0

        for _ in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state

            if terminated or truncated:
                break

        returns.append(episode_return)

    return np.mean(returns)


def train_mc(env, n_episodes=10000, eval_interval=100, n_eval_episodes=100):
    """
    Train Monte Carlo control agent.

    Parameters
    ----------
    env : gym.Env
        Environment to train in.
    n_episodes : int, optional
        Number of training episodes, by default 10000.
    eval_interval : int, optional
        Evaluate every N episodes, by default 100.
    n_eval_episodes : int, optional
        Number of episodes for evaluation, by default 100.

    Returns
    -------
    tuple
        (agent, eval_returns, eval_timesteps)
    """
    agent = MCControl(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
    )

    eval_returns = []
    eval_timesteps = []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc="Training MC"):
        state, _ = env.reset()
        episode_data = []
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_data.append((state, action, reward))
            state = next_state
            total_steps += 1

        # Update agent
        agent.update(episode_data)
        agent.decay_epsilon()

        # Evaluate
        if (episode + 1) % eval_interval == 0:
            eval_return = evaluate_policy(agent, env, n_eval_episodes)
            eval_returns.append(eval_return)
            eval_timesteps.append(total_steps)

    return agent, eval_returns, eval_timesteps


def train_sarsa(env, n_episodes=10000, eval_interval=100, n_eval_episodes=100):
    """
    Train SARSA agent.

    Parameters
    ----------
    env : gym.Env
        Environment to train in.
    n_episodes : int, optional
        Number of training episodes, by default 10000.
    eval_interval : int, optional
        Evaluate every N episodes, by default 100.
    n_eval_episodes : int, optional
        Number of episodes for evaluation, by default 100.

    Returns
    -------
    tuple
        (agent, eval_returns, eval_timesteps)
    """
    agent = SARSA(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=0.95,
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
    )

    eval_returns = []
    eval_timesteps = []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc="Training SARSA"):
        state, _ = env.reset()
        action = agent.select_action(state, training=True)
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_state, training=True)

            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_steps += 1

        agent.decay_epsilon()

        # Evaluate
        if (episode + 1) % eval_interval == 0:
            eval_return = evaluate_policy(agent, env, n_eval_episodes)
            eval_returns.append(eval_return)
            eval_timesteps.append(total_steps)

    return agent, eval_returns, eval_timesteps


def train_qlearning(env, n_episodes=10000, eval_interval=100, n_eval_episodes=100):
    """
    Train Q-Learning agent.

    Parameters
    ----------
    env : gym.Env
        Environment to train in.
    n_episodes : int, optional
        Number of training episodes, by default 10000.
    eval_interval : int, optional
        Evaluate every N episodes, by default 100.
    n_eval_episodes : int, optional
        Number of episodes for evaluation, by default 100.

    Returns
    -------
    tuple
        (agent, eval_returns, eval_timesteps)
    """
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=0.95,
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
    )

    eval_returns = []
    eval_timesteps = []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc="Training Q-Learning"):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

        agent.decay_epsilon()

        # Evaluate
        if (episode + 1) % eval_interval == 0:
            eval_return = evaluate_policy(agent, env, n_eval_episodes)
            eval_returns.append(eval_return)
            eval_timesteps.append(total_steps)

    return agent, eval_returns, eval_timesteps


def main():
    """Main training loop."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Training parameters
    n_runs = 10
    n_episodes = 10000
    eval_interval = 100

    algorithms = {
        "MC": train_mc,
        "SARSA": train_sarsa,
        "Q-Learning": train_qlearning,
    }

    slippery_settings = [True, False]

    # Train for each configuration
    for is_slippery in slippery_settings:
        slippery_str = "slippery" if is_slippery else "not_slippery"
        print(f"\n{'=' * 60}")
        print(f"Training with is_slippery={is_slippery}")
        print(f"{'=' * 60}\n")

        for algo_name, train_fn in algorithms.items():
            print(f"\nAlgorithm: {algo_name}")
            print("-" * 40)

            all_returns = []
            all_timesteps = []
            all_agents = []

            for run in range(n_runs):
                print(f"\nRun {run + 1}/{n_runs}")

                # Create environment
                env = gym.make(
                    "FrozenLake-v1",
                    map_name="4x4",
                    is_slippery=is_slippery,
                )

                # Train agent
                agent, eval_returns, eval_timesteps = train_fn(
                    env, n_episodes, eval_interval
                )

                all_returns.append(eval_returns)
                all_timesteps.append(eval_timesteps)
                all_agents.append(agent)

                env.close()

            # Save results
            results = {
                "returns": all_returns,
                "timesteps": all_timesteps,
                "agents": all_agents,
                "algorithm": algo_name,
                "is_slippery": is_slippery,
                "n_runs": n_runs,
                "n_episodes": n_episodes,
                "eval_interval": eval_interval,
            }

            save_path = results_dir / f"{algo_name}_{slippery_str}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

            print(f"\nResults saved to {save_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
