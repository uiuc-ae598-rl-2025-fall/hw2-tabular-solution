"""Quick test script to verify the implementation works.

This script performs a quick sanity check by training each algorithm
for a few episodes and verifying basic functionality.
"""

import numpy as np
import gymnasium as gym

from hw2.algo.mc import MCControl
from hw2.algo.sarsa import SARSA
from hw2.algo.q import QLearning


def test_algorithm(agent_class, agent_name, n_episodes=100):
    """
    Test an algorithm with basic training.
    
    Parameters
    ----------
    agent_class : class
        Algorithm class to test.
    agent_name : str
        Name of the algorithm for display.
    n_episodes : int, optional
        Number of test episodes, by default 100.
    """
    print(f"\nTesting {agent_name}...")
    print("-" * 40)
    
    # Create environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    
    # Create agent
    agent = agent_class(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=0.95,
    )
    
    # Test basic functionality
    if agent_name == "Monte Carlo":
        # Test MC training
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_data = []
            done = False
            
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_data.append((state, action, reward))
                state = next_state
            
            agent.update(episode_data)
            agent.decay_epsilon()
    else:
        # Test TD algorithms (SARSA/Q-Learning)
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            
            if agent_name == "SARSA":
                action = agent.select_action(state, training=True)
            
            while not done:
                if agent_name == "Q-Learning":
                    action = agent.select_action(state, training=True)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if agent_name == "SARSA":
                    next_action = agent.select_action(next_state, training=True)
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action
                else:  # Q-Learning
                    agent.update(state, action, reward, next_state, done)
                
                state = next_state
            
            agent.decay_epsilon()
    
    # Evaluate
    eval_returns = []
    for _ in range(20):
        state, _ = env.reset()
        episode_return = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state
            steps += 1
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    print(f"Average return after {n_episodes} episodes: {mean_return:.3f}")
    print(f"Q-values shape: {agent.Q.shape}")
    print(f"Policy shape: {agent.policy.shape}")
    print(f"Value function shape: {agent.get_value_function().shape}")
    
    # Check that values have changed from initial zeros
    if np.any(agent.Q != 0):
        print("✓ Q-values updated successfully")
    else:
        print("✗ Warning: Q-values still zero")
    
    env.close()
    print(f"{agent_name} test complete!")


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING TABULAR RL IMPLEMENTATIONS")
    print("="*60)
    
    # Test each algorithm
    test_algorithm(MCControl, "Monte Carlo", n_episodes=100)
    test_algorithm(SARSA, "SARSA", n_episodes=100)
    test_algorithm(QLearning, "Q-Learning", n_episodes=100)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
    print("\nIf all algorithms show updated Q-values, the implementation is working.")
    print("You can now run the full training with: python scripts/1.train_policies.py")


if __name__ == "__main__":
    main()
