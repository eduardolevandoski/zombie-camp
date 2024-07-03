import time
import numpy as np
import gymnasium as gym
from zombie_camp import ZombieCampEnvironment

# Environment
environment = gym.make('ZombieCampEnvironment', width=8, height=8, num_supplies=6, num_zombies=5, num_walls=2, num_rocks=2)

# Q-learning
learning_rate = 0.2 #alpha
discount_factor = 0.99 #ghama
exploration_rate = 1.0 #epsilon
exploration_min = 0.1
exploration_decay = 0.995
episodes = 1000
steps_per_episode = 100

q_values = np.zeros(tuple(environment.observation_space.nvec) + (environment.action_space.n,))

def select_action(state):
    if np.random.rand() < exploration_rate:
        return environment.action_space.sample()  # Exploration
    else:
        return np.argmax(q_values[state])  # Exploitation

# Training
for episode in range(episodes):
    state, _ = environment.reset()
    state = tuple(state)
    episode_reward = 0

    for step in range(steps_per_episode):
        action = select_action(state)
        next_state, reward, done, truncated, _ = environment.step(action)
        next_state = tuple(next_state)

        # Q(s,a)←Q(s,a)+α(r+γ max a′ ​Q(s′,a′)−Q(s,a))
        current_q = q_values[state][action]
        max_future_q = np.max(q_values[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        q_values[state][action] = new_q

        state = next_state
        episode_reward += reward

        if done:
            break

    if exploration_rate > exploration_min:
        exploration_rate *= exploration_decay

    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {episode_reward}")

print("Training completed.")

# Testing
state, _ = environment.reset()
state = tuple(state)

for step in range(steps_per_episode):
    action = np.argmax(q_values[state])
    next_state, reward, done, truncated, _ = environment.step(action)
    state = tuple(next_state)

    environment.render()

    time.sleep(0.6)

    if done:
        break

print(f"Total Reward: {environment.unwrapped.total_reward}")

environment.close()