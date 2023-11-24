from frozen_lake_custom import FrozenLakeCustomEnv
import random
import numpy as np
import time

Env = FrozenLakeCustomEnv(render_mode="human")

actions = [0, 1, 2, 3]

observation, info = Env.reset()
action_size = Env.action_space.n
state_size = Env.observation_space.n


# Create our Q table with state_size rows and action_size columns (64x4)
qtable = np.zeros((state_size, action_size))

total_episodes = 1000       # Total episodes
learning_rate = 0.7          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
epsilon = 1.0                # Exploration rate
max_epsilon = 1.0            # Exploration probability at start
min_epsilon = 0.01           # Minimum exploration probability 
decay_rate = 0.005           # Exponential decay rate for exploration prob

# List of rewards
rewards = []

# Function to estimate time till completion
def estimate_time_till_completion(start_time, current_episode, total_episodes):
    elapsed_time = time.time() - start_time
    episodes_remaining = total_episodes - current_episode
    time_per_episode = elapsed_time / current_episode
    time_till_completion = episodes_remaining * time_per_episode
    return time_till_completion/60

# Training loop
start_time = time.time()

for episode in range(total_episodes):
    state, info = Env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = Env.action_space.sample()

        new_state, reward, done, truncated, info = Env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        total_rewards += reward
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

    # Print progress and estimated time till completion every 100 episodes
    if episode % 2 == 0 and episode > 0:
        ttc = estimate_time_till_completion(start_time, episode, total_episodes)
        print("Episode {}: Score = {:.2f}, Time Till Completion: {:.2f} seconds".format(episode, sum(rewards) / episode, ttc))

print("Training completed.")
print("Average Score over time: {:.2f}".format(sum(rewards) / total_episodes))
print(qtable)


for episode in range(5):
    state,info = Env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(99):
        # Take the action (index) that has the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done,turnicated, info = Env.step(action)

        if done:
            # Print the environment to see the final state
            Env.render()

            # Check if the agent reached the goal or fell into a hole
            if new_state == 15:
                print("We reached our Goal 🏆")
            else:
                print("We fell into a hole ☠️")

            # Print the number of steps it took
            print("Number of steps", step + 1)  # +1 because step is zero-indexed

            break

        state = new_state
Env.close()
