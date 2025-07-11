import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# Parameters
num_vehicles = 100  # Number of vehicles in the simulation
episodes = 5  # Training episodes
max_steps = 100  # Max steps per episode
gamma = 0.9  # Discount factor for future rewards
epsilon = 1.0  # Epsilon for epsilon-greedy policy (Q-APERF)
epsilon_min = 0.05
epsilon_decay = 0.995
learning_rate = 0.001

# State space (3 continuous features and 3 binary event flags)
state_space = 6  # Change this to match the number of features
action_space = 4  # 4 QoS weight vectors for action selection
batch_size = 32

# Q-APERF (Tabular Q-learning)
Q_table = np.zeros((5, 5, 5, 5, 5, 5, action_space))  # Discretized state space (for simplicity)

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Input layer size must match state size
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

dqn_model = DQN(state_space, action_space)
optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Experience Replay for DQN
memory = deque(maxlen=10000)

# Synthetic Dataset Function (Randomized vehicular conditions)
def generate_synthetic_data():
    delay = np.random.randint(10, 100)  # Random delay in ms
    pdr = np.random.rand()  # Packet delivery ratio
    trust = np.random.rand()  # Trust score
    event_flags = np.random.choice([0, 1], size=3)  # 3 binary event flags (ambulance, crash, hazard)
    return delay, pdr, trust, event_flags

# Reward Calculation (Q-APERF)
def compute_q_aperf_reward(state, action):
    delay, pdr, trust = state[:3]  # Extract delay, PDR, and trust from the first 3 values
    event_flags = state[3:]  # Extract event flags (ambulance, crash, hazard) from the remaining 3 values
    reward = (0.5 * pdr) - (0.2 * delay) + (0.3 * trust)  # Simplified reward function
    if event_flags[0] == 1:  # Ambulance
        reward += 1.0  # Higher priority for ambulance
    return reward

# Function to discretize state into integer indices for Q-table
def discretize_state(state):
    # Normalize each feature to a value between 0 and 4 (assuming 5 discrete levels for each feature)
    discretized = np.clip(np.floor(state * 5).astype(int), 0, 4)
    return tuple(discretized)

# Initialize lists to store rewards for each episode
dqn_rewards = []
q_aperf_rewards = []

# DQN Training Loop
def train_dqn():
    global epsilon
    for episode in range(episodes):
        state = generate_synthetic_data()  # Generate synthetic data as a tuple (delay, PDR, trust, event_flags)
        # Flatten the state into a single list/array before converting to tensor
        state = np.concatenate([np.array(state[:3]), np.array(state[3])])  # Concatenate continuous and discrete values
        total_reward = 0
        for step in range(max_steps):
            # Epsilon-greedy policy for exploration/exploitation
            if random.random() < epsilon:
                action = random.choice(range(action_space))
            else:
                action = np.argmax(dqn_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy())

            next_state = generate_synthetic_data()  # Generate synthetic next state
            # Flatten the next_state
            next_state = np.concatenate([np.array(next_state[:3]), np.array(next_state[3])])  # Concatenate continuous and discrete values
            reward = compute_q_aperf_reward(next_state, action)  # Use Q-APERF reward calculation for DQN
            memory.append((state, action, reward, next_state))

            if len(memory) >= batch_size:
                # Experience replay
                batch = random.sample(memory, batch_size)
                for s, a, r, ns in batch:
                    q_value = dqn_model(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                    next_q_value = dqn_model(torch.tensor(ns, dtype=torch.float32).unsqueeze(0))
                    target = r + gamma * torch.max(next_q_value)
                    loss = loss_fn(q_value[0, a], target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            state = next_state
            total_reward += reward
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Store the total reward for the episode
        dqn_rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

# Q-APERF Training Loop (Tabular Q-learning)
def train_q_aperf():
    global epsilon  # Declare epsilon as global to modify its value
    for episode in range(episodes):
        state = generate_synthetic_data()  # Generate synthetic data as a tuple (delay, PDR, trust, event_flags)
        # Flatten the state into a single list/array before converting to tensor
        state = np.concatenate([np.array(state[:3]), np.array(state[3])])  # Concatenate continuous and discrete values
        discretized_state = discretize_state(state)  # Discretize the state into integer indices
        total_reward = 0
        for step in range(max_steps):
            # Epsilon-greedy policy for exploration/exploitation
            if random.random() < epsilon:
                action = random.choice(range(action_space))
            else:
                action = np.argmax(Q_table[discretized_state])

            next_state = generate_synthetic_data()  # Generate synthetic next state
            # Flatten the next_state
            next_state = np.concatenate([np.array(next_state[:3]), np.array(next_state[3])])  # Concatenate continuous and discrete values
            discretized_next_state = discretize_state(next_state)  # Discretize next state

            reward = compute_q_aperf_reward(next_state, action)  # Use Q-APERF reward calculation

            # Update Q-table
            Q_table[discretized_state, action] += learning_rate * (reward + gamma * np.max(Q_table[discretized_next_state]) - Q_table[discretized_state, action])

            discretized_state = discretized_next_state
            total_reward += reward

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Store the total reward for the episode
        q_aperf_rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

# Running the training loops
print("Training DQN...")
train_dqn()
print("Training Q-APERF...")
train_q_aperf()

# Plot results for comparison
plt.plot(dqn_rewards, label="DQN")
plt.plot(q_aperf_rewards, label="Q-APERF")
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Comparison of Q-APERF vs DQN')
plt.show()
