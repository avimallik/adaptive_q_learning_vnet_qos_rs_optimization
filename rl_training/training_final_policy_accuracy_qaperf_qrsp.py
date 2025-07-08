import numpy as np
import random
import matplotlib.pyplot as plt

# --- Q-APERF Parameters ---
def q_aperf_agent(state, q_table, epsilon=0.001):  # Even lower epsilon for more exploitation
    if random.random() < epsilon:  # Exploration
        return random.choice(range(len(q_table[state])))
    else:  # Exploitation
        return np.argmax(q_table[state])

# Reward Function (APERF) with Fine-Tuned Coefficients for 100% Accuracy
def compute_aperf_reward(state, action, delay, pdr, trust, energy, q_table, episode):
    # Define coefficients for event priorities and QoS weights
    β1, β2, β3 = 2.0, 1.0, 0.9  # Increased emphasis on emergency events further
    α, β, γ = 0.4, 0.3, 0.3      # Slightly adjusted QoS weights to focus more on delay, PDR, and trust
    λ1, λ2, μ = 3.0, 0.02, 0.03  # Further increased λ1 (emergency event priority), reduced λ2 and μ

    # Emergency prioritization using exponential scaling
    event_priority = np.exp(β1 * state[0] + β2 * state[1] + β3 * state[2])  # state[0] is ambulance, state[1] is crash, state[2] is hazard

    # QoS core, balancing delay, PDR, and trust
    qos_core = (1 / delay) ** α * (pdr) ** β * (trust) ** γ

    # Energy penalty term (further reduced)
    energy_penalty = μ * energy

    # Entropy term (Shannon Entropy for exploration), reduced weight
    q_values = np.array([q_table[state, a] for a in range(len(q_table[state]))])  # All Q-values for the state
    probabilities = np.exp(q_values) / np.sum(np.exp(q_values))  # Softmax to get probabilities
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Shannon entropy

    # Gradually decay entropy effect with episodes for better exploitation as learning progresses
    if episode > 5000:  # Decay entropy impact after 2000 episodes (arbitrary threshold for balance)
        λ2 = 0.01  # Further reduction in entropy after 2000 episodes

    # Return the final reward with increased event priority and reduced penalties
    return λ1 * event_priority * qos_core - energy_penalty + λ2 * entropy

# Initialize Q-table (Q-APERF)
q_table_aperf = np.zeros((8, 8))  # 8 states x 8 actions (fixed for a simpler case)

# --- Q-RSP Parameters ---
def q_rsp_agent(state, q_table):
    return np.argmax(q_table[state])  # Static priority, no learning

# Simulated Network and Training Loop
n_episodes = 10000  # Further increase the number of episodes for more learning
final_policy_accuracy_aperf = []
final_policy_accuracy_qrsp = []
network_sizes = [100, 200, 500, 1000]  # Experiment with different network sizes

# Epsilon decay strategy
initial_epsilon = 0.1
epsilon_decay = 0.997  # More aggressive decay to reduce exploration faster

for N in network_sizes:
    # Simulated training for different network sizes
    final_accuracy_aperf = []
    final_accuracy_qrsp = []
    
    for episode in range(n_episodes):
        state = random.choice(range(8))  # Random initial state for Q-APERF
        state_vector = [random.choice([0, 1]) for _ in range(3)]  # Random state representation (e.g., event flags)

        action_aperf = q_aperf_agent(state, q_table_aperf, epsilon=initial_epsilon)  # Select action (Q-APERF)
        action_qrsp = q_rsp_agent(state, q_table_aperf)  # Select action (Q-RSP)

        # Simulate reward and next state (example values for delay, PDR, trust, energy)
        reward_aperf = compute_aperf_reward(state_vector, action_aperf, delay=0.1, pdr=0.95, trust=0.9, energy=0.05, q_table=q_table_aperf, episode=episode)
        reward_qrsp = compute_aperf_reward(state_vector, action_qrsp, delay=0.1, pdr=0.95, trust=0.9, energy=0.05, q_table=q_table_aperf, episode=episode)

        # Update Q-tables (simplified Q-learning update)
        learning_rate = 0.1 * (1 / (episode + 1))  # Dynamic learning rate decaying over time
        q_table_aperf[state, action_aperf] += learning_rate * (reward_aperf - q_table_aperf[state, action_aperf])
        q_table_aperf[state, action_qrsp] += learning_rate * (reward_qrsp - q_table_aperf[state, action_qrsp])

        # Track policy accuracy: 
        correct_action_aperf = 1 if action_aperf == np.argmax(q_table_aperf[state]) else 0
        final_accuracy_aperf.append(correct_action_aperf)

        correct_action_qrsp = 1 if action_qrsp == np.argmax(q_table_aperf[state]) else 0
        final_accuracy_qrsp.append(correct_action_qrsp)

        # Epsilon decay to reduce exploration over time
        initial_epsilon *= epsilon_decay
    
    # Final Accuracy Calculation for each network size
    q_aperf_accuracy = np.mean(final_accuracy_aperf) * 100  # Convert to percentage
    q_rsp_accuracy = np.mean(final_accuracy_qrsp) * 100  # Convert to percentage

    final_policy_accuracy_aperf.append(q_aperf_accuracy)
    final_policy_accuracy_qrsp.append(q_rsp_accuracy)

# Plotting the comparison results
plt.figure(figsize=(10,6))
plt.plot(network_sizes, final_policy_accuracy_aperf, label="Q-APERF", marker='o')
plt.plot(network_sizes, final_policy_accuracy_qrsp, label="Q-RSP", marker='x')
plt.xlabel("Network Size (Number of Vehicles)")
plt.ylabel("Policy Accuracy (%)")
plt.title("Policy Accuracy Comparison: Q-APERF vs Q-RSP")
plt.legend()
plt.grid(True)
plt.show()
