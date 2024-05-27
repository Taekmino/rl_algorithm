import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import wandb

# Initialize wandb
wandb.init(project="dqn-pendulum")

# Hyperparameters
config = {
    "GAMMA": 0.99,
    "LR": 1e-3,
    "BUFFER_SIZE": 100000,
    "BATCH_SIZE": 128,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY": 0.995,
    "EPISODES": 500,
    "NUM_SEEDS" : 10
}
wandb.config.update(config)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#util function
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = config["EPSILON_START"]
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config["LR"])
        self.replay_buffer = ReplayBuffer(config["BUFFER_SIZE"])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self):
        if self.replay_buffer.size() < config["BATCH_SIZE"]:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(config["BATCH_SIZE"])

        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_state).max(1)[0]
        target_q_values = reward + (config["GAMMA"] * next_q_values * (1 - done))

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)

        return loss.item()

for seed in range(config["NUM_SEEDS"]):
    wandb.init(project="dqn-pendulum", reinit=True, name=f'seed_{seed}', config=config)
    seed_all(seed)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # Discretizing the action space into 11 bins
    agent = DQNAgent(state_dim, action_dim)

    # Discretize action space
    def discretize_action(action):
        return np.clip(np.round(action * 10) / 10.0, -2.0, 2.0)

    max_reward = -float('inf')  # Initialize the max reward variable

    for episode in range(config["EPISODES"]):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action_idx = agent.select_action(state)
            action = discretize_action(np.linspace(-2, 2, action_dim)[action_idx])
            next_state, reward, done, truncated, _ = env.step([action])
            agent.replay_buffer.add(state, action_idx, reward, next_state, done)
            state = next_state

            episode_reward += reward
            loss = agent.train()

            if truncated:
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    wandb.summary["max_reward"] = max_reward  # Update wandb summary with the max reward
                break

        wandb.log({"episode": episode + 1, "reward": episode_reward, "loss": loss, "epsilon": agent.epsilon})
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Loss: {loss}, Epsilon: {agent.epsilon}")

        # Epsilon decay
        agent.epsilon = max(config["EPSILON_END"], agent.epsilon * config["EPSILON_DECAY"])

    # Save the trained model
    torch.save(agent.q_network.state_dict(), f"dqn_pendulum_{seed}.pth")
    wandb.save(f"dqn_pendulum_{seed}.pth")

# To load the model
# agent.q_network.load_state_dict(torch.load("dqn_pendulum.pth"))