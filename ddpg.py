import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import wandb


#setting
cfg = {
    "GAMMA" :       0.99,
    "TAU" :         0.005,
    "LR_ACTOR" :    1e-4,
    "LR_CRITIC" :   1e-3,
    "BUFFER_SIZE" : 1000000,
    "BATCH_SIZE" :  128,
    "EPISODES":     500,
    "NUM_SEEDS" : 10
}
# wandb.init(project="DDPG Pendulum", reinit=True, config=cfg)

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

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x * self.max_action

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=cfg["BUFFER_SIZE"]):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=cfg["BATCH_SIZE"]):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg["LR_ACTOR"])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg["LR_CRITIC"])

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if self.replay_buffer.size() < cfg["BATCH_SIZE"]:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(cfg["BATCH_SIZE"])

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * cfg["GAMMA"] * target_q).detach()

        current_q = self.critic(state, action)

        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(cfg["TAU"] * param.data + (1 - cfg["TAU"]) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(cfg["TAU"] * param.data + (1 - cfg["TAU"]) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))

# Main training loop
for seed in range(cfg["NUM_SEEDS"]):
    wandb.init(project="DDPG Pendulum", reinit=True, name=f'seed_{seed}', config=cfg)
    seed_all(seed)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    best_reward = -float('inf')

    for episode in range(cfg["EPISODES"]):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(np.array(state))
            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            agent.train()

            if truncated:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    wandb.run.summary["best_episode_reward"] = episode_reward
                    agent.save("ddpg_pendulum")
                    print("save!!")

                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        wandb.log({
            "episode_reward" : episode_reward
        })

    # Save the trained model
    agent.save("ddpg_pendulum")

# To load the model
# agent.load("ddpg_pendulum")