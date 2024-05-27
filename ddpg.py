import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import wandb
import argparse

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds of iteration")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of testing episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update factor")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Learning rate for actor")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Learning rate for critic")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--project_name", type=str, default="DDPG Pendulum", help="wandb project name")
    return parser.parse_args()

def initialize_cfg(args):
    wandb.init(project=args.project_name)
    config = {
        "GAMMA": args.gamma,
        "TAU": args.tau,
        "LR_ACTOR": args.lr_actor,
        "LR_CRITIC": args.lr_critic,
        "BUFFER_SIZE": args.buffer_size,
        "BATCH_SIZE": args.batch_size,
        "TRAIN_EPISODES": args.train_episodes,
        "TEST_EPISODES": args.test_episodes
    }
    wandb.config.update(config)
    return config

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
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, config):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["LR_ACTOR"])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["LR_CRITIC"])

        self.replay_buffer = ReplayBuffer(config["BUFFER_SIZE"])
        self.max_action = max_action
        self.gamma = config["GAMMA"]
        self.tau = config["TAU"]
        self.batch_size = config["BATCH_SIZE"]

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = np.clip(action + np.random.normal(0, noise, size=action.shape), -self.max_action, self.max_action)
        return action
    
    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()

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
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))

def train(env, agent, episodes):
    best_reward = -float('inf')
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        loss_count = 0

        while True:
            action = agent.select_action(state, noise=0.1)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            agent.train()
            critic_loss, actor_loss = agent.train()

            if critic_loss is not None and actor_loss is not None:
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                loss_count += 1

            if truncated:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    wandb.run.summary["best_episode_reward"] = episode_reward
                break
        
        if loss_count > 0:
            episode_critic_loss /= loss_count
            episode_actor_loss /= loss_count

        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "critic_loss": episode_critic_loss if loss_count > 0 else None,
            "actor_loss": episode_actor_loss if loss_count > 0 else None
        })
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Critic Loss: {episode_critic_loss}, Actor Loss: {episode_actor_loss}")

def test(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if truncated:
                break

        rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")
    avg_reward = np.mean(rewards)
    wandb.log({"avg_test_reward": avg_reward})
    print(f"Average Test Reward: {avg_reward}")

def main():
    args = get_args()
    config = initialize_cfg(args)
    for seed in range(args.seeds):
        wandb.init(project=args.project_name, reinit=True, name=f'seed_{seed}', config=config)
        seed_all(seed)

        env = gym.make('Pendulum-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = DDPGAgent(state_dim, action_dim, max_action, config)

        if args.mode == "train":
            print("Starting training...")
            train(env, agent, args.train_episodes)
            wandb.save("ddpg_pendulum_actor.pth")
            wandb.save("ddpg_pendulum_critic.pth")
            agent.save("ddpg_pendulum")

        elif args.mode == "test":
            agent.load("ddpg_pendulum")
            print("Starting testing...")
            test(env, agent, args.test_episodes)

if __name__ == "__main__":
    main()