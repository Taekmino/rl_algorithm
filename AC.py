import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic:
    def __init__(self, state_dim, action_dim, device, gamma=0.99):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.gamma = gamma
        self.action_dim = action_dim

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, transition, actor_optimizer, critic_optimizer):
        state, action_log_prob, reward, next_state, done = transition

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Compute the target value
        target_value = reward + self.gamma * self.critic(next_state) * (1 - done)
        # Compute the value of the current state
        current_value = self.critic(state)

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(target_value, current_value).mean()

        # Update critic network
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        advantage = (target_value - current_value).detach()
        actor_loss = -action_log_prob * advantage

        # Update actor network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
    def save_model(self, actor_path='actor.pth', critic_path='critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path='actor.pth', critic_path='critic.pth'):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))