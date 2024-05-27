import gymnasium as gym
from AC import ActorCritic
import torch
import torch.optim as optim

def train():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    best_reward = -float('inf')

    agent = ActorCritic(state_dim, action_dim, device)

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-4)

    actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.9)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.9)

    # agent.load_model('best_actor.pth', 'best_critic.pth')

    num_episodes = 5000
    for episode in range(num_episodes):
        state, _ = env.reset()

        episode_reward = 0

        while True:
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            transition = (state, action_log_prob, reward, next_state, done)
            agent.update(transition, actor_optimizer, critic_optimizer)

            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode {episode+1}: {episode_reward}")
                break

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('best_actor.pth', 'best_critic.pth')
            print("save model")
        
        actor_scheduler.step()
        critic_scheduler.step()


if __name__ == "__main__":
    train()