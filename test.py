import gymnasium as gym
import torch
from AC import ActorCritic

def test():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    env = gym.make('CartPole-v1', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(state_dim, action_dim, device)
    agent.load_model('best_actor.pth', 'best_critic.pth')

    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            env.render()
            action, _ = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                print(f"Test Episode {episode+1}: {episode_reward}")
                break

    env.close()

if __name__ == "__main__":
    test()