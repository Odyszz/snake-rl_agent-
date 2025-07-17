import torch
import numpy as np

from src.train import select_action
from src.model import SnakeDQN
from snake_env import SnakeEnv, MAX_STEPS
from collections import Counter
from matplotlib import pyplot as plt
from loguru import logger
logger.remove()
logger.add("test_snake.log", format="{time} {level} {message}", level="INFO")

def testing_model(EPISODES):


    env = SnakeEnv()
    n_actions = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SnakeDQN().to(device)
    model.load_state_dict(torch.load("snake_model_best.pth", map_location=device))
    model.eval()


    success_count = 0
    total_rewards = []
    len_snake = []


    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        total_step = 0
        while not done:
            action = select_action(state, model, 0.0, n_actions, device)
            next_state, reward, status = env.step(action)
            done = status in ["win", "death", "timeout"]
            total_step +=1
            if total_step > MAX_STEPS:
                done = True

            state = next_state
            total_reward += reward
            if done:
                logger.info(
                    f"Episode {episode:03d} | Steps: {env.steps:<3} | "
                    f"Reward: {total_reward:<6.1f} | len: {len(env.snake)} "
                    f"Status: {status:<10}"
                )
        len_snake.append(len(env.snake))
        total_rewards.append(total_reward)

        print(f"Episode {episode + 1}: Reward = {total_reward:.1f} len: {len(env.snake)} success_count: {success_count}")

    print(f"Средняя награда: {np.mean(total_rewards):.2f}")
    print(f"Победы: {success_count / EPISODES * 100:.2f}% len stat: {Counter(len_snake)} avg len: {np.mean(len_snake)}")

    graph = []
    for i in sorted(Counter(len_snake)):
        print(f"{i}: {(Counter(len_snake)[i] / EPISODES * 100):.2f}%")
        graph.append(Counter(len_snake)[i])
    plt.hist(len_snake, bins=50, label="len")
    plt.show()


if __name__ == "__main__":
    print("start testing")
    testing_model(500)

