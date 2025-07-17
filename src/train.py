import random
import logging
logging.basicConfig(level=logging.INFO, filename="train_snake.log",filemode="w",format="%(asctime)s %(levelname)s %(message)s")

import numpy as np
import torch
import torch.nn as nn
from collections import deque, Counter


# память
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# выбор действия
def select_action(state, policy_net, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values))


# шаг тренировки
def train_step(memory, policy_net, target_net, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return

    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states)
    q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()


    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        q_target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def model_training(EPISODES, BATCH_SIZE, GAMMA, epsilon, EPS_END, EPS_DECAY, TARGET_UPDATE, env, policy_net, target_net, optimizer, device):

    memory = ReplayBuffer()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #Обучение
    success_count = 0
    best_len = 0
    max_ins_stat_len = 0
    n_actions = len(env.actions)
    ins_stat_len = []
    out_stat_len = []
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, n_actions, device)
            next_state, reward, status = env.step(action)
            done = status in ["win", "death", "timeout", "incorrect_move", "done"]

            memory.push(state, action, reward, next_state, done)
            train_step(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)
            state = next_state
            total_reward += reward

            if len(env.snake) >= best_len and np.mean(ins_stat_len) > max_ins_stat_len:
                print(f"model save\n best_len, len{best_len, len(env.snake)} \nmax_ins_stat_len, avg ns_stat_len{max_ins_stat_len, np.mean(ins_stat_len)}")
                best_len = len(env.snake)
                max_ins_stat_len = np.mean(ins_stat_len)
                torch.save(policy_net.state_dict(), "snake_model_best.pth")

            if done:
                logging.info(
                    f"Episode {episode:03d} | Steps: {env.steps:<3} | "
                    f"Reward: {total_reward:<6.1f} | len: {len(env.snake)} "
                    f"| Status: {status:<10}"
                )

        ins_stat_len.append(len(env.snake))
        out_stat_len.append(len(env.snake))

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print(f"Episode {episode} | "
                  f"epsilon: {epsilon:.4f} | "
                  f"avg len: {np.mean(ins_stat_len)} | "
                  f"len stat: {Counter(ins_stat_len)}")
            ins_stat_len = []

        if episode % 5000 == 0:
            print(f"success_count: {success_count}")

    torch.save(policy_net.state_dict(), "snake_model.pth")
    print("Модель сохранена как snake_model.pth")

    print(f"Полное прохождение карты: {success_count / EPISODES * 100:.2f}% \n len stat: {Counter(out_stat_len)} avg len: {np.mean(out_stat_len)}")

