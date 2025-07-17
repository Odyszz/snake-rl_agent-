import torch
import torch.optim as optim

from src.train import model_training
from src.model import SnakeDQN
from testing import testing_model
from snake_env import SnakeEnv



# Параметры
EPISODES = 30_000
EPISODES_TEST = 500
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 0.999
LR = 0.0005
TARGET_UPDATE = 30

# Подготовка
env = SnakeEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = SnakeDQN().to(device)
target_net = SnakeDQN().to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=0.01)

# Обучение
model_training(EPISODES, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, env, policy_net, target_net, optimizer, device)

# тестирование
testing_model(EPISODES_TEST)