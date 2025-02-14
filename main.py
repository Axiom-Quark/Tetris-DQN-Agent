import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# TETRIS ENVIRONMENT
class TetrisEnv:
    def __init__(self, grid_size=(10, 20)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.score = 0
        self.done = False
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.expand_dims(self.grid, axis=0)  # Shape (1, 10, 20)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        # Simulate a simple gravity drop and clearing mechanic
        self.grid[action, -1] = 1  # Drop a piece at column "action"
        rows_cleared = self.clear_rows()
        reward = 10 * rows_cleared
        self.score += reward

        if np.any(self.grid[0] > 0):  # Game over if top row is filled
            self.done = True
            reward -= 50

        return self.get_state(), reward, self.done

    def clear_rows(self):
        full_rows = [i for i in range(self.grid.shape[1]) if np.all(self.grid[:, i] > 0)]
        for row in full_rows:
            self.grid[:, row] = 0
        return len(full_rows)

    def render(self):
        plt.imshow(self.grid, cmap="gray_r")
        plt.show()

# DQN NEURAL NETWORK
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN AGENT
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation tradeoff
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor).detach().clone()
            target_f[action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# TRAINING LOOP
def train_tetris(episodes=500):
    env = TetrisEnv()
    agent = DQNAgent(state_size=200, action_size=10)  # 10 columns in Tetris
    scores = []

    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            agent.store(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done:
                break

        scores.append(total_reward)
        print(f"Episode {episode + 1}: Score {total_reward}, Epsilon {agent.epsilon:.4f}")

    return agent, scores

# RUN TRAINING
agent, scores = train_tetris(episodes=10000)

# PLOT RESULTS
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Tetris DQN Training Performance")
plt.show()
