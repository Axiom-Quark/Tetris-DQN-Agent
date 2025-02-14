import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import time

# Pygame

# Constants
WIDTH, HEIGHT = 300, 600
GRID_SIZE = 30
ROWS = HEIGHT // GRID_SIZE  # 20
COLS = WIDTH // GRID_SIZE  # 10
FPS = 60  # Increase for faster simulation during training

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [
    (0, 255, 255),  # Cyan
    (0, 0, 255),  # Blue
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),  # Green
    (255, 0, 0),  # Red
    (128, 0, 128)  # Purple
]

# Tetris pieces (I, J, L, O, S, Z, T)
SHAPES = [
    [[1, 1, 1, 1]],  # I

    [[1, 0, 0],
     [1, 1, 1]],  # J

    [[0, 0, 1],
     [1, 1, 1]],  # L

    [[1, 1],
     [1, 1]],  # O

    [[0, 1, 1],
     [1, 1, 0]],  # S

    [[1, 1, 0],
     [0, 1, 1]],  # Z

    [[0, 1, 0],
     [1, 1, 1]]  # T
]


class Piece:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = COLORS[SHAPES.index(shape)]
        self.rotation = 0  # Current rotation state


class Tetris:
    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = GRID_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.reset()
        self.action_space = 4
        self.observation_space = (ROWS, COLS)

    def reset(self):
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.game_over = False
        self.score = 0
        self.current_piece = self.new_piece()
        self.rows_cleared_total = 0
        return self.get_state()

    def new_piece(self):
        shape = random.choice(SHAPES)
        return Piece(COLS // 2 - len(shape[0]) // 2, 0, shape)


    def valid_move(self, piece, x, y, rotation):
        # Checks if a move is valid
        rotated_shape = self.rotate_piece(piece.shape, rotation)
        for i, row in enumerate(rotated_shape):
            for j, cell in enumerate(row):
                try:
                    if cell and (y + i >= ROWS or x + j < 0 or x + j >= COLS or self.board[y + i][x + j]):
                        return False
                except IndexError:
                    return False
        return True

    def rotate_piece(self, shape, rotation):
        return [list(reversed(col)) for col in zip(*shape[::-1])] * rotation

    def place_piece(self, piece):
        shape = self.rotate_piece(piece.shape, piece.rotation) # Get the rotated shape
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    self.board[piece.y + i][piece.x + j] = SHAPES.index(piece.shape) + 1 # Use the index for the board state

    def clear_lines(self):
        lines_cleared = 0
        for i, row in enumerate(self.board):
            if all(cell != 0 for cell in row):
                lines_cleared += 1
                del self.board[i]
                self.board.insert(0, [0] * COLS)  # Add a new blank row at the top
        self.rows_cleared_total += lines_cleared
        return lines_cleared  # Return number of lines cleared

    def step(self, action):

        if self.game_over:
          return self.get_state(), 0, True, {}

        reward = 0.1  # Base reward for surviving
        self.current_piece.y += 1  # Move down by default

        if action == 0:  # Move left
            if self.valid_move(self.current_piece, self.current_piece.x - 1, self.current_piece.y, self.current_piece.rotation):
                self.current_piece.x -= 1
        elif action == 1:  # Move right
            if self.valid_move(self.current_piece, self.current_piece.x + 1, self.current_piece.y, self.current_piece.rotation):
                self.current_piece.x += 1
        elif action == 2:  # Rotate
             if self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y, (self.current_piece.rotation + 1)%4):
                self.current_piece.rotation = (self.current_piece.rotation + 1) % 4

        # Check if the piece has landed (either at the bottom or on another piece)
        if not self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y, self.current_piece.rotation):
            self.current_piece.y -= 1  # Move back up one step
            self.place_piece(self.current_piece)  # Place the piece on the board
            lines_cleared = self.clear_lines()  # Clear lines and update the score
            reward += 10 * (lines_cleared ** 2) # Increased reward with lines squared
            self.score += lines_cleared
            self.current_piece = self.new_piece()  # Generate a new piece

            # Check for game over (if the new piece cannot be placed)
            if not self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y, self.current_piece.rotation):
                self.game_over = True
                reward = -10 # Penalty on game end

        return self.get_state(), reward, self.game_over, {}


    def get_state(self): # Return the current game state as a NumPy array
        return np.array(self.board) # Convert the board to a NumPy array

    def render(self):
        # Render the game state using Pygame for visualization
        self.screen.fill(BLACK)
        for i in range(ROWS):
            for j in range(COLS):
                if self.board[i][j]:
                    pygame.draw.rect(self.screen, COLORS[self.board[i][j] - 1],
                                     (j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(self.screen, WHITE, (j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

        # Draw the current piece
        if self.current_piece:
            shape = self.rotate_piece(self.current_piece.shape, self.current_piece.rotation)
            for i, row in enumerate(shape):
                for j, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(self.screen, self.current_piece.color,
                                         ((self.current_piece.x + j) * GRID_SIZE,
                                          (self.current_piece.y + i) * GRID_SIZE,
                                          GRID_SIZE, GRID_SIZE))

        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, WHITE)
        text_cleared = font.render(f"Rows: {self.rows_cleared_total}", True, WHITE) # Display rows cleared
        self.screen.blit(text, [0, 0])
        self.screen.blit(text_cleared, [0, 30])


        pygame.display.flip()
        self.clock.tick(FPS)

# --- Deep Q-Network (DQN) ---

class DQN(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(DQN, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

        # Convolutional Layers (for processing the game board)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input channels = 1 (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened features after the convolutional layers
        self.flattened_size = self._get_conv_output_size()

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self):
        # Helper function to calculate the output size of the convolutional layers
        dummy_input = torch.zeros(1, 1, *self.state_shape)  # Add a batch dimension and a channel dimension
        dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
        return int(np.prod(dummy_output.size()))


    def forward(self, x):
        # Input x (batch_size, height, width)
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, 1, height, width)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten (batch_size, flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output Q-values for each action
        return x

# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- Agent ---

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=10000, batch_size=32, target_update_freq=100):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.steps_done = 0


    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)  # Explore
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
                q_values = self.policy_net(state)
                return q_values.argmax().item()  # Exploit

    def update_model(self):
      if len(self.replay_buffer) < self.batch_size:
          return

      # Sample a batch from the replay buffer
      state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

      # Convert to tensors, move to the correct device
      state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)  # Keep as numpy array, then convert
      next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device) # Keep as numpy array, then convert
      action = torch.tensor(action, dtype=torch.long, device=self.device)
      reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
      done = torch.tensor(done, dtype=torch.float32, device=self.device)

      # Compute Q-values for current states and selected actions
      q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

      # Compute target Q-values using the target network
      with torch.no_grad():
          next_q_values = self.target_net(next_state).max(1)[0]
          target_q_values = reward + (1 - done) * self.gamma * next_q_values

      # Compute loss (Huber loss for stability)
      loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

      # Optimize the model
      self.optimizer.zero_grad()
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # Gradient clipping (optional)
      self.optimizer.step()

      return loss.item() # Return the loss value


    def update_target_network(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
      self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- Training Loop ---

def train(env, agent, num_episodes, visualize_every=50, save_every=500):
    episode_rewards = []
    episode_losses = []  # To store losses per episode
    running_reward = 0 # For tracking a moving average of the reward

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0

        while not done:
            for event in pygame.event.get():  # Allow closing the window
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return #important to quit the training

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            loss = agent.update_model()  # Update the model and get the loss
            if loss is not None: # update_model() returns None if buffer is not full
              total_loss += loss

            if steps % agent.target_update_freq == 0:
              agent.update_target_network()

            if (episode + 1) % visualize_every == 0:
                env.render()
                #time.sleep(0.05)  # Slow down for visualization (removed for speed)

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Calculate average loss for the episode (handle case where no updates happened)
        avg_loss = total_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)

        running_reward = 0.95 * running_reward + 0.05 * total_reward if running_reward != 0 else total_reward


        print(f"Episode: {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}, Buffer Size: {len(agent.replay_buffer)}")


        if (episode + 1) % save_every == 0:
            agent.save(f"tetris_dqn_model_ep{episode+1}.pth")

    return episode_rewards, episode_losses

# --- Main ---

if __name__ == "__main__":
    env = Tetris()  # Use your Tetris environment
    agent = DQNAgent(env.observation_space, env.action_space)

    num_episodes = 100
    episode_rewards, episode_losses = train(env, agent, num_episodes, visualize_every=100, save_every=100)

    # Plotting results (optional)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")

    plt.tight_layout()
    plt.show()
