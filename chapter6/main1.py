
import gym
from gym import spaces
import numpy as np
import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import ptan
import numpy as np

class CIFAR10Env(gym.Env):
    def __init__(self, subset='train'):
        super(CIFAR10Env, self).__init__()

        # Load CIFAR-10 dataset
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = datasets.CIFAR10(root='./data', train=(subset == 'train'), download=True, transform=self.transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=2)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # 10 classes in CIFAR-10
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 32, 32), dtype=np.float32)

        # Initialize state
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self._get_observation()

    def step(self, action):
        # Take action and return next state, reward, done, and info
        reward = self._calculate_reward(action)
        done = self.current_index == len(self.dataset) - 1
        info = {}

        self.current_index += 1

        return obs, reward, done, info

    def _get_observation(self):
        # Get the current observation (image)
        image, _ = next(iter(self.loader))
        return image.squeeze(0).numpy()

    def _calculate_reward(self, action):
        # Placeholder reward function (you may want to customize this based on your task)
        true_label = self.dataset[self.current_index][1]
        temp_reward = 1.0 if action == true_label else 0.0
        
        return float(temp_reward)

# Example of using the CIFAR10Env
env = CIFAR10Env(subset='train')

# Reset the environment
obs = env.reset()

# Sample random actions for 10 steps
for _ in range(10):
    observation = env._get_observation()

    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")


env.observation_space.shape[0]



class DQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DQN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of the output after convolutional layers
        self.fc_input_size = self._calculate_conv_output_size(input_channels, 16, 2) * \
                             self._calculate_conv_output_size(16, 32, 2) * \
                             self._calculate_conv_output_size(32, 64, 2) * \
                             self._calculate_conv_output_size(64, 128, 2)

        self.fc_layers = nn.Sequential(
            # nn.Linear(2048, 2048),  # Update input size based on flattened output size
            # nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        
        x = self.conv_layers(x)
        
        # Print the shape of the output after convolutional layers
        print("Conv Output Shape:", x.shape)
        
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        
        # Print the shape of the flattened output
        print("Flattened Output Shape:", x.shape)
        
        x = self.fc_layers(x)
        
        # Print the shape of the output after fully connected layers
        print("FC Output Shape:", x.shape)
        
        return x

    def _calculate_conv_output_size(self, in_channels, out_channels, stride):
        # Function to calculate the size of the output after a convolutional layer
        dummy_input = torch.zeros(1, in_channels, 32, 32)
        dummy_output = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)(dummy_input)
        return dummy_output.size(2)


def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for experience in batch:
        states.append(experience.state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        dones.append(experience[-1])  # Accessing done flag directly
        next_states.append(experience.last_state)

    states_v = torch.tensor(states, dtype=torch.float32)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    
    # Convert 'dones' to a NumPy array before creating the PyTorch tensor
    dones_mask = torch.tensor(np.array(dones), dtype=torch.bool)
    
    next_states_v = torch.tensor(next_states, dtype=torch.float32)

    return states_v, actions_v, rewards_v, dones_mask, next_states_v


net = DQN(env.observation_space.shape[0], env.action_space.n)
net


# Create CartPole environment
env = CIFAR10Env(subset="train")

# Set random seeds for reproducibility
#env.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Neural network and optimizer
net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net = ptan.agent.TargetNet(net)
selector = ptan.actions.ArgmaxActionSelector()
epsilon_greedy = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1, selector=selector)
agent = ptan.agent.DQNAgent(net, epsilon_greedy, preprocessor=ptan.agent.float32_preprocessor)

# Experience source
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)

# Experience buffer
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=1000)

# Loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

exp_source.agent

best_reward = float('-inf')  # Initialize with negative infinity or other appropriate value
best_model_path = "best_model.pth"  # Define the path where you want to save the best model
checkpoint_interval = 5
current_reward = 0.0
# Training loop
for step in range(1000):  # You may want to adjust the number of steps
    buffer.populate(1)

    # Get batch from the buffer
    batch = buffer.sample(1)
    states_v, actions_v, rewards_v, dones_mask, next_states_v = unpack_batch(batch)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    q_values = net(states_v)

    # Get Q-values for taken actions
        # Get target Q-values
    target_q_values = target_net.target_model(next_states_v).max(1)[0]

    # Use view(-1) to ensure dones_mask is 1-dimensional
    dones_mask = dones_mask.view(-1)

    # Ensure target_q_values is also 1-dimensional
    target_q_values = target_q_values.view(-1)
    print(target_q_values)
    # # Update Q-values based on dones_mask
    # target_q_values[dones_mask] = 0.0

    # Detach target_q_values
    target_q_values = target_q_values.detach()

    # Calculate TD error
    expected_q_values = rewards_v + target_q_values * 0.99
    loss = loss_func(q_values, expected_q_values)

    # Backward pass
    loss.backward()

    # Optimize
    optimizer.step()


    if step % 10 == 0:
        target_net.sync()
    print(f"\n----------------------------Loss {loss.item()}------------\n")
    if step % checkpoint_interval == 0:
            # Check the performance and save the model if it's the best so far
            if current_reward > best_reward:
                best_reward = current_reward
                torch.save(net.state_dict(), best_model_path)
# After training, you can use the trained model for inference.
# For example, you can run the trained agent in the environment:
# env.close()

state = env.reset()
while True:
    env.render()
    
    # Wrap the state in a batch-like structure
    state_batch = torch.tensor([state], dtype=torch.float32)
    
    action = agent(state_batch)
    state, _, done, _ = env.step(action.item())  # Extract the action value from the tensor
    if done:
        break

env.close()




