
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
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_list = self._load_video_list()

    def _load_video_list(self):
        # Implement logic to get a list of video file paths in self.root_dir
        # For example, you might use os.listdir or glob.glob.
        # Make sure the list includes both training and testing videos.
        pass

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        video, audio, info = read_video(video_path, pts_unit='sec')

        # You can apply transformations here if needed
        if self.transform:
            video = self.transform(video)

        # Assuming a classification task, you might want to return the label
        # You need to implement the logic to extract the label from the video_path
        label = self._extract_label(video_path)

        return video, label

    def _extract_label(self, video_path):
        # Implement logic to extract the label from the video_path
        # For example, you might have a naming convention like "class1_video1.mp4"
        pass
class VideoEnv(gym.Env):
    def __init__(self, video_path):
        super(VideoEnv, self).__init__()

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)

        # Get video properties
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))
        self.channels = 3  # Assuming a color video
        self.frame_shape = (self.height, self.width, self.channels)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # Example: two possible actions
        self.observation_space = spaces.Box(low=0, high=255, shape=self.frame_shape, dtype=np.uint8)

        # Variables to track the previous frame
        self.prev_frame = None

    def step(self, action):
        # Take an action (e.g., read the next frame)
        if action == 0:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning if the video ends
                frame = self.cap.read()[1]
        else:
            frame = np.zeros_like(self.observation_space.low, dtype=np.uint8)

        # Calculate reward based on the difference between frames
        reward = self.calculate_reward(frame)

        # Check if the episode is done (you should customize this based on your task)
        done = False

        # Update the previous frame
        self.prev_frame = frame

        return frame, reward, done, {}

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning of the video
        frame = self.cap.read()[1]
        self.prev_frame = frame
        return frame

    def render(self, mode='human'):
        # Render the current frame (you can customize this based on your needs)
        cv2.imshow('VideoEnv', self.state)
        cv2.waitKey(1)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_reward(self, current_frame):
        # Example: calculate reward based on the absolute difference between frames
        if self.prev_frame is None:
            return 0.0  # No reward for the first frame

        diff = np.abs(current_frame.astype(np.float32) - self.prev_frame.astype(np.float32))
        reward = -np.mean(diff)  # Higher difference leads to a lower reward

        return reward
# Example of using the CIFAR10Env
env = CIFAR10Env(subset='train',path_to_train="")

# Reset the environment
obs = env.reset()

# Sample random actions for 10 steps
for _ in range(10):
    observation = env._get_observation()

    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")


env.observation_space.shape[0]


# Define the ViT model
class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.video_backbone = r2plus1d_18(pretrained=True)  # You can choose other video backbones
        self.fc = nn.Linear(512, 1)  # Adjust input size based on the backbone

    def forward(self, x):
        x = self.video_backbone(x)
        x = x.mean(dim=2).mean(dim=2)  # Global average pooling
        x = self.fc(x)

        # Apply ReLU activation to clip negative values
        x = F.relu(x)

        
        return x

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor(),
])
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


class LinearAlgebraActionSelector(ptan.actions.ActionSelector):
    def __call__(self, q_values):
        # Round the output to obtain integer predictions
        q_values = torch.round(q_values)
        action = np.argmax(q_values)
        return action
def custom_reward_selector(experience):
    """
    Custom reward selector function.

    Args:
        experience (list): List containing information about the episode.

    Returns:
        float: Selected reward value.
    """
    print(f"Exprinace in reward selector {experience}")
    return experience[0]



# Create environment
env = CIFAR10Env(subset="train")

# Set random seeds for reproducibility
#env.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Neural network and optimizer
net = ViT(env.observation_space.shape[0], env.action_space.n)
target_net = ptan.agent.TargetNet(net)
selector = LinearAlgebraActionSelector()
epsilon_greedy = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1, selector=selector)
agent = ptan.agent.DQNAgent(net, epsilon_greedy, preprocessor=ptan.agent.float32_preprocessor)

# Experience source
tracker = ptan.trackers.EpisodeRewardTracker(reward_selector=custom_reward_selector)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1,experience_transformer=tracker)

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
    nstate, _, done, _ = env.step(action.item())  # Extract the action value from the tensor
    exp_source.append(state=state, action=action, reward=reward, last_state=nstate, done=done)
    if done:
        break

env.close()




