# Misc imports
import random
import logging
import os
from tqdm import tqdm

# Import vector handling
import numpy as np

# Import plot tooling
import matplotlib.pyplot as plt

# Import environment
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from gymnasium.wrappers.numpy_to_torch import numpy_to_torch

# Import Arcade Learning Environment
import ale_py

# Import NN training tooling
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torchinfo import summary

# Import ML training helpers (Azure)
import mlflow

# Distinguish between local and cloud environment
# AZUREML_RUN_ID only available on Azure
if "AZUREML_RUN_ID" not in os.environ:
    try:
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_experiment("DQN_PONG_EXP")
    except:
        pass

# Get the available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Azure job needs
os.makedirs('outputs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/history.log")
    ])

class InteractivePlot():
    def __init__(self):
        # Turn on interactive mode
        plt.ion()
        
        # Create axis
        _, self.ax = plt.subplots(3)

        # Line is updated dynamically
        self.line0, = self.ax[0].plot([], [], 'b-') 
        self.ax[0].set_xlabel('Episode')
        self.ax[0].set_ylabel('Cumulative reward over episodes')
        self.ax[0].set_title('Cumulative reward over episodes')

        self.line1, = self.ax[1].plot([], [], 'r-')
        self.ax[1].set_xlabel('Episode')
        self.ax[1].set_ylabel('Average action value Q')
        self.ax[1].set_title('Average action value Q')

        self.line2, = self.ax[2].plot([], [], 'r-')
        self.ax[2].set_xlabel('Episode')
        self.ax[2].set_ylabel('Loss function')
        self.ax[2].set_title('Loss function')

    def plot_stats(self, cum_reward, expl_rate, loss_fn):
        self.line0.set_xdata(np.arange(len(cum_reward)))
        self.line0.set_ydata(cum_reward)
        self.ax[0].relim()
        self.ax[0].autoscale_view()

        self.line1.set_xdata(np.arange(len(expl_rate)))
        self.line1.set_ydata(expl_rate)
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.line2.set_xdata(np.arange(len(loss_fn)))
        self.line2.set_ydata(loss_fn)
        self.ax[2].relim()
        self.ax[2].autoscale_view()

        plt.draw()
        plt.pause(0.001)

class ALE():
    def __init__(self):
        # Register environment
        gym.register_envs(ale_py)

    @numpy_to_torch.register(np.uint32)
    def _np_uint32_to_torch(value, device=None):
        # Cast Gymnasium's np.uint32 (np_random_seed) to an int64 tensor to accept the seed as int
        return torch.tensor(int(value), dtype=torch.int64, device=device)

    def setupENV(self):
        # No seed is set
        env = gym.make(
            "ALE/Pong-v5", 
            obs_type="rgb", # dimensionality: 210 x 160 x 3
            frameskip = 1, # frame skip is made by preprocessing function Φ
            repeat_action_probability = 0. # only deterministic actions
            ) 

        # Preprocessing function Φ
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True
            )

        # Enable frame stacking
        # The state dimensionality becomes 84 x 84 x 4
        env = FrameStackObservation(env, stack_size=4)

        # Convert the native Numpy environment to a Torch-compatible version and push tensors to the device
        env = NumpyToTorch(env, 'cpu')

        # Record episode statistics
        env = RecordEpisodeStatistics(env)

        return env

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # Define CNN's architecture
        self.layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), # -> 64*7*7 = 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) # 6 valid actions in Pong acc. to ALE
        )

    def forward(self, x):
        return self.layer(x)

class ExperienceReplay():
    def __init__(self, buffer_size: int = int(1e6), minibatch_size: int = 32, min_samples_in_buffer: int = 1000):
        # Set the buffer parameters
        self.buffer_size = buffer_size
        self.buffer_pos_idx = 0
        self.num_inrush_samples = 0

        # The buffer D will store the #n most recent the experience tuples
        self.storage = np.zeros(self.buffer_size, dtype=object) # Immediatly allocate

        # Set the replay parameters
        self.minibatch_size = minibatch_size
        self.min_samples_in_buffer = min_samples_in_buffer # Buffer must have made #n many experiences

    def reset(self):
        self.storage = np.zeros(self.buffer_size, dtype=object)
        self.buffer_pos_idx = 0
        self.num_inrush_samples = 0

    def is_buffer_warmed_up(self):
        # Assure that buffer contains #n many samples already
        return self.num_inrush_samples > self.min_samples_in_buffer

    def push(self, experience):
        # Push the most recent sars tuple into circular buffer
        if self.buffer_pos_idx >= self.buffer_size:
            # Buffer is full, so tuple must replace oldest previous tuple in ring buffer
            self.buffer_pos_idx = 0
        else:
            # Buffer is vacant - Index must stay
            pass

        # Place new experience tuple into buffer
        self.storage[self.buffer_pos_idx] = (self.buffer_pos_idx, experience)

        # Count buffer index
        self.buffer_pos_idx += 1

        # Count the total number of tuples put into the buffer
        self.num_inrush_samples += 1

    def buffer_length(self):
        return min(self.num_inrush_samples, self.buffer_size)
        
    def random_sampling(self):
        # Naive sampling - sample #n many minibatches with random samples from buffer
        if self.num_inrush_samples > self.minibatch_size:
          	# Sampling is allowed iff buffer is already situated with at least #minibacth_size samples
            current_size = self.buffer_length()
            minibatches = random.sample(list(self.storage[0:current_size]), k=self.minibatch_size)
            return minibatches
        else:
            # Sampling not allowed due to insufficient sample size in buffer
            return []

class QLearning():
    def __init__(self, env, n_actions: int = 6, epsilon: list = [1.0, 0.1, 1e6], episodes: int = 1000, DF: float = 0.99, LR: list = [1e-4, 1e-5, 2e6], momentum: float = 1e-4, momentum_squared: float = 1e-4, min_squared_grad: float = 1e-4, target_update_rate: int = 10000, replay_buffer_size: int = int(1e6), minibatch_size: int = 32, min_samples_in_buffer: int = 1000, target_variant: str = "DQN"):
        # Situate the interactive plotter
        self.ipl = InteractivePlot()

        # Helpers for plot/logging
        self.avg_action_value_list = []
        self.cum_reward_list = []
        self.sum_loss = []
        self.cum_reward_avg = 0
        self.cum_cnt = 0
        self.log = open('history.log', 'a')

        # Create the CNNs for target and online DQN
        self.online_DQN = DQN(n_actions=n_actions).to(device)
        self.target_DQN = DQN(n_actions=n_actions).to(device)

        # Initialize from weights if possible
        try:
            checkpoint = torch.load('checkpoint.pth', map_location=torch.device(device))
            self.online_DQN.load_state_dict(checkpoint['online_dqn'])
            self.target_DQN.load_state_dict(checkpoint['target_dqn'])
            print('Checkpoint loaded')
        except:
            # No proper file in place
            print('No checkpoint loaded')

        # Set the training environment
        self.optimizer = torch.optim.RMSprop(self.online_DQN.parameters(), lr=LR[0], momentum=momentum, alpha=momentum_squared, eps=min_squared_grad) # As proposed in the original paper
        self.loss = nn.HuberLoss(reduction='none') # Using Huber instead of vanilla MSE loss for improved stability
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=LR[1], total_iters=LR[2])

        # Define environment
        self.env = env

        # Situate the replay buffer for experience replay
        self.minibatch_size = minibatch_size
        self.experience = ExperienceReplay(buffer_size=replay_buffer_size, minibatch_size=self.minibatch_size, min_samples_in_buffer=min_samples_in_buffer)

        # Define hyperparameters
        self.epsilon = epsilon
        self.episodes = episodes
        self.discount_factor = DF
        self.target_update_rate = target_update_rate
        self.target_variant = target_variant

        # Initialize target with same weights
        self.target_DQN.load_state_dict(self.online_DQN.state_dict())

        # Initialize a counter for updating the target network from time to time only
        self.update_cnt = 0

    def store_model_parameters(self, checkpoint):
        # Set the path
        ckpt_path = f"outputs/checkpoint_{checkpoint}.pth"

        # Store the parameters to checkpoint
        torch.save({
            'online_dqn': self.online_DQN.state_dict(),
            'target_dqn': self.target_DQN.state_dict()
            }, ckpt_path)
        
        # Store using mlflow for Azure
        try:
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        except:
            pass

    def epsilon_schedule(self, info):
        # Returns a decaying epsilon
        # Get current frame from environment info signal
        current_frame_num = info.get('frame_number')

        # Split decay hyperparameters
        epsilon_start = self.epsilon[0]
        epsilon_end = self.epsilon[1]
        max_frames_decay = self.epsilon[2]

        # Decay epsilon value
        if current_frame_num <= max_frames_decay:
            # Decay epsilon linearly
            epsilon = ((epsilon_end - epsilon_start) / max_frames_decay) * current_frame_num + epsilon_start
        else:
            # Fix epsilon if threshold is reached
            epsilon = epsilon_end

        return epsilon

    def printDQN(self):
        # Print the summary of CNN architecture given the expected input shape
        summary(self.online_DQN, input_size=(1, 4, 84, 84)) # N, C, H, W

    def policy(self, state, info):
        # Implements epsilon greedy policy
        if np.random.uniform(0,1) < self.epsilon_schedule(info):
            # Random action
            action = self.env.action_space.sample()
        else:
            # Greedy action
            with torch.no_grad():
                # DQN shape (1, n_actions)
                action = torch.argmax(self.getQ(state.unsqueeze(0)), dim=1).item()
        return action
    
    def norm(self, state):
        # Normalizes the value range to [0,1]
        return state / 255.0
    
    def getQ(self, state):
        q_vals_per_action = self.online_DQN(self.norm(state.to(device)))
        return q_vals_per_action
    
    def trainDQN(self):
        # Train the actual CNN
        # Train the CNN only if there is a sufficient number of samples stored in the buffer
        if not self.experience.is_buffer_warmed_up():
            return
        
        # Take a single mini batch from experience replay buffer
        # Use random sampling
        minibatch = self.experience.random_sampling()

        # Enable training for online parameters
        self.online_DQN.train(True)

        # Enable gradient updates
        with torch.set_grad_enabled(True):

            # Experience tuples
            (id),(experience) = zip(*minibatch)
            state, action, reward, next_state, terminated, truncated = zip(*experience)

            # Convert to correct shapes
            state = torch.stack(state).float().to(device)
            action = torch.tensor(action, dtype=torch.long).to(device)
            reward = torch.tensor(reward, dtype=torch.float).to(device)
            next_state = torch.stack(next_state).float().to(device)
            done = torch.tensor([term or trunc for term, trunc in zip(terminated, truncated)], dtype=torch.float).to(device)

            # Infer the current Q(s,a;θ) function from online network
            Q_val_fun = self.getQ(state).gather(1, action.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Target: y = r + gamma * max_a Q_target(s', a') * (1 - done)
                # (1 - done): Set target to reward for terminal state, otherwise estimate Q value function
                if self.target_variant == "DQN":
                    y = reward + self.discount_factor * self.target_DQN(self.norm(next_state)).max(dim=1)[0] * (1.0 - done)
                elif self.target_variant == "DDQN":
                    # Decoupling of the action decision from the evaluation of the actual action value
                    action = torch.argmax(self.getQ(next_state), dim=1)
                    y = reward + self.discount_factor * self.target_DQN(self.norm(next_state)).gather(1, action.unsqueeze(1)).squeeze(1) * (1.0 - done)

            # Estimate the actual loss between target and Q(s,a;θ)
            # Reduction is disabled, thus returns the loss for each sample in batch
            per_sample_loss = self.loss(Q_val_fun, y)

            # Prepare for experience replay
            with torch.no_grad():
                # Update experience tuple
                for idx, e in enumerate(id):
                    self.experience.storage[e] = (e, (state[idx].detach().cpu(), action[idx].detach().cpu(), reward[idx].detach().cpu(), next_state[idx].detach().cpu(), terminated[idx], truncated[idx]))

            # Estimate mean
            loss = (per_sample_loss).mean()

            # Reset optimizer
            self.optimizer.zero_grad()

            # Apply chain rule
            loss.backward()

            # In-place gradient clipping [-1,1]
            torch.nn.utils.clip_grad_value_(self.online_DQN.parameters(), 1)

            # Do a gradient step
            self.optimizer.step()

            # Do a scheduler step to decay LR furthermore
            self.scheduler.step()

            # Append loss
            self.sum_loss.append(loss.item())

        # Update frequency must align to parameter update rate
        self.update_cnt += 1

        # Update the target network from time to time utilizing the online network parameters
        if self.update_cnt > self.target_update_rate:
            self.update_cnt = 0
            self.target_DQN.load_state_dict(self.online_DQN.state_dict())

    def runQLearning(self):
        # Run the actual training for #j episodes
        for j in tqdm(range(self.episodes)):

            # Reset environment (frame_number is stored permanently along all episodes)
            state, info = self.env.reset(seed=1234)

            # Reset termination state flag
            terminated = False

            # Reset time out state flag
            truncated = False

            # Helper
            cnt = 0

            # The average action value
            avg_action_value = 0

            # Run training episode until termination
            while not (terminated):
                # Sample random action from action space
                action = self.policy(state, info)

                # Take action in environment and retrieve next state
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Add experience to replay buffer for experience replay
                self.experience.push((state.detach().cpu(), action, reward, next_state.detach().cpu(), terminated, truncated))

                # Run actual DQN (CNN) training with experience gathered so far
                self.trainDQN()

                # Off policy - Actions taken independently
                state = next_state

                # Estimate the average action value
                with torch.no_grad():
                    cnt += 1
                    action_value = self.getQ(state.unsqueeze(0))[0, action].item()
                    avg_action_value = avg_action_value + (action_value - avg_action_value) / cnt
            
            # Collect average action value per episode
            self.avg_action_value_list.append(avg_action_value)

            # Collect cumulative reward per episode
            self.cum_reward_list.append(info.get('episode').get('r'))

            # Estimate average of cumulative reward along entire training
            self.cum_cnt += 1
            self.cum_reward_avg = self.cum_reward_avg + (info.get('episode').get('r') - self.cum_reward_avg) / self.cum_cnt

            # Print some logs during training
            tqdm.write(f"#{j}/#Frame {info.get('frame_number')}: Avg_Q_val: {avg_action_value}; Cum. reward: {info.get('episode').get('r')}/Avg: {self.cum_reward_avg}; Eps: {self.epsilon_schedule(info)}; Loss: {np.mean(self.sum_loss)}")

            # Merely used when training is done on Azure
            try:
                mlflow.log_metric("Cumulative Episode Reward", info.get('episode').get('r'), step=j)
                mlflow.log_metric("Cumulative Average Reward", self.cum_reward_avg, step=j)
                mlflow.log_metric("Current Frame Number", int(info.get('frame_number')), step=j)
                mlflow.log_metric("Average Action Value", avg_action_value, step=j)
                mlflow.log_metric("Current Exploration Rate", self.epsilon_schedule(info), step=j)
                mlflow.log_metric("Training Loss", np.mean(self.sum_loss), step=j)
            except:
                pass

            # Write log messages
            self.log.write(str((j, int(info.get('frame_number')), avg_action_value, info.get('episode').get('r'), self.cum_reward_avg, float(self.epsilon_schedule(info)), np.mean(self.sum_loss)))+'\n')
            self.log.flush()

            # Update the interactive plot
            self.ipl.plot_stats(self.cum_reward_list, self.avg_action_value_list, self.sum_loss)

            # Store the recent learned model parameters
            if j % 10 == 0:
                self.store_model_parameters(j)
        
        # When training is finished the env must be closed
        self.env.close()

        # End MLFlow
        try:
            mlflow.end_run()
        except:
            pass

    def rollout(self):
        # Performs the rollout of the trained policy
        cum_reward_mean = 0
        cnt = 0
        actions = []

        # Run the actual training for #j episodes
        for j in range(100):

            # Reset environment (frame_number is stored permanently along all episodes)
            state, info = self.env.reset()

            # Reset termination state flag
            terminated = False

            # Reset time out state flag
            truncated = False

            # Run training episode until termination
            while not (terminated or truncated):
                
                # Get greedy action from trained DQN
                with torch.no_grad():
                    # DQN shape (1, n_actions)
                    action = torch.argmax(self.getQ(state.unsqueeze(0)), dim=1).item()

                    # Store action for insights
                    actions.append(action)

                # Take action in environment and retrieve next state
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Off policy - Actions taken independently
                state = next_state

            cnt += 1
            cum_reward_mean = cum_reward_mean + (info.get('episode').get('r') - cum_reward_mean) / cnt
            print(f"Cum. reward: {info.get('episode').get('r')} / Mean: {cum_reward_mean}")
        
        # When training is finished the env must be closed
        self.env.close()


ALEPY = ALE()

# Epsilon decay schedule
eps_start = 1 # Epsilon start value
eps_end = 0.1 # Epsilon end value
eps_threshold_frame = int(1e6) # Frame threshold value when epsilon end value becomes active

# Decay learning rate for Q-Learning
lr_start = 2.5e-4
lr_end = 2.5e-5
lr_threshold = 2000000 / 4
end_factor   = lr_end / lr_start # Schedule only takes factors

QL = QLearning(
    env = ALEPY.setupENV(), # Hand over the actual ALE environment
    n_actions = 6, # Typically, Atari Pong has 6 admissable actions
    epsilon = [eps_start, eps_end, eps_threshold_frame], # ε annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    episodes = 1000, # Number of training episodes in the MDP
    DF = 0.99, # Discount factor
    LR = [lr_start, end_factor, lr_threshold], # Learning rate
    momentum = 0.95, # Gradient momentum
    momentum_squared = 0.95, # Gradient momentum decay rate
    min_squared_grad = 0.01, # Min squared gradient
    target_update_rate = 10000/4, # Update rate at which target network parameters θ- are updated from online network parameters θ
    replay_buffer_size = int(1e6), # Size of experience tuples stored in the replay buffer
    minibatch_size = 32, # Batch size for SGD in DQN training
    min_samples_in_buffer = 50000/4, # Same as proposed in the paper
    target_variant = "DQN" # Select target variant for training
    )

QL.printDQN()

#QL.runQLearning()
QL.rollout()