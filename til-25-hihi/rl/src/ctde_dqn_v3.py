import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import imageio
import traceback # For better error logging
from collections import defaultdict, deque, namedtuple
from til_environment.gridworld import RewardNames # Assuming this import works in your environment

import functools
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from gymnasium import spaces

# --- Configuration (Adjust these as needed) ---
# Environment specific (match your game)
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Neural Network Hyperparameters
INPUT_FEATURES = 1128  # Expected output size of TrainableRLAgent.process_observation
                        # Original: 7*5*8 (viewcone) * 4 (stack size) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 384 # For Actors
HIDDEN_LAYER_2_SIZE = 384 # For Actors
HIDDEN_LAYER_3_SIZE = 256 # For Actors
OUTPUT_ACTIONS = 5        # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay (for agents we train, e.g. scout/guards)

CRITIC_HIDDEN_LAYER_1_SIZE = 512 # For Centralized Critic
CRITIC_HIDDEN_LAYER_2_SIZE = 256 # For Centralized Critic

# Training Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 32         # Minibatch size for training
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 1e-4    # Learning rate for the optimizer (can be split for actor/critic)
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4 # Often critic LR is higher, e.g., 1e-3 or 3e-4

TARGET_UPDATE_EVERY = 100 # How often to update the target critic network (in *learning steps*)
UPDATE_EVERY = 4        # How often to run a learning step (in *environment steps*)
STACK_SIZE = 4          # Used by CustomWrapper and affects INPUT_FEATURES

# Epsilon-greedy exploration parameters (for training)
EPSILON_START = 0.02
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999 # Multiplicative decay factor per episode

# PER Parameters
PER_ALPHA = 0.6  # Prioritization exponent (0 for uniform, 1 for full prioritization)
PER_BETA_START = 0.4 # Initial importance sampling exponent
PER_BETA_FRAMES = int(1e5) # Number of *environment frames/steps* over which beta is annealed to 1.0
PER_EPSILON = 1e-6 # Small constant to ensure non-zero priority

EXPLORATION_BONUS_REWARD = 0.0 # Set to 0 for CTDE unless specifically designed for multi-agent exploration

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reward shaping (ensure this matches your environment's RewardNames)
CUSTOM_REWARDS_DICT = {
    RewardNames.SCOUT_MISSION: +5.0,
    RewardNames.SCOUT_RECON: +1.0,
    RewardNames.SCOUT_TRUNCATION: +5.0,
    RewardNames.SCOUT_CAPTURED: -30.0,
    RewardNames.SCOUT_STEP: -0.02,
    RewardNames.WALL_COLLISION: -10.0,
    RewardNames.AGENT_COLLIDER: -10.0,
    RewardNames.AGENT_COLLIDEE: -10.0,
    RewardNames.STATIONARY_PENALTY: -0.2,
    RewardNames.GUARD_WINS: +10.0,
    RewardNames.GUARD_CAPTURES: +40.0,
    RewardNames.GUARD_TRUNCATION: -30.0,
    RewardNames.GUARD_STEP: -0.02,
}

# Calculate PER_BETA_FRAMES_INCREMENT based on how often learn_ctde is called
if PER_BETA_FRAMES > 0 and UPDATE_EVERY > 0:
    # Number of learning steps expected within PER_BETA_FRAMES
    num_learning_steps_for_beta_anneal = PER_BETA_FRAMES / UPDATE_EVERY
    if num_learning_steps_for_beta_anneal > 0 :
        PER_BETA_FRAMES_INCREMENT = (1.0 - PER_BETA_START) / num_learning_steps_for_beta_anneal
    else:
        PER_BETA_FRAMES_INCREMENT = 0 # Beta will not anneal if no learning steps
else:
    PER_BETA_FRAMES_INCREMENT = 0


# --- Custom Wrapper (from original file, ensure it's compatible) ---
class CustomWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        manhattan_reward_scale: float = 0.1, # Unused in provided code, but kept for signature
        stack_size: int = STACK_SIZE, # Use global STACK_SIZE
    ):
        super().__init__(env)
        self.manhattan_reward_scale = manhattan_reward_scale
        self.stack_size = stack_size
        self.viewcone_stack: dict[AgentID, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.stack_size) # Corrected: use self.stack_size
        )

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.viewcone_stack.clear() # Clear stacks for new episode
        for agent in self.agents: # self.agents are live agents after reset
            obs = super().observe(agent)
            if obs is not None and "viewcone" in obs:
                viewcone = obs["viewcone"]
                # Initialize deque with copies of the first frame
                self.viewcone_stack[agent] = deque([viewcone.copy() for _ in range(self.stack_size)], maxlen=self.stack_size)
            elif obs is not None : # if no viewcone, try to get shape for padding
                 # This case needs careful handling if viewcone isn't guaranteed
                 # For now, assume viewcone is always present if agent is active
                 print(f"Warning: Agent {agent} has no 'viewcone' in initial observation during CustomWrapper reset.")


    def step(self, action: ActionType):
        super().step(action)
        agent_just_acted = self.agent_selection # Agent that will be selected *next* by agent_iter
                                                # The observation update is for the agent whose turn it *will be*.
                                                # This needs careful thought with AEC.
                                                # PettingZoo's BaseWrapper step usually doesn't need manual obs update.
                                                # observe() is called by the user/training loop.
                                                # Let's assume viewcone_stack is updated when observe() is called.
        pass


    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None

        if "viewcone" not in obs: # If agent has no viewcone (e.g. different type or done)
            return {**obs, "stacked_viewcone": np.zeros((self.stack_size, 7, 5), dtype=np.uint8)} # Provide dummy stack

        current_viewcone = obs["viewcone"]
        
        # Initialize stack for new agent or if missing (e.g. if agent just respawned/added)
        if agent not in self.viewcone_stack or not self.viewcone_stack[agent]:
             self.viewcone_stack[agent] = deque([current_viewcone.copy() for _ in range(self.stack_size)], maxlen=self.stack_size)
        else:
            self.viewcone_stack[agent].append(current_viewcone.copy()) # Add copy to avoid issues

        stacked_viewcone_array = np.stack(list(self.viewcone_stack[agent]), axis=0)

        return {
            **obs,
            "stacked_viewcone": stacked_viewcone_array,
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID): # Use AgentID type hint
        base_space = super().observation_space(agent)
        if not isinstance(base_space, spaces.Dict):
            # This can happen if the underlying env has a simple obs space for some agents
            # Or if agent is None. For CTDE, we usually expect Dict obs.
            print(f"Warning: Base observation space for agent {agent} is not Dict: {base_space}. Stacked_viewcone cannot be added directly to non-Dict space.")
            # Fallback: return base_space, or handle as error depending on requirements
            # If the agent is one we plan to train with viewcone, this is an issue.
            # For now, let's assume trainable agents will have a Dict space where "viewcone" exists.
            # A robust solution would check if "viewcone" is in base_space.spaces.
            if "viewcone" not in base_space.spaces:
                 # This agent doesn't have a viewcone, so no stacked_viewcone.
                 # Return the original space. This is important if some agents are not image-based.
                return base_space


        viewcone_space = base_space.spaces["viewcone"]
        viewcone_shape = viewcone_space.shape # e.g. (7, 5)

        stacked_shape = (self.stack_size,) + viewcone_shape # (stack_size, 7, 5)
        stacked_viewcone_space = spaces.Box(
            low=0, # viewcone_space.low.min() if using more complex viewcones
            high=255, # viewcone_space.high.max()
            shape=stacked_shape,
            dtype=viewcone_space.dtype, # typically np.uint8
        )
        
        new_spaces = base_space.spaces.copy()
        new_spaces["stacked_viewcone"] = stacked_viewcone_space
        return spaces.Dict(new_spaces)


# --- SumTree for Prioritized Replay Buffer (from original) ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx] or self.tree[right_child_idx] == 0: # Ensure we pick left if right is empty
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

# --- CTDE Experience & Centralized Replay Buffer ---
CTDEExperience = namedtuple("CTDEExperience", field_names=[
    "global_state",          # Concatenated observations of all agents (numpy array)
    "ind_agent_states",      # Dict: {agent_id: processed_obs_agent (numpy array)}
    "joint_actions",         # Dict: {agent_id: action_taken_by_agent (int)}
    "rewards",               # Dict: {agent_id: reward_received_by_agent (float)}
    "next_global_state",     # Concatenated next observations of all agents (numpy array)
    "ind_agent_next_states", # Dict: {agent_id: next_processed_obs_agent (numpy array)}
    "dones",                 # Dict: {agent_id: done_status_for_agent (bool)}
    "episode_done"           # Bool: True if the entire episode terminated
])

class CentralizedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA, device=DEVICE):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self.device = device

    def add(self, global_state, ind_agent_states, joint_actions, rewards,
            next_global_state, ind_agent_next_states, dones, episode_done):
        experience = CTDEExperience(global_state, ind_agent_states, joint_actions, rewards,
                                    next_global_state, ind_agent_next_states, dones, episode_done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size, beta=PER_BETA_START):
        if self.tree.n_entries < batch_size: # Not enough samples for a full batch
            return None, None, None

        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_data = [None] * batch_size # Use list of Nones for easier assignment
        weights = np.empty(batch_size, dtype=np.float32)

        total_p = self.tree.total_priority
        if total_p == 0: # Avoid division by zero if all priorities are somehow zero
             print("Warning: Total priority in SumTree is 0. Cannot sample.")
             # Potentially sample uniformly if this happens, or return None
             # For now, returning None as this indicates an issue.
             return None, None, None

        priority_segment = total_p / batch_size
        
        # Get a consistent list of agent IDs from a recent experience if possible
        # This helps in structuring the batched tensors later.
        # Fallback: dynamically discover from sampled experiences.
        # For now, we'll discover from the first valid sample.
        
        temp_agent_ids = None

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, min(b, total_p - 1e-7)) # Ensure value < total_p

            index, priority, data = self.tree.get_leaf(value)
            
            if data is None: # Should ideally not happen with correct SumTree logic
                print(f"Error: SumTree.get_leaf returned None data for value {value}, index {index}. Total_p={total_p}, n_entries={self.tree.n_entries}")
                # This indicates a bug in SumTree or how it's being used.
                # Try to re-sample as a quick fix, but this needs investigation if frequent.
                value_retry = np.random.uniform(0, total_p - 1e-7)
                index, priority, data = self.tree.get_leaf(value_retry)
                if data is None:
                    print("Error: Resampling failed. Returning None from buffer sample.")
                    return None, None, None # Critical error

            sampling_probabilities = priority / total_p
            weights[i] = np.power(self.tree.n_entries * sampling_probabilities, -beta) if sampling_probabilities > 0 else 0
            batch_idx[i] = index
            batch_data[i] = data # data is a CTDEExperience namedtuple
            
            if temp_agent_ids is None and data is not None:
                temp_agent_ids = list(data.ind_agent_states.keys())


        if temp_agent_ids is None and batch_data[0] is not None: # Fallback if first was problematic
            temp_agent_ids = list(batch_data[0].ind_agent_states.keys())
        elif temp_agent_ids is None:
            print("Error: Could not determine agent IDs from sampled batch data.")
            return None, None, None # Cannot proceed without knowing agent structure


        agent_ids_in_batch_structure = sorted(list(temp_agent_ids)) # Consistent order

        weights /= (weights.max() if weights.max() > 0 else 1.0)

        # Unpack experiences into structured batches
        global_states_list = [exp.global_state for exp in batch_data]
        next_global_states_list = [exp.next_global_state for exp in batch_data]
        episode_dones_list = [exp.episode_done for exp in batch_data]

        batched_ind_agent_states = {aid: [] for aid in agent_ids_in_batch_structure}
        batched_joint_actions = {aid: [] for aid in agent_ids_in_batch_structure}
        batched_rewards = {aid: [] for aid in agent_ids_in_batch_structure}
        batched_ind_agent_next_states = {aid: [] for aid in agent_ids_in_batch_structure}
        batched_dones = {aid: [] for aid in agent_ids_in_batch_structure}

        for exp in batch_data:
            for agent_id in agent_ids_in_batch_structure:
                batched_ind_agent_states[agent_id].append(exp.ind_agent_states.get(agent_id, np.zeros(INPUT_FEATURES, dtype=np.float32))) # Pad if agent missing
                batched_joint_actions[agent_id].append(exp.joint_actions.get(agent_id, 0)) # Default action 0
                batched_rewards[agent_id].append(exp.rewards.get(agent_id, 0.0))
                batched_ind_agent_next_states[agent_id].append(exp.ind_agent_next_states.get(agent_id, np.zeros(INPUT_FEATURES, dtype=np.float32)))
                batched_dones[agent_id].append(exp.dones.get(agent_id, True)) # Default to done

        # Convert lists to tensors
        global_states_tensor = torch.from_numpy(np.array(global_states_list, dtype=np.float32)).to(self.device)
        next_global_states_tensor = torch.from_numpy(np.array(next_global_states_list, dtype=np.float32)).to(self.device)
        episode_dones_tensor = torch.from_numpy(np.array(episode_dones_list, dtype=np.uint8)).float().unsqueeze(1).to(self.device) # (batch, 1)

        for agent_id in agent_ids_in_batch_structure:
            batched_ind_agent_states[agent_id] = torch.from_numpy(np.array(batched_ind_agent_states[agent_id], dtype=np.float32)).to(self.device)
            batched_joint_actions[agent_id] = torch.from_numpy(np.array(batched_joint_actions[agent_id])).long().unsqueeze(1).to(self.device) # (batch, 1)
            batched_rewards[agent_id] = torch.from_numpy(np.array(batched_rewards[agent_id], dtype=np.float32)).float().unsqueeze(1).to(self.device) # (batch, 1)
            batched_ind_agent_next_states[agent_id] = torch.from_numpy(np.array(batched_ind_agent_next_states[agent_id], dtype=np.float32)).to(self.device)
            batched_dones[agent_id] = torch.from_numpy(np.array(batched_dones[agent_id], dtype=np.uint8)).float().unsqueeze(1).to(self.device) # (batch, 1)
        
        return (global_states_tensor, batched_ind_agent_states, batched_joint_actions,
                batched_rewards, next_global_states_tensor, batched_ind_agent_next_states,
                batched_dones, episode_dones_tensor, agent_ids_in_batch_structure), \
               batch_idx, torch.from_numpy(weights).float().to(self.device)


    def update_priorities(self, batch_indices, td_errors):
        priorities = np.abs(td_errors) + PER_EPSILON
        priorities = np.power(priorities, self.alpha)
        for idx, priority in zip(batch_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

# --- Deep Q-Network (DQN) Model (Actor Policy Network) ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# --- Centralized Critic Network (CriticDQN) ---
class CriticDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1): # Output_dim = 1 for state-value V(s_global)
        super(CriticDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x): # x is global_state
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- TrainableRLAgent (Decentralized Actor) ---
class TrainableRLAgent:
    def __init__(self, input_features, output_actions, model_load_path=None, model_save_path="trained_actor_model"):
        self.input_features = input_features
        self.output_actions = output_actions
        self.policy_net = DQN(self.input_features, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, self.output_actions).to(DEVICE)
        self.model_save_path = "ctde_model.pth"

        if model_load_path and os.path.exists(model_load_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
                print(f"Loaded pre-trained policy_net from {model_load_path}")
            except Exception as e:
                print(f"Error loading model for from {model_load_path}: {e}. Initializing with random weights.")
                self.policy_net.apply(self._initialize_weights)
        else:
            print(f"No model path or path {model_load_path} does not exist. Initializing policy_net with random weights.")
            self.policy_net.apply(self._initialize_weights)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ACTOR_LEARNING_RATE)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        tile_features = []
        tile_features.append(float(tile_value & 0b01))
        tile_features.append(float((tile_value & 0b10) >> 1))
        for i in range(2, 8): # 8 features per tile
            tile_features.append(float((tile_value >> i) & 1))
        return tile_features

    def process_observation(self, observation_dict_raw):
        # Ensure observation_dict_raw is a mutable dictionary if needed (e.g. for .get with default)
        observation_dict = dict(observation_dict_raw) if observation_dict_raw is not None else {}

        processed_features = []
        
        # Stacked Viewcone (STACK_SIZE x Height x Width)
        # CustomWrapper should provide "stacked_viewcone"
        # The shape of individual frames in viewcone (e.g. 7x5) should be fixed by env.
        viewcone_frame_height = 7 # Example, make this configurable if it can change
        viewcone_frame_width = 5  # Example

        stacked_viewcone = observation_dict.get("stacked_viewcone")

        if stacked_viewcone is not None and stacked_viewcone.shape[0] == STACK_SIZE:
            # Expected shape (STACK_SIZE, viewcone_frame_height, viewcone_frame_width)
            for frame_idx in range(STACK_SIZE):
                frame = stacked_viewcone[frame_idx]
                for r in range(viewcone_frame_height):
                    for c in range(viewcone_frame_width):
                        # Ensure r, c are within bounds of the frame (e.g. if padding was used)
                        tile_value = frame[r, c] if r < frame.shape[0] and c < frame.shape[1] else 0
                        processed_features.extend(self._unpack_viewcone_tile(tile_value))
        else: # Fallback or error: Fill with zeros for the viewcone part
            num_viewcone_features_expected = STACK_SIZE * viewcone_frame_height * viewcone_frame_width * 8
            processed_features.extend([0.0] * num_viewcone_features_expected)
            if stacked_viewcone is None:
                 print(f"Warning Agent: 'stacked_viewcone' missing in obs. Using zeros. Obs keys: {list(observation_dict.keys())}")
            else:
                 print(f"Warning Agent: 'stacked_viewcone' shape mismatch. Expected first dim {STACK_SIZE}, got {stacked_viewcone.shape}. Using zeros.")


        # Direction one-hot (4 values)
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot)

        # Location normalized (2 values)
        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y])

        # Role (e.g. "scout" flag) (1 value)
        # The key "scout" in obs_dict indicates if the agent IS the scout.
        # This should be 1.0 if this agent is scout, 0.0 otherwise.
        # The agent_id_str can be used to determine role if "scout" flag isn't directly in obs
        # Or, assume obs_dict contains a role identifier.
        # Original code had `float(observation_dict.get("scout", 0))`
        # This implies obs from env tells agent if it's a scout.
        scout_role_feature = float(observation_dict.get("scout", 0.0))
        processed_features.append(scout_role_feature)


        # Step normalized (1 value)
        step = observation_dict.get("step", 0) # Current step in episode
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        processed_features.append(norm_step)

        if len(processed_features) != self.input_features:
            raise ValueError(
                f"Agent: Feature length mismatch in process_observation. "
                f"Expected {self.input_features}, got {len(processed_features)}. "
                f"Stacked viewcone was {'present' if stacked_viewcone is not None else 'missing'}. "
                f"Obs dict keys: {list(observation_dict.keys())}"
            )
        return np.array(processed_features, dtype=np.float32)

    def select_action(self, processed_state_np, epsilon=0.0):
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(processed_state_np).float().unsqueeze(0).to(DEVICE)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.output_actions))

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Actor model saved to {self.model_save_path}")

    def reset_state(self): # For compatibility, not used in this stateless actor
        pass


# --- Main CTDE Training Loop ---
def train_ctde_agents(
        env_module,
        num_episodes=2000,
        novice_track=False,
    
        scout_model=None, 
        save_scout_model_to="trained_ctde_agent.pth", 
        guard_model=None,
        save_guard_model_to="trained_ctde_agent.pth",
    
        # actor_model_load_paths=None, # Dict: {agent_id_str: path}
        # actor_model_save_prefix="trained_ctde_actor",
    
        critic_model_load_path=None,
        save_critic_model_to="trained_ctde_critic.pth",
        render_mode=None,
        video_folder=None
):
    print(f"Using device: {DEVICE}")
    env = env_module.env(env_wrappers=[CustomWrapper], render_mode=render_mode, novice=novice_track, rewards_dict=CUSTOM_REWARDS_DICT)
    
    if render_mode == "rgb_array" and video_folder:
        os.makedirs(video_folder, exist_ok=True)
        print(f"Video folder: {video_folder}")

    env.reset() # Initialize env to get possible_agents
    possible_agent_ids = list(env.possible_agents)
    if not possible_agent_ids:
        raise ValueError("Environment returned no possible_agents after reset.")
    print(f"Training with possible agents: {possible_agent_ids}")

    # actors = {}
    
    # Loading scout and guard models
    scout = TrainableRLAgent(
        input_features=INPUT_FEATURES, # Global, from actor's process_observation
        output_actions=OUTPUT_ACTIONS,
        model_load_path=scout_model,
        model_save_path=save_scout_model_to
    )
    guard = TrainableRLAgent(
        input_features=INPUT_FEATURES, # Global, from actor's process_observation
        output_actions=OUTPUT_ACTIONS,
        model_load_path=guard_model,
        model_save_path=save_guard_model_to
    )
    
    # for agent_id_str in possible_agent_ids:
    #     load_path = actor_model_load_paths.get(agent_id_str, None) if actor_model_load_paths else None
    #     actors[agent_id_str] = TrainableRLAgent(
    #         agent_id_str=agent_id_str,
    #         input_features=INPUT_FEATURES, # Global, from actor's process_observation
    #         output_actions=OUTPUT_ACTIONS,  # Global, for actor's policy net
    #         model_load_path=load_path,
    #         model_save_path_prefix=actor_model_save_prefix
    #     )

    num_all_agents = len(possible_agent_ids)
    critic_input_dim = num_all_agents * INPUT_FEATURES # Global state is concatenation
    
    critic_net = CriticDQN(critic_input_dim, CRITIC_HIDDEN_LAYER_1_SIZE, CRITIC_HIDDEN_LAYER_2_SIZE, output_dim=1).to(DEVICE)
    target_critic_net = CriticDQN(critic_input_dim, CRITIC_HIDDEN_LAYER_1_SIZE, CRITIC_HIDDEN_LAYER_2_SIZE, output_dim=1).to(DEVICE)
    
    if critic_model_load_path and os.path.exists(critic_model_load_path):
        try:
            critic_net.load_state_dict(torch.load(critic_model_load_path, map_location=DEVICE))
            print(f"Loaded pre-trained critic_net from {critic_model_load_path}")
        except Exception as e:
            print(f"Error loading critic model from {critic_model_load_path}: {e}. Initializing random.")
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic_net.eval()
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=CRITIC_LEARNING_RATE)

    replay_buffer = CentralizedPrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA, device=DEVICE)

    scores_deque_dict = {agent_id: deque(maxlen=100) for agent_id in possible_agent_ids}
    # episode_scores_dict = {agent_id: [] for agent_id in possible_agent_ids} # For all scores, if needed for plotting

    epsilon = EPSILON_START
    total_env_steps = 0
    learning_steps_done = 0
    beta_ctde = PER_BETA_START

    for i_episode in range(1, num_episodes + 1):
        env.reset() # Also resets CustomWrapper's viewcone_stacks
        scout.reset_state()
        guard.reset_state()
        
        # for actor in actors.values(): actor.reset_state()

        current_episode_rewards_sum = {agent_id: 0 for agent_id in possible_agent_ids}
        
        # Store (s_i, a_i) for agents that have acted in current "conceptual global step"
        # and are awaiting (r_i, s'_i, d_i)
        # s_i here is the *processed* observation for agent i
        pending_current_ind_states = {} # {agent_id: processed_obs_array}
        pending_actions_taken = {}    # {agent_id: action_int}

        episode_frames = []
        should_record_video = (render_mode == "rgb_array" and video_folder and i_episode % 100 == 0)


        # Loop over agent turns within an episode using PettingZoo's agent_iter
        for agent_id_turn in env.agent_iter():
            # This agent_id_turn is now about to make a decision.
            # Its (reward_for_prev_action, observation_raw, termination, truncation) are for its *previous* action.
            observation_raw, reward_for_prev_action, termination, truncation, info = env.last()
            current_episode_rewards_sum[agent_id_turn] += reward_for_prev_action

            # --- Try to complete and store a global transition ---
            # This happens if agent_id_turn was the *last* one to act in a "round"
            # and now its (r, s') are available, completing its part of a pending global experience.
            if agent_id_turn in pending_actions_taken: # Means this agent acted previously in this global step and now we have its reward and next_state.
                prev_ind_state_for_this_agent = pending_current_ind_states.pop(agent_id_turn)
                action_taken_by_this_agent = pending_actions_taken.pop(agent_id_turn)

                # observation_raw is now s'_i for agent_id_turn
                # reward_for_prev_action is r_i for agent_id_turn
                # termination/truncation is d_i for agent_id_turn
                if observation_raw.get("scout",0)==1:
                    current_actor = scout
                else:
                    current_actor = guard
                next_ind_state_processed = np.zeros_like(prev_ind_state_for_this_agent) # Default if done
                if not (termination or truncation) and observation_raw is not None:
                     next_ind_state_processed = current_actor.process_observation(observation_raw)

                # Store these parts: s_i, a_i, r_i, s'_i, d_i for this agent
                # We need to buffer these until all agents in a "global step" have their parts.
                # Let's define `buffered_step_data` outside the loop for one global step.
                if 'buffered_step_data' not in locals() or agent_id_turn not in buffered_step_data.get('agent_ids_this_global_step', []):
                    # This indicates a new global step might be starting, or logic error.
                    # For simplicity, assume we collect one full set of (s,a,r,s',d) for all agents active at start of round.
                    pass # This part is complex. Refer to revised logic below.

            # --- Action selection and stepping environment ---
            action_to_take = None
            current_ind_state_processed_for_action = None

            if termination or truncation: # Agent is done, cannot act
                # No action needed, PZ handles this via env.step(None) or agent removal from iter
                # Ensure its pending data (if any) contributes to a final global step if it was last.
                pass
            else: # Agent is active and needs to select an action
                if observation_raw is None:
                    print(f"Error: Agent {agent_id_turn} active but got None observation. Taking random if possible.")
                    if env.action_space(agent_id_turn) is not None:
                        action_to_take = env.action_space(agent_id_turn).sample()
                else:
                    if observation_raw.get("scout",0)==1:
                        current_actor = scout
                    else:
                        current_actor = guard
                    current_ind_state_processed_for_action = current_actor.process_observation(observation_raw)
                    action_to_take = current_actor.select_action(current_ind_state_processed_for_action, epsilon)
                
                # Store s_i (current_ind_state_processed_for_action) and a_i (action_to_take)
                # These will be used when this agent_id_turn comes up *again* in agent_iter with its r, s'
                pending_current_ind_states[agent_id_turn] = current_ind_state_processed_for_action
                pending_actions_taken[agent_id_turn] = action_to_take
            
            env.step(action_to_take)
            total_env_steps += 1
            
            if should_record_video:
                try:
                    frame = env.render()
                    if frame is not None: episode_frames.append(frame)
                except Exception as e: print(f"Frame render error: {e}")


            # --- CTDE Replay Buffer Logic ---
            # A "global step" is considered complete when all agents that were alive at the beginning
            # of the current agent_iter cycle have had their (s, a, r, s', d) resolved.
            # This is when `pending_current_ind_states` becomes empty IF it was previously populated
            # by all live agents at the start of the "round".

            # More robust: After each env.step(), check if enough data is gathered for a global transition.
            # We need:
            #   Global state (s_t): Concat of ind_states *before* actions were taken.
            #   Ind actions (a_t): Actions just taken by agents.
            #   Ind rewards (r_{t+1}): Rewards received by agents *after* their actions.
            #   Ind next_states (s'_{t+1}): Next states for agents *after* their actions.
            #   Ind dones (d_{t+1}): Done status for agents *after* their actions.
            #   Episode done: Global done status.

            # This is tricky with AEC. Let's collect data per-agent over their turns.
            # `agent_transitions_this_step` stores (s_i, a_i, r_i, s'_i, d_i, episode_done_i)
            # We need a way to group these into a global (S, A_joint, R_joint, S', D_joint, Episode_Done)
            
            # Simplified CTDE buffer add:
            # When an agent `k` takes action `a_k` based on `s_k`:
            #   - `s_k` is `current_ind_state_processed_for_action`
            #   - `a_k` is `action_to_take`
            # After `env.step(a_k)`, the *next* time agent `k`'s turn comes up (or it's terminal):
            #   - `r_k` is `reward_for_prev_action`
            #   - `s'_k` is `actors[k].process_observation(new_observation_raw)`
            #   - `d_k` is `new_termination or new_truncation`
            #
            # We need to form a global state S and S' around these individual transitions.
            # S = concat of all agents' states *at the time agent k chose a_k*.
            # S' = concat of all agents' states *after agent k received r_k, s'_k*.
            # This requires snapshotting all other agents' states.

            # Alternative: Independent DQN updates for actors using critic's value (simpler for AEC)
            # If agent `k` completes transition (s_k, a_k, r_k, s'_k, d_k_agent, d_k_episode_global):
            #   Construct s_global_k (snapshot of all agent states when k was in s_k)
            #   Construct s'_global_k (snapshot of all agent states when k was in s'_k)
            #   Add (s_global_k, {k:s_k}, {k:a_k}, {k:r_k}, s'_global_k, {k:s'_k}, {k:d_k_agent}, d_k_episode_global)
            # This means buffer stores transitions focused on one agent's action but with global context.

            # Let's try a simpler buffer add:
            # If agent `X` just acted (`action_to_take`) based on `current_ind_state_processed_for_action`.
            # And the *previous* agent in `agent_iter` (let's say `Y`) has just had its outcome `(r_Y, next_obs_Y, done_Y)` revealed.
            # The transition for `Y` is `(s_Y, a_Y, r_Y, s'_Y, d_Y)`.
            # We need to construct global_S (when Y took a_Y) and global_S' (when Y observed s'_Y).

            # The current code structure `pending_current_ind_states` and `pending_actions_taken`
            # aims to collect (s,a) for all agents in a "round", then when their (r,s') come back,
            # we can form the transition.

            # For each agent `j` that has a completed (s_j, a_j, r_j, s'_j, d_j):
            #   - s_j = completed_s_j
            #   - a_j = completed_a_j
            #   - r_j = completed_r_j
            #   - s'_j = completed_s'_j
            #   - d_j = completed_d_j
            #   - episode_done = True if env.agents is empty OR global truncation signal
            #
            #   To construct global_S for this (s_j, a_j, ...):
            #     Need states of ALL other agents at the time s_j was current for agent j.
            #     This requires careful state snapshotting. CustomWrapper's observe gives current stack.
            #
            # Let's assume `initial_pending_agents_in_round` is the set of agents live at start of a cycle.
            # When all of them have their (s,a,r,s') resolved, we form ONE global experience.
            # This is complex.

            # --- Learning Step Trigger ---
            # (Moved inside the "if all agent transitions for a global step are collected" block,
            #  but that block's detailed logic is challenging with AEC for true global S, S')

            # Heuristic: If an agent completes its turn (i.e., we have s, a, r, s', d for it)
            # Form a "pseudo-global" experience where S_global is concat of current states of all agents NOW,
            # and S'_global_next is concat of their next states (some of which might be estimations if not yet observed).
            # This is common in some forms of IA2C/IDDPG adaptations.

            # Let's use the provided logic from the thought process:
            # If `agent_id_turn` just had its previous action's outcome revealed (reward, obs, done):
            #   `prev_s = pending_current_ind_states[agent_id_turn]` (from its previous turn)
            #   `prev_a = pending_actions_taken[agent_id_turn]` (from its previous turn)
            #   `rew = reward_for_prev_action`
            #   `done_agent = termination or truncation`
            #   `next_s_raw = observation_raw`
            #
            #   If `prev_s` and `prev_a` exist for `agent_id_turn`: this completes its transition.
            if agent_id_turn in pending_current_ind_states and agent_id_turn in pending_actions_taken:
                s_i = pending_current_ind_states.pop(agent_id_turn)
                a_i = pending_actions_taken.pop(agent_id_turn)
                r_i = reward_for_prev_action # Reward for (s_i, a_i)
                d_i_agent = termination or truncation # Done status for agent i after (s_i, a_i)
                
                # Process s'_i
                s_prime_i = np.zeros_like(s_i) # Default if done or obs is None
                if not d_i_agent and observation_raw is not None:
                    if observation_raw.get("scout",0) == 1:
                        s_prime_i = scout.process_observation(observation_raw)
                    else:
                        s_prime_i = guard.process_observation(observation_raw)

                # Episode done flag
                # True if all agents are done (env.agents is empty) or global truncation
                # For simplicity, if this agent_id_turn terminates, and it's the last one, episode is done.
                # Or, if PettingZoo sets a global done via info or truncation of last agent.
                # PZ standard: if agent_iter ends, episode is done.
                episode_is_done_flag = not env.agents # True if agent_iter will end after this

                # Construct global states S and S' for THIS transition (s_i, a_i, r_i, s'_i, d_i_agent)
                # S: Concatenate current `processed_observation` of all `possible_agent_ids`
                #    where for agent_i, its state is s_i. For others, it's their *current* state.
                # S': Concatenate next `processed_observation` of all `possible_agent_ids`
                #    where for agent_i, its state is s'_i. For others, it's their *next* state (if known, else current).
                # This is tricky for true CTDE. A common simplification for "concurrent" CTDE from AEC:
                # The global state is formed by all agents' current observations when agent `i` acts.
                # The next global state is formed by all agents' observations after agent `i` has received its next state.

                # For a simpler buffer add, let's assume `possible_agent_ids` for fixed order.
                # This is an approximation of global state if not all agents act simultaneously.
                
                # Global state S (at time of s_i for agent i)
                # For agent i: use s_i
                # For other agents j: use their latest known s_j (from pending_current_ind_states if they are about to act,
                # or from their s_prime if they just acted in this "conceptual" multi-agent step).
                # This is very hard to get right with AEC for a fully "correct" S_global.

                # Compromise: each agent `i`'s transition `(s_i, a_i, r_i, s'_i, d_i)` is stored along
                # with a global context `S_all_agents_at_s_i` and `S'_all_agents_at_s'_i`.
                # The replay buffer stores these tuples. The critic uses `S_all_agents` and `S'_all_agents`.
                # The actor `i` uses `s_i, a_i, r_i` and the value from `critic(S'_all_agents)`.

                # Create current global state snapshot:
                current_global_s_list = []
                current_ind_s_dict = {} # For buffer storage
                for agent_id in possible_agent_ids:
                    if agent_id == agent_id_turn:
                        obs_to_add = s_i
                    elif agent_id in pending_current_ind_states: # This other agent has obs, waiting to act
                        obs_to_add = pending_current_ind_states[agent_id]
                    else: # This other agent has already acted and its s' is not yet its new s, or it's done
                          # Use last known observation or padding. This is an approximation.
                          # Try to get current obs via env.observe() if possible, then process.
                          # This makes it very slow.
                          # Fallback: Use padding if no easy current state.
                        raw_obs_other = env.observe(agent_id) # Get current obs for others
                        if raw_obs_other is not None:
                            if observation_raw.get("scout",0) == 1:
                                obs_to_add = scout.process_observation(raw_obs_other)
                            else:
                                obs_to_add = guard.process_observation(raw_obs_other)
                        else: # Agent is done or not in env.agents
                            obs_to_add = np.zeros(INPUT_FEATURES, dtype=np.float32)
                    current_global_s_list.append(obs_to_add)
                    current_ind_s_dict[agent_id] = obs_to_add # Store all for global S context

                s_global_t = np.concatenate(current_global_s_list)

                # Create next global state snapshot (after agent_i is in s_prime_i)
                next_global_s_list = []
                next_ind_s_dict = {}
                for agent_id in possible_agent_ids:
                    if agent_id == agent_id_turn:
                        obs_to_add_next = s_prime_i
                    else:
                        # For others, their "next state" relative to s_global_t -> s_global_t+1 transition
                        # is their state *after* agent_i has moved to s_prime_i.
                        # This is effectively their current state again, unless they also moved.
                        raw_obs_other_next = env.observe(agent_id) # Get current obs for others
                        if raw_obs_other_next is not None:
                            if observation_raw.get("scout",0) == 1:
                                obs_to_add_next = scout.process_observation(raw_obs_other_next)
                            else:
                                obs_to_add_next = guard.process_observation(raw_obs_other_next)
                        else:
                            obs_to_add_next = np.zeros(INPUT_FEATURES, dtype=np.float32)
                    next_global_s_list.append(obs_to_add_next)
                    next_ind_s_dict[agent_id] = obs_to_add_next


                s_global_t_plus_1 = np.concatenate(next_global_s_list)
                
                # Store this experience (focused on agent_id_turn's action but with global context)
                # For rewards, joint_actions, dones in buffer:
                #   joint_actions: {agent_id_turn: a_i, others: can be None/default or their actual action if simultaneous}
                #   rewards: {agent_id_turn: r_i, others: their rewards if part of same conceptual step}
                #   dones: {agent_id_turn: d_i_agent, others: their dones}

                # Simplification for buffer: store agent_i's transition and the global S, S'
                # The dicts for ind_states, actions, rewards, next_states, dones in buffer
                # will then be primarily for agent_i for this specific entry.
                # When batching, we'd need to ensure this structure is handled.

                # The CTDEExperience expects full dicts.
                # Let's make them sparse for this entry, focused on agent_id_turn
                exp_ind_agent_states = {agent_id_turn: s_i}
                exp_joint_actions = {agent_id_turn: a_i}
                exp_rewards = {agent_id_turn: r_i}
                exp_ind_agent_next_states = {agent_id_turn: s_prime_i}
                exp_dones = {agent_id_turn: d_i_agent}
                
                # Populate with other agents' current states for consistent structure if needed by buffer/sampling
                for ag_id in possible_agent_ids:
                    if ag_id not in exp_ind_agent_states: # Fill current states for others for S
                        exp_ind_agent_states[ag_id] = current_ind_s_dict[ag_id]
                    if ag_id not in exp_ind_agent_next_states: # Fill next states for others for S'
                        exp_ind_agent_next_states[ag_id] = next_ind_s_dict[ag_id]
                    # Actions, rewards, dones for other agents are not directly tied to agent_id_turn's specific (s,a,r,s',d)
                    # So, keep them minimal or use defaults.
                    if ag_id not in exp_joint_actions: exp_joint_actions[ag_id] = 0 # Default action
                    if ag_id not in exp_rewards: exp_rewards[ag_id] = 0.0
                    if ag_id not in exp_dones: exp_dones[ag_id] = True # Default done


                replay_buffer.add(
                    global_state=s_global_t,
                    ind_agent_states=exp_ind_agent_states, # Dict primarily for agent_id_turn, others are context
                    joint_actions=exp_joint_actions,     # Dict primarily for agent_id_turn
                    rewards=exp_rewards,                 # Dict primarily for agent_id_turn
                    next_global_state=s_global_t_plus_1,
                    ind_agent_next_states=exp_ind_agent_next_states, # Dict primarily for agent_id_turn
                    dones=exp_dones,                     # Dict primarily for agent_id_turn's done status
                    episode_done=episode_is_done_flag
                )

                # --- Centralized Learning Step ---
                if len(replay_buffer) > BATCH_SIZE and total_env_steps % UPDATE_EVERY == 0:
                    for _ in range(1): # Number of learning updates per trigger
                        experience_batch_tuple = replay_buffer.sample(BATCH_SIZE, beta=beta_ctde)
                        if experience_batch_tuple[0] is None:
                            # print("Skipping learning: buffer sample was None.")
                            continue # Buffer not ready or error in sampling

                        experiences, batch_indices, importance_weights = experience_batch_tuple
                        
                        learn_ctde(
                            env=env,
                            scout=scout,
                            guard=guard,
                            critic_net=critic_net,
                            target_critic_net=target_critic_net,
                            critic_optimizer=critic_optimizer,
                            # Unpack experience batch tuple
                            global_states_b=experiences[0],
                            ind_agent_states_b=experiences[1],
                            joint_actions_b=experiences[2],
                            rewards_b=experiences[3],
                            next_global_states_b=experiences[4],
                            ind_agent_next_states_b=experiences[5],
                            dones_b=experiences[6], # Dict of individual agent dones
                            episode_dones_b=experiences[7], # Tensor of global episode dones
                            agent_ids_in_batch_structure=experiences[8], # List of agent_id strings
                            
                            importance_sampling_weights=importance_weights,
                            gamma=GAMMA,
                            replay_buffer=replay_buffer, # To call update_priorities
                            batch_indices=batch_indices
                        )
                        learning_steps_done += 1
                        beta_ctde = min(1.0, beta_ctde + PER_BETA_FRAMES_INCREMENT)

                # --- Update Target Critic Network ---
                if learning_steps_done > 0 and learning_steps_done % TARGET_UPDATE_EVERY == 0:
                    target_critic_net.load_state_dict(critic_net.state_dict())
                    # print(f"Target critic network updated at learning step {learning_steps_done}")


            # If all agents are done (env.agents is empty), break from agent_iter for this episode
            if not env.agents:
                break
        
        # End of episode actions
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        
        avg_scores_print = {}
        for ag_id, deq in scores_deque_dict.items():
            deq.append(current_episode_rewards_sum[ag_id])
            avg_scores_print[ag_id] = f"{np.mean(deq):.2f}" if deq else "N/A"
        
        print(f"\rEps {i_episode}, EnvSteps {total_env_steps}, LrnSteps {learning_steps_done}, Eps {epsilon:.3f}, Beta {beta_ctde:.3f} | AvgScores: {avg_scores_print}", end="")

        if i_episode % 100 == 0:
            print(f"\rEps {i_episode}, EnvSteps {total_env_steps}, LrnSteps {learning_steps_done}, Eps {epsilon:.3f}, Beta {beta_ctde:.3f} | AvgScores: {avg_scores_print}")
            
            
            scout.save_model()
            guard.save_model()
            torch.save(critic_net.state_dict(), save_critic_model_to)
            print(f"Saved critic model to {save_critic_model_to} and scout/guard models.")
            if should_record_video and episode_frames: # Save video if recorded
                video_path = os.path.join(video_folder, f"episode_{i_episode:06d}.mp4")
                try:
                    imageio.mimsave(video_path, episode_frames, fps=15)
                    print(f"Saved video: {video_path}")
                except Exception as e:
                    print(f"Video save error: {e}")
        
        # Example stopping: if scout average score is high
        # scout_id = next((id for id in possible_agent_ids if "scout" in id.lower()), None)
        # if scout_id and np.mean(scores_deque_dict[scout_id]) > 200: # Some target score
        #    print(f"Scout agent solved in {i_episode} episodes!")
        #    break

    env.close()
    print("\nTraining finished.")
    # Return episode_scores_dict if detailed scores per episode are needed

def learn_ctde(
    env, scout, guard, 
    critic_net, target_critic_net, critic_optimizer,
    global_states_b,       # Batch of global states (batch, global_feature_dim)
    ind_agent_states_b,    # Dict: {agent_id: tensor (batch, ind_feature_dim)}
    joint_actions_b,       # Dict: {agent_id: tensor (batch, 1)} long
    rewards_b,             # Dict: {agent_id: tensor (batch, 1)} float
    next_global_states_b,
    ind_agent_next_states_b,
    dones_b,               # Dict: {agent_id: tensor (batch, 1)} float (0 or 1) - individual agent done
    episode_dones_b,       # Tensor (batch, 1) float (0 or 1) - episode truly terminated
    agent_ids_in_batch_structure, # List of agent_id strings that define structure of dicts
    importance_sampling_weights,  # Tensor (batch,)
    gamma,
    replay_buffer, 
    batch_indices
):
    observation, reward, termination, truncation, info = env.last()
    # --- Critic Update (Learns V(s_global)) ---
    current_global_values_v = critic_net(global_states_b) # Shape: (batch, 1)

    with torch.no_grad():
        next_global_values_v_target = target_critic_net(next_global_states_b) # Shape: (batch, 1)
    
    # Global reward for critic: sum of individual rewards in the batch entry
    sum_rewards_for_critic_b = torch.zeros_like(episode_dones_b) # (batch, 1)
    for agent_id in agent_ids_in_batch_structure:
        if agent_id in rewards_b and rewards_b[agent_id] is not None:
             sum_rewards_for_critic_b += rewards_b[agent_id]

    # Critic Target: R_global_sum + gamma * V_target(s'_global) * (1 - episode_done)
    critic_targets_v = sum_rewards_for_critic_b + gamma * next_global_values_v_target * (1 - episode_dones_b)
    
    # Critic Loss
    critic_loss = (importance_sampling_weights.unsqueeze(1) * nn.MSELoss(reduction='none')(current_global_values_v, critic_targets_v.detach())).mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0)
    critic_optimizer.step()

    # Update PER priorities using critic's TD error
    td_errors_critic = (current_global_values_v - critic_targets_v).abs().squeeze(1).cpu().detach().numpy() # (batch,)
    replay_buffer.update_priorities(batch_indices, td_errors_critic)

    # --- Actor Updates (for each agent type/instance) ---
    for agent_id_str in agent_ids_in_batch_structure:
        if observation.get("scout",0)==1:
            actor = scout
        else:
            actor = guard

        s_i = ind_agent_states_b.get(agent_id_str)    # (batch, ind_features)
        a_i = joint_actions_b.get(agent_id_str)       # (batch, 1) long
        r_i = rewards_b.get(agent_id_str)           # (batch, 1) float
        # d_i_agent = dones_b.get(agent_id_str)     # (batch, 1) float - individual agent done

        if s_i is None or a_i is None or r_i is None : # or d_i_agent is None:
            # print(f"Skipping actor update for {agent_id_str}: missing data in batch.")
            continue
        
        # Actor's Q-values for its actions: Q_i(s_i, a_i)
        q_expected_i = actor.policy_net(s_i).gather(1, a_i) # (batch, 1)

        # Target for actor's Q-value: r_i + gamma * V_target_critic(s'_global) * (1 - episode_done)
        # V_target_critic(s'_global) is `next_global_values_v_target`
        # Use `episode_dones_b` because the future value is from the global perspective.
        
        actor_q_targets_i = r_i + gamma * next_global_values_v_target.detach() * (1 - episode_dones_b) # (batch,1)
        
        # Actor Loss for agent i
        # Importance sampling weights should apply here too.
        actor_loss_i = (importance_sampling_weights.unsqueeze(1) * nn.MSELoss(reduction='none')(q_expected_i, actor_q_targets_i.detach())).mean()
        
        actor.optimizer.zero_grad()
        actor_loss_i.backward()
        torch.nn.utils.clip_grad_norm_(actor.policy_net.parameters(), 1.0)
        actor.optimizer.step()

if __name__ == '__main__':
    import time
    start_time = time.time()
    
    # Ensure til_environment is accessible
    try:
        from til_environment import gridworld
        print("Successfully imported til_environment.gridworld")

        # Define paths for loading/saving models (example)
        # For loading, create a dictionary: {agent_id_str: path_to_model_for_that_agent}
        # Agent IDs depend on your environment, e.g., "scout_0", "guard_0", etc.
        # You need to know these from `env.possible_agents`

        critic_load_path = None # "path/to/critic_model.pth" # Optional

        train_ctde_agents(
            gridworld,
            num_episodes=50000, # Adjust as needed
            novice_track=False,  # Or True
            scout_model="agent03_stacked_merged_40k_eps.pth", 
            save_scout_model_to="agent03_ctde_90k.pth", 
            guard_model="guard03_stacked_merged_40k_eps.pth",
            save_guard_model_to="guard03_ctde_90k.pth",
            critic_model_load_path=critic_load_path,
            save_critic_model_to="ctde_critic_final.pth",
            render_mode="rgb_array", # "rgb_array" to save videos, None for no rendering
            video_folder="./ctde_videos" # Folder for videos if render_mode is rgb_array
        )

    except ImportError:
        print("Could not import 'til_environment.gridworld'.")
        print("Please ensure the environment module is correctly set up and accessible.")
        print(traceback.format_exc())
    except Exception as e:
        print(f"An error occurred during CTDE training: {e}")
        print(traceback.format_exc())

    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time):.2f} seconds.")