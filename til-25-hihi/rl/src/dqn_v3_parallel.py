import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import imageio
from collections import defaultdict, deque, namedtuple
from til_environment.gridworld import RewardNames

import functools
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from gymnasium import spaces

import multiprocessing as mp
import traceback # For debugging in worker
import time # For timing

# --- Map configurations ---
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Neural Network Hyperparameters
INPUT_FEATURES = 1128  # 7*5*8 (viewcone) * 4 (stack size) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 384
HIDDEN_LAYER_2_SIZE = 384
HIDDEN_LAYER_3_SIZE = 256
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Training Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 32         # Minibatch size for training
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 1e-4    # Learning rate for the optimizer
TARGET_UPDATE_EVERY = 100 # How often to update the target network (in policy agent steps)
UPDATE_EVERY = 4        # How often to run a learning step (in policy agent steps)
STACK_SIZE = 4

# Epsilon-greedy exploration parameters (for training)
EPSILON_START = 0.015
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999 # Multiplicative decay factor per episode/fixed number of steps

# PER Parameters
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = int(1e5) # Number of learning steps over which beta is annealed to 1.0
PER_EPSILON = 1e-6

EXPLORATION_BONUS_REWARD = 0.5 # Exploration bonus

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reward shaping
CUSTOM_REWARDS_DICT = {
    RewardNames.SCOUT_MISSION: +5.0,
    RewardNames.SCOUT_RECON: +1.0,
    RewardNames.SCOUT_TRUNCATION: +5.0,
    RewardNames.SCOUT_CAPTURED: -40.0,
    RewardNames.SCOUT_STEP: -0.02,
    RewardNames.WALL_COLLISION: -10.0,
    RewardNames.AGENT_COLLIDER: -10.0,
    RewardNames.AGENT_COLLIDEE: -10.0,
    RewardNames.STATIONARY_PENALTY: -0.2,
    RewardNames.GUARD_WINS: +10.0,
    RewardNames.GUARD_CAPTURES: +40.0,
    RewardNames.GUARD_TRUNCATION: -15.0,
    RewardNames.GUARD_STEP: -0.02,
}

class CustomWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        manhattan_reward_scale: float = 0.1, # Example, not used in current DQN
        stack_size: int = 4,
    ):
        super().__init__(env)
        self.manhattan_reward_scale = manhattan_reward_scale
        self.stack_size = stack_size
        self.viewcone_stack: dict[AgentID, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.stack_size) # Use self.stack_size
        )

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        for agent_id in self.possible_agents: # Initialize for all possible agents
            # Create a dummy observation to get viewcone shape if needed
            # Or, ensure observe(agent_id) returns a valid obs after reset
            # For now, we will populate on first real observe
            self.viewcone_stack[agent_id] = deque([np.zeros((7,5), dtype=np.uint8)] * self.stack_size, maxlen=self.stack_size)

        # Initialize frame stacks for all agents by calling observe for each
        # This requires the underlying env to be in a state where observe can be called for each agent after reset
        # PettingZoo's AECEnv standard is that after reset, observe(first_agent_to_act) is valid.
        # We'll populate stacks lazily or ensure first observation fills them.
        # The original code populated on first observe in step, which is safer.
        # Let's stick to initializing with zeros, and real data will fill in.

    def _ensure_stack_initialized(self, agent_id, current_viewcone):
        if not self.viewcone_stack[agent_id] or \
           all(np.array_equal(frame, np.zeros_like(current_viewcone)) for frame in self.viewcone_stack[agent_id]):
            self.viewcone_stack[agent_id] = deque([current_viewcone] * self.stack_size, maxlen=self.stack_size)


    def step(self, action: ActionType):
        super().step(action)
        agent = self.agent_selection # Agent whose turn it WILL BE next (after this step call completes for current agent)
                                    # Or agent who JUST acted. Depends on when observe is called.
                                    # For observe(), it's usually the agent whose turn it is.
        # The observation for an agent is typically available *before* its step.
        # Let's update the stack based on the observation that will be *returned* for the current agent.
        # This means observe() should handle the update.

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None

        current_viewcone = obs["viewcone"]
        self._ensure_stack_initialized(agent, current_viewcone) # Initialize if first time or all zeros
        self.viewcone_stack[agent].append(current_viewcone)

        stacked_viewcone = np.stack(list(self.viewcone_stack[agent]), axis=0)

        return {
            **obs,
            "stacked_viewcone": stacked_viewcone,
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base_space = super().observation_space(agent)
        if isinstance(base_space, spaces.Dict) and "viewcone" in base_space.spaces:
            viewcone_shape = base_space.spaces["viewcone"].shape
            viewcone_dtype = base_space.spaces["viewcone"].dtype # Important for Box space

            stacked_shape = (self.stack_size,) + viewcone_shape
            stacked_viewcone_space = spaces.Box(
                low=0,
                high=np.iinfo(viewcone_dtype).max if np.issubdtype(viewcone_dtype, np.integer) else np.inf, # Use dtype max
                shape=stacked_shape,
                dtype=viewcone_dtype, # Use original dtype
            )
            new_spaces = {**base_space.spaces, "stacked_viewcone": stacked_viewcone_space}
            return spaces.Dict(new_spaces)
        return base_space


# --- SumTree for Prioritized Replay Buffer ---
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
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(self.max_priority**self.alpha, experience) # Add with priority powered by alpha

    def sample(self, batch_size, beta=PER_BETA_START):
        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_data = np.empty(batch_size, dtype=object)
        weights = np.empty(batch_size, dtype=np.float32)
        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            
            sampling_probabilities = priority / self.tree.total_priority
            weights[i] = np.power(self.tree.n_entries * sampling_probabilities, -beta)
            batch_idx[i] = index
            batch_data[i] = data
        
        weights /= weights.max()

        # Ensure data is not None before processing
        valid_batch_data = [e for e in batch_data if e is not None]
        if len(valid_batch_data) != batch_size:
             # This can happen if buffer is not full enough or has None entries
             # For now, we'll proceed, but ideally, sample only when enough valid data
             # print(f"Warning: Sampled batch contains {batch_size - len(valid_batch_data)} None entries.")
             if not valid_batch_data: # All None, cannot proceed
                 return None, None, None


        states, actions, rewards, next_states, dones = zip(*[e for e in valid_batch_data])

        states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones), batch_idx[:len(valid_batch_data)], torch.from_numpy(weights[:len(valid_batch_data)]).float().to(DEVICE)


    def update_priorities(self, batch_indices, td_errors):
        priorities = np.abs(td_errors) + PER_EPSILON
        priorities = np.power(priorities, self.alpha)
        for idx, priority in zip(batch_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

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

class TrainableRLAgent:
    def __init__(self, model_load_path=None, model_save_path="trained_dqn_model.pth", input_features=INPUT_FEATURES, output_actions=OUTPUT_ACTIONS, is_learner=True):
        self.device = DEVICE
        self.is_learner = is_learner # To control prints and optimizer init

        self.policy_net = DQN(input_features, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, output_actions).to(self.device)
        self.target_net = DQN(input_features, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, output_actions).to(self.device)
        
        if model_load_path and os.path.exists(model_load_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_load_path, map_location=self.device))
                if self.is_learner: print(f"Learner loaded pre-trained policy_net from {model_load_path}")
            except Exception as e:
                if self.is_learner: print(f"Learner error loading model from {model_load_path}: {e}. Initializing with random weights.")
                self.policy_net.apply(self._initialize_weights)
        else:
            if self.is_learner: print(f"Learner: No model path {model_load_path} or DNE. Initializing policy_net with random weights.")
            self.policy_net.apply(self._initialize_weights)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.is_learner:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
            self.model_save_path = model_save_path
            self.t_step = 0
            self.beta = PER_BETA_START
            # Beta increment per learning step
            self.beta_increment_per_learning_step = (1.0 - PER_BETA_START) / PER_BETA_FRAMES if PER_BETA_FRAMES > 0 else 0
            self.global_step_counter = 0 # Tracks learning steps or experiences processed by learner

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) 
        tile_features.append(float((tile_value & 0b10) >> 1))
        for i in range(2, 8):
            tile_features.append(float((tile_value >> i) & 1))
        return tile_features

    def process_observation(self, observation_dict):
        processed_features = []
        stacked_viewcone = observation_dict.get("stacked_viewcone", None)

        if stacked_viewcone is not None:
            for frame_idx in range(STACK_SIZE): # Iterate up to STACK_SIZE
                frame = stacked_viewcone[frame_idx]
                for r in range(7):
                    for c in range(5):
                        # Check bounds for safety, though viewcone should be 7x5
                        tile_value = frame[r][c] if r < len(frame) and c < len(frame[r]) else 0
                        processed_features.extend(self._unpack_viewcone_tile(tile_value))
        else: # Fallback or if STACK_SIZE is 1
            # This case should ideally not happen if CustomWrapper is used and STACK_SIZE > 1
            # If it does, we pad to maintain INPUT_FEATURES length.
            # For now, assume stacked_viewcone is correctly provided by CustomWrapper.
            # If not, a ValueError will be raised later.
            # Fallback for single frame if STACK_SIZE = 1 or no stack
            viewcone = observation_dict.get("viewcone", np.zeros((7,5), dtype=np.uint8)) # Default to zeros
            num_frames_to_process = 1 # STACK_SIZE if STACK_SIZE == 1 else 1
            for _ in range(num_frames_to_process): # Simulate stack if STACK_SIZE is 1
                for r in range(7):
                    for c in range(5):
                        tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0
                        processed_features.extend(self._unpack_viewcone_tile(tile_value))

            # If STACK_SIZE > 1 and no stacked_viewcone, fill remaining features with zeros
            if STACK_SIZE > 1 and (stacked_viewcone is None or stacked_viewcone.shape[0] != STACK_SIZE):
                # This path handles the case where CustomWrapper might not have run or STACK_SIZE=1 was intended for INPUT_FEATURES
                # but the global STACK_SIZE is > 1. For safety, we ensure the feature vector is the correct size.
                # This indicates a mismatch in expectation vs. what's provided.
                # The most robust way is that CustomWrapper *always* provides the correctly stacked viewcone.
                # If it's not present, we create zero padding for the missing frames.

                # Calculate features for one frame
                single_frame_features_len = 7 * 5 * 8
                # Pad remaining frames with zeros
                # (Current processed_features has 1 frame)
                # We need (STACK_SIZE - 1) more frames of zeros
                if not processed_features : # if viewcone was also empty
                    processed_features.extend([0.0] * single_frame_features_len)


                remaining_frames_to_pad = STACK_SIZE - 1 # Assuming 1 frame was processed above
                processed_features.extend([0.0] * (remaining_frames_to_pad * single_frame_features_len) )


        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y])

        scout_role = float(observation_dict.get("scout", 0))
        processed_features.append(scout_role)

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        processed_features.append(norm_step)

        if len(processed_features) != INPUT_FEATURES:
            # This is a critical error if it happens.
            # It means observation processing is not creating the fixed-size vector the NN expects.
            # This can happen if STACK_SIZE > 1 but stacked_viewcone is not provided or has wrong dimensions.
            error_message = (
                f"Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}. "
                f"Stacked_viewcone shape: {stacked_viewcone.shape if stacked_viewcone is not None else 'None'}. "
                f"Global STACK_SIZE: {STACK_SIZE}."
            )
            # In a worker, this should not crash the whole pool. Log and return zeros or raise specific exception.
            # For now, let it raise to highlight the issue during debugging.
            raise ValueError(error_message)
        return np.array(processed_features, dtype=np.float32)


    def select_action(self, state_np, epsilon=0.0):
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            # self.policy_net.train() # Only set to train if it's the learner's policy_net
            if self.is_learner: self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(OUTPUT_ACTIONS))

    def add_experience(self, state, action, reward, next_state, done):
        if not self.is_learner: return # Only learner has memory
        self.memory.add(state, action, reward, next_state, done)

    def learn(self, experiences, indices, importance_sampling_weights, gamma):
        if not self.is_learner: return
        states, actions, rewards, next_states, dones = experiences

        q_next_policy_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.target_net(next_states).detach().gather(1, q_next_policy_actions)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.policy_net(states).gather(1, actions)

        td_errors = (q_targets - q_expected).abs().cpu().detach().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        loss = (importance_sampling_weights * nn.MSELoss(reduction='none')(q_expected, q_targets)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_learning_step)


    def update_target_net(self):
        if not self.is_learner: return
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        if not self.is_learner or not self.model_save_path: return
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Learner model saved to {self.model_save_path}")

    def get_policy_net_state_dict_cpu(self):
        self.policy_net.to('cpu')
        state_dict = self.policy_net.state_dict()
        self.policy_net.to(self.device)
        return state_dict

    def load_policy_net_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(self.device)
        # For workers, ensure it's in eval mode after loading. Learner will manage its mode.
        if not self.is_learner: self.policy_net.eval()
        else: self.policy_net.train()

    def reset_state(self): pass


# --- Worker Function ---
def run_episode_worker(
    worker_id,
    env_module_name,
    env_config,
    initial_model_paths,
    experience_queue,
    model_update_pipe_recv,
    stop_event,
    worker_epsilon_queue,
    global_episode_completed_counter,
    video_config_worker
):
    if env_module_name == "til_environment.gridworld":
        from til_environment import gridworld as env_module
    else:
        raise ImportError(f"Unknown environment module: {env_module_name}")

    scout_policy_agent = TrainableRLAgent(model_load_path=initial_model_paths['scout_model_path'], is_learner=False)
    guard_policy_agent = TrainableRLAgent(model_load_path=initial_model_paths['guard_model_path'], is_learner=False)

    env = env_module.env(
        env_wrappers=[CustomWrapper],
        render_mode=env_config['render_mode_worker'],
        novice=env_config['novice_track'],
        rewards_dict=env_config['rewards_dict']
    )

    current_epsilon = EPSILON_START
    local_episode_count = 0

    while not stop_event.is_set():
        local_episode_count += 1
        with global_episode_completed_counter.get_lock():
            current_global_episode_num = global_episode_completed_counter.value

        if model_update_pipe_recv.poll():
            try:
                scout_state_dict, guard_state_dict = model_update_pipe_recv.recv()
                if scout_state_dict:
                    scout_policy_agent.load_policy_net_state_dict(scout_state_dict)
                if guard_state_dict:
                    guard_policy_agent.load_policy_net_state_dict(guard_state_dict)
            except (EOFError, BrokenPipeError):
                break
            except Exception as e:
                print(f"Worker {worker_id} error receiving model: {e}")

        try:
            current_epsilon = worker_epsilon_queue.get_nowait()
        except mp.queues.Empty:
            pass

        env.reset(seed=worker_id + current_global_episode_num)

        current_rewards_this_episode = {ag_id: 0 for ag_id in env.possible_agents}
        pending_step_data_scout = {}
        pending_step_data_guard = {}

        # --- NEW: Track visited locations per guard ---
        visited_guard_locations = defaultdict(set)  # Key: guard_id, Value: set of (x, y) tuples

        episode_frames = []
        should_record_video_worker = (env_config['render_mode_worker'] == "rgb_array" and
                                      video_config_worker['video_folder'] and
                                      (current_global_episode_num + 1) % video_config_worker['record_interval'] == 0)

        scout_agent_id_identified = None

        for agent_id in env.agent_iter():
            observation, raw_reward_from_env, termination, truncation, info = env.last()

            if should_record_video_worker:
                try:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                except Exception as e:
                    print(f"Worker {worker_id} frame render error: {e}")

            for live_agent_id in env.agents:
                current_rewards_this_episode[live_agent_id] += env.rewards.get(live_agent_id, 0)

            done = termination or truncation
            action_to_take = None

            is_scout_agent = observation and observation.get("scout") == 1
            if is_scout_agent and scout_agent_id_identified is None:
                scout_agent_id_identified = agent_id

            agent_type_str = 'scout' if is_scout_agent else 'guard'
            pending_this_agent_step_data = pending_step_data_scout if is_scout_agent else pending_step_data_guard
            policy_agent_for_action = scout_policy_agent if is_scout_agent else guard_policy_agent

            # --- Finalize and send previous transition ---
            if agent_id in pending_this_agent_step_data:
                prev_state_np, prev_action = pending_this_agent_step_data.pop(agent_id)
                reward_for_prev_action = raw_reward_from_env
                next_state_np = None
                current_exploration_bonus = 0.0

                if not done and observation is not None:
                    obs_dict_current = {
                        k: v if isinstance(v, (int, float, bool)) else (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in observation.items()
                    }
                    try:
                        next_state_np = policy_agent_for_action.process_observation(obs_dict_current)

                        # --- Apply exploration bonus for guards only ---
                        if not is_scout_agent:
                            location = tuple(obs_dict_current.get("location", [None, None]))
                            if location != (None, None) and location not in visited_guard_locations[agent_id]:
                                visited_guard_locations[agent_id].add(location)
                                current_exploration_bonus = EXPLORATION_BONUS_REWARD
                    except ValueError as e:
                        print(f"Worker {worker_id} obs processing error: {e}")
                        continue
                else:
                    next_state_np = np.zeros_like(prev_state_np)

                final_reward = reward_for_prev_action + current_exploration_bonus

                experience_queue.put(
                    (agent_type_str, prev_state_np, prev_action, final_reward, next_state_np, done, None, worker_id)
                )

            # --- Select action if not done ---
            if not done and observation is not None:
                obs_dict_current = {
                    k: v if isinstance(v, (int, float, bool)) else (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in observation.items()
                }
                try:
                    current_state_np = policy_agent_for_action.process_observation(obs_dict_current)
                    action_to_take = policy_agent_for_action.select_action(current_state_np, current_epsilon)
                    pending_this_agent_step_data[agent_id] = (current_state_np, action_to_take)
                except ValueError as e:
                    print(f"Worker {worker_id} obs processing error during action selection: {e}")
                    if env.action_space(agent_id):
                        action_to_take = env.action_space(agent_id).sample()
            elif done:
                action_to_take = None
            else:
                if env.action_space(agent_id):
                    action_to_take = env.action_space(agent_id).sample()

            env.step(action_to_take)
            if stop_event.is_set():
                break

        if stop_event.is_set():
            break

        episode_info_payload = {
            'scores': dict(current_rewards_this_episode),
            'scout_id_identified': scout_agent_id_identified,
            'episode_num_worker_local': local_episode_count,
            'episode_num_global_approx': current_global_episode_num
        }
        experience_queue.put(('episode_info', None, None, None, None, None, episode_info_payload, worker_id))

        if should_record_video_worker and video_config_worker['video_folder'] and episode_frames:
            video_path = os.path.join(video_config_worker['video_folder'], f"episode_{current_global_episode_num + 1:06d}_worker{worker_id}.mp4")
            try:
                imageio.mimsave(video_path, episode_frames, fps=30)
                print(f"Worker {worker_id} saved video: {video_path}")
            except Exception as e:
                print(f"Worker {worker_id} failed to save video {video_path}: {e}")

    env.close()


# --- Main Parallel Training Function ---
def train_parallel(
        env_module_name, # Pass module name string
        num_episodes_total=2000,
        num_workers=4,
        novice_track=False,
        scout_model_load_path=None,
        save_scout_model_to="trained_dqn_agent_scout.pth",
        guard_model_load_path=None,
        save_guard_model_to="trained_dqn_agent_guard.pth",
        render_mode="rgb_array",
        video_folder="./rl_renders_parallel",
        video_record_interval=100,
        model_broadcast_learning_steps = 200 # How many LEARNING steps before broadcasting model
):
    print(f"Main process started. Device: {DEVICE}. Num workers: {num_workers}")
    if video_folder:
        os.makedirs(video_folder, exist_ok=True)
        print(f"Video folder created at {video_folder}")
    else:
        print("No video folder specified or render mode not set to 'rgb_array'. No videos will be saved.")

    scout_learner = TrainableRLAgent(model_load_path=scout_model_load_path, model_save_path=save_scout_model_to, is_learner=True)
    guard_learner = TrainableRLAgent(model_load_path=guard_model_load_path, model_save_path=save_guard_model_to, is_learner=True)

    experience_queue = mp.Queue(maxsize=num_workers * MAX_STEPS_PER_EPISODE * 2) # Buffer for experiences
    # List of (main_conn_to_worker, worker_conn_for_main) pipes
    model_update_pipes_main_ends = []
    model_update_pipes_worker_ends = []
    for _ in range(num_workers):
        worker_conn, main_conn = mp.Pipe(duplex=False)
        model_update_pipes_main_ends.append(main_conn)
        model_update_pipes_worker_ends.append(worker_conn)

    stop_event = mp.Event()
    worker_epsilon_queue = mp.Queue(maxsize=num_workers * 2) # For sending epsilon
    global_episode_completed_counter = mp.Value('i', 0) # Main process increments this

    # Initial model broadcast
    initial_scout_state_dict = scout_learner.get_policy_net_state_dict_cpu()
    initial_guard_state_dict = guard_learner.get_policy_net_state_dict_cpu()
    for i in range(num_workers):
        try:
            model_update_pipes_main_ends[i].send((initial_scout_state_dict, initial_guard_state_dict))
        except Exception as e:
             print(f"Error sending initial model to worker {i}: {e}")
    
    current_epsilon = EPSILON_START
    for _ in range(num_workers * 2): worker_epsilon_queue.put(current_epsilon) # Populate epsilon queue

    processes = []
    for i in range(num_workers):
        env_conf = {
            'novice_track': novice_track,
            'render_mode_worker': render_mode,
            'rewards_dict': CUSTOM_REWARDS_DICT,
        }
        model_p = { # Workers load these initially, then get updates via pipe
            'scout_model_path': scout_model_load_path, 
            'guard_model_path': guard_model_load_path  
        }
        video_conf_w = {
            'video_folder': video_folder,
            'record_interval': video_record_interval
        }
        # Pass worker's receiving end of the pipe
        p_args = (
            i, env_module_name, env_conf, model_p,
            experience_queue, model_update_pipes_worker_ends[i], 
            stop_event, worker_epsilon_queue,
            global_episode_completed_counter, video_conf_w
        )
        process = mp.Process(target=run_episode_worker, args=p_args, daemon=True) # Daemon ensures they exit with main
        processes.append(process)
        process.start()

    scout_scores_deque = deque(maxlen=100)
    guard_scores_deque = deque(maxlen=100)
    
    last_broadcast_scout_learn_step = 0
    last_broadcast_guard_learn_step = 0
    
    # Counters for learning steps taken by main learner
    scout_learning_steps_done = 0
    guard_learning_steps_done = 0

    experiences_processed_total = 0
    learn_calls_scout = 0
    learn_calls_guard = 0

    try:
        while global_episode_completed_counter.value < num_episodes_total:
            if stop_event.is_set(): break 

            processed_in_this_cycle = 0
            for _ in range(BATCH_SIZE * num_workers): # Process a limited number of items per cycle
                try:
                    experience_data = experience_queue.get(timeout=0.001) # Small timeout
                    experiences_processed_total +=1
                    processed_in_this_cycle +=1

                    agent_type, state, action, reward, next_state, done, episode_info, worker_id_report = experience_data

                    if agent_type == 'scout':
                        scout_learner.add_experience(state, action, reward, next_state, done)
                        scout_learner.global_step_counter += 1 # Tracks experiences added
                        scout_learner.t_step = (scout_learner.t_step + 1) % UPDATE_EVERY
                    elif agent_type == 'guard':
                        guard_learner.add_experience(state, action, reward, next_state, done)
                        guard_learner.global_step_counter += 1
                        guard_learner.t_step = (guard_learner.t_step + 1) % UPDATE_EVERY
                    elif agent_type == 'episode_info':
                        with global_episode_completed_counter.get_lock():
                            global_episode_completed_counter.value += 1
                        
                        ep_scores = episode_info['scores']
                        scout_id = episode_info['scout_id_identified']
                        if scout_id and scout_id in ep_scores:
                            scout_scores_deque.append(ep_scores[scout_id])
                        
                        current_guard_ep_scores_list = [s for id, s in ep_scores.items() if id != scout_id]
                        if current_guard_ep_scores_list:
                            guard_scores_deque.append(np.mean(current_guard_ep_scores_list))

                        avg_s_score = np.mean(scout_scores_deque) if scout_scores_deque else -1
                        avg_g_score = np.mean(guard_scores_deque) if guard_scores_deque else -1

                        if global_episode_completed_counter.value % 10000 == 0: # Log less frequently
                            print(f"\rEps: {global_episode_completed_counter.value}/{num_episodes_total} | "
                                  f"S_AvgS: {avg_s_score:.1f} | G_AvgS: {avg_g_score:.1f} | Eps: {current_epsilon:.4f} | "
                                  f"S_Buf: {len(scout_learner.memory)} | G_Buf: {len(guard_learner.memory)} | "
                                  f"S_LSteps: {scout_learning_steps_done} | G_LSteps: {guard_learning_steps_done}   ", end="")

                        if global_episode_completed_counter.value % 10000 == 0:
                            print(f"\rEps: {global_episode_completed_counter.value}/{num_episodes_total} | "
                                  f"S_AvgS: {avg_s_score:.1f} | G_AvgS: {avg_g_score:.1f} | Eps: {current_epsilon:.4f} | "
                                  f"S_Buf: {len(scout_learner.memory)} | G_Buf: {len(guard_learner.memory)} | "
                                  f"S_LSteps: {scout_learning_steps_done} | G_LSteps: {guard_learning_steps_done}   ")
                            if scout_learner.model_save_path: scout_learner.save_model()
                            if guard_learner.model_save_path: guard_learner.save_model()
                        
                        current_epsilon = max(EPSILON_END, EPSILON_DECAY * current_epsilon)
                        while not worker_epsilon_queue.full(): # Try to fill if space
                            try: worker_epsilon_queue.put_nowait(current_epsilon)
                            except mp.queues.Full: break
                        
                        if avg_s_score >= 200.0 and global_episode_completed_counter.value > 100: # Example solve condition
                            print(f"\nScout solved (avg score {avg_s_score:.2f}) in {global_episode_completed_counter.value} episodes!")
                            stop_event.set()
                            break
                    if stop_event.is_set(): break
                except mp.queues.Empty: break 
            if stop_event.is_set(): break

            # --- Scout Learning ---
            if scout_learner.t_step == 0 and len(scout_learner.memory) >= BATCH_SIZE: # Use >= for safety
                sample_result = scout_learner.memory.sample(BATCH_SIZE, beta=scout_learner.beta)
                if sample_result and sample_result[0] is not None:
                    experiences_s, indices_s, weights_s = sample_result
                    scout_learner.learn(experiences_s, indices_s, weights_s, GAMMA)
                    scout_learning_steps_done +=1
                    learn_calls_scout += 1
                    if learn_calls_scout % model_broadcast_learning_steps == 0:
                        s_dict = scout_learner.get_policy_net_state_dict_cpu()
                        for pipe_main_end in model_update_pipes_main_ends: pipe_main_end.send((s_dict, None))
                        # print(f"\nBroadcasted scout model after {learn_calls_scout} learning steps.")
            if scout_learner.global_step_counter > 0 and scout_learner.global_step_counter % TARGET_UPDATE_EVERY == 0 :
                scout_learner.update_target_net()

            # --- Guard Learning ---
            if guard_learner.t_step == 0 and len(guard_learner.memory) >= BATCH_SIZE:
                sample_result_g = guard_learner.memory.sample(BATCH_SIZE, beta=guard_learner.beta)
                if sample_result_g and sample_result_g[0] is not None:
                    experiences_g, indices_g, weights_g = sample_result_g
                    guard_learner.learn(experiences_g, indices_g, weights_g, GAMMA)
                    guard_learning_steps_done += 1
                    learn_calls_guard +=1
                    if learn_calls_guard % model_broadcast_learning_steps == 0:
                        g_dict = guard_learner.get_policy_net_state_dict_cpu()
                        for pipe_main_end in model_update_pipes_main_ends: pipe_main_end.send((None, g_dict))
                        # print(f"\nBroadcasted guard model after {learn_calls_guard} learning steps.")
            if guard_learner.global_step_counter > 0 and guard_learner.global_step_counter % TARGET_UPDATE_EVERY == 0:
                guard_learner.update_target_net()
            
            if processed_in_this_cycle == 0: # No new experiences, pause briefly
                time.sleep(0.01)


    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    except Exception as e: print(f"\nError in main training loop: {e}"); traceback.print_exc()
    finally:
        print("\nShutting down main loop and workers...")
        stop_event.set()
        
        # Clear queues that workers might be waiting on
        for q in [worker_epsilon_queue, experience_queue]:
            while not q.empty():
                try: q.get_nowait()
                except: break
            q.close() # Close before join_thread
            q.join_thread()

        # Close pipes from main end
        for main_conn in model_update_pipes_main_ends: main_conn.close()
        # Worker ends of pipes are managed by worker processes.

        for i, p in enumerate(processes):
            # print(f"Joining worker {i} (PID {p.pid})...")
            p.join(timeout=10) # Wait for worker to exit
            if p.is_alive():
                print(f"Worker {i} (PID {p.pid}) is still alive. Terminating.")
                p.terminate() # Force if needed
                p.join(timeout=5) # Wait for termination
        
        print("All workers joined.")
        if scout_learner.model_save_path: scout_learner.save_model()
        if guard_learner.model_save_path: guard_learner.save_model()
        print("Training finished or stopped.")

if __name__ == '__main__':
    # Important for multiprocessing with CUDA and some OS, set start method
    # Needs to be in the if __name__ == '__main__': block
    try:
        mp.set_start_method('spawn', force=True) 
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method 'spawn' (may have already been set or not supported): {e}")


    start_time = time.time()
    
    # Ensure the environment module name is correct here
    env_module_name_for_training = "til_environment.gridworld"
    
    try:
        # No direct import needed here if name is passed, but good for sanity check
        # from til_environment import gridworld
        # print("Successfully checked til_environment.gridworld import.")

        train_parallel(
            env_module_name=env_module_name_for_training,
            num_episodes_total=1000000, # Total episodes across all workers
            num_workers=max(1, mp.cpu_count() -7), # Example: Use all but one CPU core
            # num_workers=2, # Or set a fixed number for testing
            novice_track=False,
            scout_model_load_path="agent03_parallel_scout_1.15m_eps.pth",
            save_scout_model_to="agent03_parallel_scout_2.15m_eps.pth",
            guard_model_load_path="agent03_parallel_guard_1.15m_eps.pth",
            save_guard_model_to="agent03_parallel_guard_2.15m_eps.pth",
            render_mode="rgb_array",
            video_folder="./rl_renders_parallel_merged",
            video_record_interval=1000, # Record video every X global episodes
            model_broadcast_learning_steps=TARGET_UPDATE_EVERY * 2 # Broadcast model less frequently than target update
        )
    except ImportError:
        print(f"Could not import '{env_module_name_for_training}'. Ensure it's installed and accessible.")
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred during parallel training setup or execution: {e}")
        traceback.print_exc()

    end_time = time.time()
    print(f"Total training time: {(end_time - start_time):.2f} seconds.")