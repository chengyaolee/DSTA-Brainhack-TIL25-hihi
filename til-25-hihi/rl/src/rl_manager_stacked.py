import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque, namedtuple
import random
import os

# --- Configuration ---
# Neural Network Hyperparameters
INPUT_FEATURES = 1128  # 7*5*8 (viewcone) * 4 (stack size) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 384
HIDDEN_LAYER_2_SIZE = 384
HIDDEN_LAYER_3_SIZE = 256
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay
STACK_SIZE = 4

# Game Environment Constants
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Agent settings
EPSILON_INFERENCE = 0.01 # Small epsilon for some exploration even during inference, or 0 for pure exploitation

# --- Deep Q-Network (DQN) Model (same as in rl_agent_python_v1) ---
# Ensure this class is defined and available
class DQN_3_hl(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DQN_3_hl, self).__init__()
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

# --- RL Agent (Inference Only) ---
class RLManager:
    """
    The Reinforcement Learning Agent for inference using a pre-trained model.
    It processes observations and uses a DQN to select actions.
    """
    def __init__(
        self, 
        scout_model_path="agent03_parallel_scout_1.25m_eps.pth", 
        guard_model_path="agent03_parallel_guard_1.25m_eps.pth",
        stack_size: int = 4,
    ):
        """
        Initialises the RL Agent.
        Args:
            model_path (str): Path to a pre-trained model file (.pth).
        """
        # Ensure DEVICE is defined (e.g., globally or passed in)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            self.scout_model = DQN_3_hl(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS).to(self.device)
            self.guard_model = DQN_3_hl(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS).to(self.device)
        except NameError as e:
            print(f"Error: Required constant not defined: {e}")
            print("Please ensure INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS are defined.")
            raise

        if scout_model_path and guard_model_path and os.path.exists(scout_model_path) and os.path.exists(guard_model_path):
            try:
                # Load the state_dict onto the correct device
                # Scout
                self.scout_model.load_state_dict(torch.load(scout_model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {scout_model_path}")
                
                # Guard
                self.guard_model.load_state_dict(torch.load(guard_model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {guard_model_path}")
            except Exception as e:
                print(f"Error loading model from {scout_model_path} and {guard_model_path}: {e}. Initialising with random weights.")
                # Fallback to random weights if loading fails or model mismatch
                self.scout_model.apply(self._initialise_weights)
                self.guard_model.apply(self._initialise_weights)
        else:
            print(f"No model path provided or path {scout_model_path} / {guard_model_path} does not exist. Initialising model with random weights.")
            # Initialise with random weights if no path is given or file not found
            self.scout_model.apply(self._initialise_weights)
            self.guard_model.apply(self._initialise_weights)

        # Set the model to evaluation mode (disables dropout, batch norm stats etc.)
        self.scout_model.eval()
        self.guard_model.eval()
        
        #stacked viewcone
        self.stack_size = stack_size
        self.viewcone_stack = deque(maxlen=self.stack_size)

    def _initialise_weights(self, m):
        """
        Initialises weights of the neural network layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value): # Same as in rl_agent_python_v1
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) 
        tile_features.append(float((tile_value & 0b10) >> 1))
        for i in range(2, 8):
            tile_features.append(float((tile_value >> i) & 1))
        return tile_features

    def process_observation(self, observation_dict):
        processed_viewcone_features = []
        target_num_frames = self.stack_size  # The number of frames the NN expects

        # Features for a completely empty/zero frame (all tiles are 0)
        # Each tile unpacks to 8 features. A frame is 7x5 tiles.
        num_features_per_tile = 8
        tiles_per_frame = 7 * 5
        empty_frame_tile_features = self._unpack_viewcone_tile(0)
        empty_frame_features = []
        for _ in range(tiles_per_frame):
            empty_frame_features.extend(empty_frame_tile_features)

        input_stacked_viewcone = observation_dict.get("stacked_viewcone", None)

        if input_stacked_viewcone is not None and input_stacked_viewcone.size > 0 :
            # input_stacked_viewcone comes from np.stack(self.viewcone_stack)
            # Shape: (num_available_frames, 7, 5)
            num_available_frames = input_stacked_viewcone.shape[0]
            
            frames_data_to_process = [None] * target_num_frames # List to hold frame data or None for padding

            # Place available frames at the end (most recent part of the stack)
            # The frames in input_stacked_viewcone are [oldest_in_deque, ..., newest_in_deque]
            start_idx_in_target = target_num_frames - num_available_frames
            for i in range(num_available_frames):
                frames_data_to_process[start_idx_in_target + i] = input_stacked_viewcone[i]

            # Process all target_num_frames, using actual data or padding
            for frame_data in frames_data_to_process:
                if frame_data is not None: # Actual frame from deque
                    for r in range(7):
                        for c in range(5):
                            # Ensure safe access to tile_value if frame_data could be smaller than 7x5
                            tile_value = frame_data[r][c] if r < frame_data.shape[0] and c < frame_data.shape[1] else 0
                            processed_viewcone_features.extend(self._unpack_viewcone_tile(tile_value))
                else: # This is a padding frame (older than what's in deque)
                    processed_viewcone_features.extend(empty_frame_features)
        else:
            # No "stacked_viewcone" provided, or it was empty.
            # This might be the first step, or if stacking is conceptually turned off.
            # We should still produce target_num_frames worth of features.
            # If "viewcone" (single frame) is present, use it as the most recent frame.
            single_viewcone = observation_dict.get("viewcone", []) # Default to empty list

            for i in range(target_num_frames):
                if i == target_num_frames - 1: # Slot for the most recent frame
                    if single_viewcone and len(single_viewcone) == 7 and len(single_viewcone[0]) == 5 : # Check if valid
                        for r in range(7):
                            for c in range(5):
                                tile_value = single_viewcone[r][c] if r < len(single_viewcone) and c < len(single_viewcone[r]) else 0
                                processed_viewcone_features.extend(self._unpack_viewcone_tile(tile_value))
                    else: # single_viewcone is invalid or not present, pad this frame too
                        processed_viewcone_features.extend(empty_frame_features)
                else: # Pad older frames
                    processed_viewcone_features.extend(empty_frame_features)
        
        # --- Combine with other features ---
        final_processed_features = list(processed_viewcone_features)

        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        final_processed_features.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        final_processed_features.extend([norm_x, norm_y])

        scout_role = float(observation_dict.get("scout", 0))
        final_processed_features.append(scout_role)
        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        final_processed_features.append(norm_step)

        # --- Sanity Check ---
        # The expected length should ALWAYS correspond to what the NN model expects (INPUT_FEATURES).
        # This is self.stack_size * (features_per_frame) + other_features.
        expected_total_feature_len = self.stack_size * (7 * 5 * num_features_per_tile) + 4 + 2 + 1 + 1
        
        if len(final_processed_features) != expected_total_feature_len:
            num_actual_frames_info = "N/A"
            if input_stacked_viewcone is not None and input_stacked_viewcone.size > 0:
                num_actual_frames_info = str(input_stacked_viewcone.shape[0])

            raise ValueError(
                f"Feature length mismatch. Expected {expected_total_feature_len}, got {len(final_processed_features)}. "
                f"self.stack_size={self.stack_size}, num_available_frames_in_input_stack={num_actual_frames_info}."
            )
        return np.array(final_processed_features, dtype=np.float32)

    def rl(self, observation_dict):
        """Selects an action based on the current observation using the loaded DQN model.
        Uses epsilon-greedy for exploration during inference/testing if EPSILON_INFERENCE > 0.
        For pure greedy inference, set EPSILON_INFERENCE = 0.

        Args:
            observation_dict (dict): The observation dictionary provided by the environment.
        Returns:
            int: The selected action (0 to OUTPUT_ACTIONS-1).
                 See environment documentation for action mapping.
        """
        # Assuming EPSILON_INFERENCE is defined
        if random.random() < EPSILON_INFERENCE:
            # Explore: select a random action
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            # attach stacked viewcone to observation dict
            self.viewcone_stack.append(observation_dict["viewcone"])
            current_stack_list = list(self.viewcone_stack)
            if not current_stack_list: # Handle case where deque is empty
                 # Create an empty array with correct subsequent dimensions for shape[0] to be 0
                 # but allow process_observation to know the intended frame structure if needed (though it doesn't use it for empty)
                stacked_viewcone_for_obs = np.empty((0, 7, 5), dtype=type(observation_dict["viewcone"][0][0] if observation_dict["viewcone"] and observation_dict["viewcone"][0] else 0))
            else:
                stacked_viewcone_for_obs = np.array(current_stack_list)

            observation_dict["stacked_viewcone"] = stacked_viewcone_for_obs
            # Exploit: select the best action based on Q-values from the model
            state_np = self.process_observation(observation_dict)
            # Convert numpy array to torch tensor and move to the appropriate device
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)

            with torch.no_grad(): # Important: disable gradient calculation for inference
                if observation_dict.get("scout",0) == 1:
                    # Get Q values from the scout model
                    q_values = self.scout_model(state_tensor)
                    # Select the action with the maximum Q-value
                    action = torch.argmax(q_values, dim=1).item()
                else:
                    # Get Q values from the guard model
                    q_values = self.guard_model(state_tensor)
                    # Select the action with the maximum Q-value
                    action = torch.argmax(q_values, dim=1).item()
            return action

    def reset_state(self):
        """
        Resets any internal state if the agent was stateful (e.g., for RNNs).
        Not strictly necessary for this feedforward DQN, but included for compatibility.
        """
        self.viewcone_stack.clear()
        pass # No state to reset for this simple DQN


