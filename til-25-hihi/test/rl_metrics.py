import json
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from til_environment import gridworld

# Load team config
load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

# Constants
NUM_ROUNDS = 8
NUM_EPISODES = 100  # Change as needed


def run_episode(novice: bool, episode_id: int, save_video: bool = False):
    env = gridworld.env(env_wrappers=[], render_mode="rgb_array", novice=novice)
    _agent = env.possible_agents[0]
    rewards = {agent: 0 for agent in env.possible_agents}
    frames = []

    for rd in range(NUM_ROUNDS):
        env.reset()
        _ = requests.post("http://localhost:5004/reset")

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            observation = {
                k: v if isinstance(v, int) else v.tolist() for k, v in observation.items()
            }

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == _agent:
                response = requests.post(
                    "http://localhost:5004/rl",
                    data=json.dumps({"instances": [{"observation": observation}]}),
                )
                predictions = response.json()["predictions"]
                action = int(predictions[0]["action"])
            else:
                action = env.action_space(agent).sample()
            env.step(action)

            if save_video:
                frame = env.render()
                frames.append(frame)

    env.close()

    if save_video:
        import imageio
        imageio.mimsave(f"rl_test_videos/episode_{episode_id}.mp4", frames, fps=30)

    # Same normalization as before
    return rewards[_agent] / NUM_ROUNDS / 100


def main(novice: bool):
    scores = []
    for ep in range(NUM_EPISODES):
        score = run_episode(novice=novice, episode_id=ep, save_video=(ep % 10 == 0))  # Save only first 3
        scores.append(score)
        print(f"Episode {ep + 1}: Score = {score:.3f}")

    scores = np.array(scores)

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Mean score:         {np.mean(scores):.4f}")
    print(f"Std deviation:      {np.std(scores):.4f}")
    print(f"Min score:          {np.min(scores):.4f}")
    print(f"Max score:          {np.max(scores):.4f}")
    print(f"15th percentile:    {np.percentile(scores, 15):.4f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title("Score Distribution Over Episodes")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.axvline(np.percentile(scores, 15), color='red', linestyle='dashed', linewidth=1.5, label="15th Percentile")
    plt.legend()
    plt.savefig("score_distribution.png")
    plt.show()

if __name__ == "__main__":
    main(novice=False)
