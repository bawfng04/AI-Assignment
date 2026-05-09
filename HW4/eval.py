# ==============================================================================
# HW4 — D3QN Evaluation & Recording
# ==============================================================================
"""
Load a trained D3QN checkpoint and evaluate/record gameplay.

Usage:
    python eval.py --checkpoint checkpoints/d3qn_PongNoFrameskip-v4_final.pt
    python eval.py --checkpoint checkpoints/d3qn_final.pt --record gameplay.gif
    python eval.py --checkpoint checkpoints/d3qn_final.pt --episodes 20
"""

import argparse
import os
import yaml
import numpy as np
import torch
import imageio
from pathlib import Path
from typing import Dict, Any, List

from src.agent import D3QNAgent
from src.utils import make_atari_env, set_global_seeds, preprocess_observation


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained D3QN Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment name")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--record", type=str, default=None,
                        help="Output path for recording (.gif or .mp4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--render", action="store_true",
                        help="Render gameplay in a window")
    return parser.parse_args()


def evaluate(
    agent: D3QNAgent,
    env_name: str,
    n_episodes: int,
    seed: int,
    render: bool = False,
    record_path: str = None,
) -> Dict[str, Any]:
    """
    Run evaluation episodes and optionally record gameplay.
    
    Args:
        agent:       Trained D3QN agent.
        env_name:    Environment ID.
        n_episodes:  Number of episodes to evaluate.
        seed:        Random seed.
        render:      Whether to render in a window.
        record_path: If set, save recording to this path (.gif/.mp4).
    
    Returns:
        Dict with mean_reward, std_reward, all_rewards, all_lengths.
    """
    render_mode = "human" if render else "rgb_array" if record_path else None
    
    env = make_atari_env(
        env_name, seed=seed, clip_rewards=False,
        episodic_life=False, render_mode=render_mode,
    )

    all_rewards: List[float] = []
    all_lengths: List[int] = []
    frames: List[np.ndarray] = []  # For recording

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Greedy action (no exploration)
            state_t = torch.tensor(
                state, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            action = agent.online_net.get_action(state_t)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Capture frame for recording
            if record_path and render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            state = preprocess_observation(next_obs)
            episode_reward += float(reward)
            episode_length += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        print(f"  Episode {ep}/{n_episodes}: "
              f"Reward = {episode_reward:.1f}, Length = {episode_length}")

    env.close()

    # Save recording
    if record_path and frames:
        save_recording(frames, record_path)

    results = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "all_rewards": all_rewards,
        "all_lengths": all_lengths,
    }

    return results


def save_recording(frames: List[np.ndarray], output_path: str) -> None:
    """
    Save recorded frames as GIF or MP4.
    
    Args:
        frames:      List of RGB frames (numpy arrays).
        output_path: Output file path (.gif or .mp4).
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith(".gif"):
        imageio.mimsave(output_path, frames, fps=30, loop=0)
        print(f"  [Recording] GIF saved to {output_path} ({len(frames)} frames)")
    elif output_path.endswith(".mp4"):
        writer = imageio.get_writer(output_path, fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  [Recording] MP4 saved to {output_path} ({len(frames)} frames)")
    else:
        print(f"  [Warning] Unsupported format. Use .gif or .mp4")


def main() -> None:
    """Entry point for evaluation."""
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_name = args.env or config["env"]["name"]
    seed = args.seed
    set_global_seeds(seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create agent and load checkpoint
    env_tmp = make_atari_env(env_name, seed=seed, clip_rewards=False,
                             episodic_life=False)
    obs_shape = env_tmp.observation_space.shape
    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    agent = D3QNAgent(state_shape, n_actions, config, device)
    agent.load_checkpoint(args.checkpoint)
    agent.online_net.eval()

    print("=" * 60)
    print(f"  D3QN Evaluation — {env_name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device} | Episodes: {args.episodes}")
    print("=" * 60)

    # Run evaluation
    results = evaluate(
        agent, env_name, args.episodes, seed,
        render=args.render, record_path=args.record,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Mean Reward:  {results['mean_reward']:.2f} "
          f"± {results['std_reward']:.2f}")
    print(f"  Min Reward:   {results['min_reward']:.1f}")
    print(f"  Max Reward:   {results['max_reward']:.1f}")
    print(f"  Episodes:     {args.episodes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
