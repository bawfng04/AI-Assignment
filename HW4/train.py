# ==============================================================================
# HW4 — D3QN Training Loop
# ==============================================================================
"""
Main training script for Dueling Double DQN on Atari environments.

Usage:
    python train.py                           # Use default config
    python train.py --config configs/pong.yaml
    python train.py --env BreakoutNoFrameskip-v4 --total-timesteps 2000000
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from src.agent import D3QNAgent
from src.utils import make_atari_env, set_global_seeds, preprocess_observation
from src.logger import RLLogger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with YAML config overrides."""
    parser = argparse.ArgumentParser(
        description="Train D3QN Agent on Atari Games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment name")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/auto)")
    return parser.parse_args()


def get_device(device_str: str = None) -> torch.device:
    """Resolve the compute device."""
    if device_str and device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config: Dict[str, Any], device: torch.device) -> None:
    """
    Main training loop for D3QN.
    
    Flow:
        1. Create environment with DeepMind wrappers
        2. Initialize D3QN agent
        3. Collect transitions with epsilon-greedy
        4. Train after warmup period
        5. Periodically log, evaluate, and checkpoint
    """
    seed = config["seed"]
    set_global_seeds(seed)

    # --- Paths ---
    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Environment ---
    env_name = config["env"]["name"]
    env = make_atari_env(
        env_name, seed=seed,
        frame_stack=config["env"]["frame_stack"],
        clip_rewards=config["env"]["clip_rewards"],
        episodic_life=config["env"]["episodic_life"],
    )
    
    # Derive state shape (CHW) and action count
    obs_shape = env.observation_space.shape  # (H, W, C*stack)
    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    n_actions = env.action_space.n

    print("=" * 60)
    print(f"  D3QN Training — {env_name}")
    print(f"  State shape: {state_shape} | Actions: {n_actions}")
    print(f"  Device: {device} | Seed: {seed}")
    print("=" * 60)

    # --- Agent ---
    agent = D3QNAgent(state_shape, n_actions, config, device)

    # --- Logger ---
    logger = RLLogger(log_dir=config["paths"]["log_dir"])

    # --- Training ---
    training_cfg = config["training"]
    total_timesteps = training_cfg["total_timesteps"]
    learning_starts = training_cfg["learning_starts"]
    train_freq = training_cfg["train_freq"]
    save_interval = training_cfg["save_interval"]
    log_interval = training_cfg["log_interval"]

    obs, _ = env.reset()
    state = preprocess_observation(obs)

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    recent_rewards: list[float] = []

    pbar = tqdm(range(1, total_timesteps + 1), desc="Training", unit="step")

    for step in pbar:
        # --- Action Selection ---
        action = agent.select_action(state)

        # --- Environment Step ---
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = preprocess_observation(next_obs)

        # --- Store Transition ---
        agent.store_transition(state, action, float(reward), next_state, done)

        state = next_state
        episode_reward += float(reward)
        episode_length += 1

        # --- Learn ---
        metrics = {"loss": 0.0, "mean_q": 0.0, "grad_norm": 0.0}
        if step >= learning_starts and step % train_freq == 0:
            metrics = agent.learn()
            logger.log_training_step(
                step, metrics["loss"], metrics["mean_q"], metrics["grad_norm"]
            )
            logger.log_buffer_stats(
                step, len(agent.replay_buffer), agent.replay_buffer.beta
            )

        # --- Episode End ---
        if done:
            episode_count += 1
            recent_rewards.append(episode_reward)

            logger.log_episode(
                episode_count, episode_reward, episode_length, agent.current_epsilon
            )

            # Update progress bar
            avg_reward = np.mean(recent_rewards[-100:])
            pbar.set_postfix({
                "ep": episode_count,
                "reward": f"{episode_reward:.1f}",
                "avg100": f"{avg_reward:.1f}",
                "eps": f"{agent.current_epsilon:.3f}",
                "loss": f"{metrics['loss']:.4f}",
            })

            # Print periodic summary
            if episode_count % log_interval == 0:
                print(
                    f"\n[Episode {episode_count}] "
                    f"Avg Reward (100): {avg_reward:.2f} | "
                    f"Epsilon: {agent.current_epsilon:.4f} | "
                    f"Buffer: {len(agent.replay_buffer)} | "
                    f"Steps: {step}"
                )

            # Reset
            obs, _ = env.reset()
            state = preprocess_observation(obs)
            episode_reward = 0.0
            episode_length = 0

        # --- Checkpoint ---
        if step % save_interval == 0:
            ckpt_path = str(ckpt_dir / f"d3qn_{env_name}_{step}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"\n  [Checkpoint] Saved to {ckpt_path}")

    # --- Final Save ---
    final_path = str(ckpt_dir / f"d3qn_{env_name}_final.pt")
    agent.save_checkpoint(final_path)
    print(f"\n[Training Complete] Final checkpoint: {final_path}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Final avg reward (100): {np.mean(recent_rewards[-100:]):.2f}")

    logger.close()
    env.close()


def main() -> None:
    """Entry point: load config, apply CLI overrides, train."""
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.env:
        config["env"]["name"] = args.env
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps

    device = get_device(args.device)
    train(config, device)


if __name__ == "__main__":
    main()
