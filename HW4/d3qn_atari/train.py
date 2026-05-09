import yaml
import argparse
from typing import Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(description="Train D3QN on Atari")
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    return parser.parse_args()


# (Training loop implementation would go here)
if __name__ == "__main__":
    args = parse_args()
    print(f"Starting training on {args.env} with seed {args.seed}...")
