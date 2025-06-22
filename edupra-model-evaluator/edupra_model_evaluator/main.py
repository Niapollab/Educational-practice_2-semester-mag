#!/usr/bin/env python3
import argparse
from pathlib import Path

import gym
from edupra_core.models import TDGammon
from edupra_core.path import ensure_exists


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценщик моделей Edupra")

    parser.add_argument(
        "-f", "--first", help="Путь к первой модели", type=Path, default=None
    )
    parser.add_argument(
        "-s", "--second", help="Путь ко второй модели", type=Path, default=None
    )
    parser.add_argument(
        "-e", "--episodes", help="Количество эпизодов/игр", type=int, default=200000
    )
    parser.add_argument(
        "-h1", help="Количество скрытых нейронов для первой модели", type=int, default=40
    )
    parser.add_argument(
        "-h2", help="Количество скрытых нейронов для второй модели", type=int, default=40
    )

    args, *_ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_arguments()

    first = TDGammon(hidden_units=args.h1, lr=0.1, lamda=None, init_weights=False)
    second = (
        TDGammon(hidden_units=args.h2, lr=0.1, lamda=None, init_weights=False)
        if args.second
        else None
    )

    env = gym.make("gym_backgammon:backgammon-v0")

    ensure_exists(args.first)
    first.load(
        checkpoint_path=args.first,
        optimizer=None,
        eligibility_traces=True,
    )

    if second:
        ensure_exists(args.second)
        second.load(
            checkpoint_path=args.second,
            optimizer=None,
            eligibility_traces=True,
        )

    first.compare_with(env=env, n_episodes=args.episodes, other=second)
    env.close()


if __name__ == "__main__":
    main()
