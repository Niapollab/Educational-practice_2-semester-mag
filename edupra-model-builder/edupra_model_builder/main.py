#!/usr/bin/env python3
from edupra_core.models import TDGammon, TDGammonCNN
from edupra_core.path import ensure_exists
import argparse
import gym
import os
import sys


def save_parameters(path: str, **kwargs) -> None:
    with open(f"{path}/parameters.txt", "w+") as file:
        print("Parameters:")

        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")
            print(f"{key}={value}")

        print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edupra model builder")

    parser.add_argument(
        "--save_path", help="Save directory location", type=str, default=None
    )
    parser.add_argument(
        "--save_step", help="Save the model every n episodes/games", type=int, default=0
    )
    parser.add_argument(
        "--episodes", help="Number of episodes/games", type=int, default=200000
    )
    parser.add_argument(
        "--init_weights", help="Init Weights with zeros", action="store_true"
    )
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--hidden_units", help="Hidden units", type=int, default=40)
    parser.add_argument(
        "--lamda", help="Credit assignment parameter", type=float, default=0.7
    )
    parser.add_argument(
        "--model",
        help="Directory location to the model to be restored",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name", help="Name of the experiment", type=str, default="exp1"
    )
    parser.add_argument(
        "--type", help="Model type", choices=["cnn", "nn"], type=str, default="nn"
    )
    parser.add_argument(
        "--seed", help="Seed used to reproduce results", type=int, default=123
    )

    args, *_ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_arguments()

    is_nn_model = args.type == "nn"
    need_eligibility = is_nn_model
    optimizer = None if is_nn_model else True

    ai = (
        TDGammon(
            hidden_units=args.hidden_units,
            lr=args.lr,
            lamda=args.lamda,
            init_weights=args.init_weights,
            seed=args.seed,
        )
        if is_nn_model
        else TDGammonCNN(lr=args.lr, seed=args.seed)
    )

    env = (
        gym.make("gym_backgammon:backgammon-v0")
        if is_nn_model
        else gym.make("gym_backgammon:backgammon-pixel-v0")
    )

    if args.model:
        ensure_exists(args.model)
        ai.load(
            checkpoint_path=args.model,
            optimizer=optimizer,
            eligibility_traces=need_eligibility,
        )

    if args.save_path:
        ensure_exists(args.save_path)

        save_parameters(
            args.save_path,
            save_path=args.save_path,
            command_line_args=args,
            type=args.type,
            hidden_units=args.hidden_units,
            init_weights=args.init_weights,
            alpha=ai.lr,
            lamda=ai.lamda,
            n_episodes=args.episodes,
            save_step=args.save_step,
            start_episode=ai.start_episode,
            name_experiment=args.name,
            env=env.spec.id,
            restored_model=args.model,
            seed=args.seed,
            eligibility=need_eligibility,
            optimizer=optimizer,
            modules=[module for module in ai.modules()],
        )

    ai.train_agent(
        env=env,
        n_episodes=args.episodes,
        save_path=args.save_path,
        save_step=args.save_step,
        eligibility=need_eligibility,
        name_experiment=args.name,
    )


if __name__ == "__main__":
    main()
