#!/usr/bin/env python3
from edupra_core.models import TDGammon
from edupra_core.path import ensure_exists
import argparse
import gym


def save_parameters(path: str, **kwargs) -> None:
    with open(f"{path}/parameters.txt", "w+") as file:
        print("Параметры:")

        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")
            print(f"{key}={value}")

        print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построитель моделей Edupra")

    parser.add_argument(
        "--save_path", help="Путь для сохранения модели", type=str, default=None
    )
    parser.add_argument(
        "--save_step", help="Сохранять модель каждые n эпизодов/игр", type=int, default=0
    )
    parser.add_argument(
        "--episodes", help="Количество эпизодов/игр", type=int, default=200000
    )
    parser.add_argument(
        "--init_weights", help="Инициализировать веса нулями", action="store_true"
    )
    parser.add_argument("--lr", help="Скорость обучения", type=float, default=1e-4)
    parser.add_argument("--hidden_units", help="Скрытые нейроны", type=int, default=40)
    parser.add_argument(
        "--lamda", help="Параметр распределения заслуг", type=float, default=0.7
    )
    parser.add_argument(
        "--model",
        help="Путь к модели для восстановления",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name", help="Название эксперимента", type=str, default="exp1"
    )
    parser.add_argument(
        "--seed", help="Начальное значение для воспроизведения результатов", type=int, default=123
    )

    args, *_ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_arguments()

    ai = TDGammon(
        hidden_units=args.hidden_units,
        lr=args.lr,
        lamda=args.lamda,
        init_weights=args.init_weights,
        seed=args.seed,
    )

    env = gym.make("gym_backgammon:backgammon-v0")

    if args.model:
        ensure_exists(args.model)
        ai.load(
            checkpoint_path=args.model,
            optimizer=None,
            eligibility_traces=True,
        )

    if args.save_path:
        ensure_exists(args.save_path)

        save_parameters(
            args.save_path,
            save_path=args.save_path,
            command_line_args=args,
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
            modules=[module for module in ai.modules()],
        )

    ai.train_agent(
        env=env,
        n_episodes=args.episodes,
        save_path=args.save_path,
        save_step=args.save_step,
        eligibility=True,
        name_experiment=args.name,
    )


if __name__ == "__main__":
    main()
