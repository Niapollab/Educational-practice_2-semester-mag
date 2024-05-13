#!/usr/bin/env python3
from .models import BaseResponse, Request, GameState, Response
from .state_repositories import RedisStateRepository, DictStateRepository
from edupra_core.agents import TDAgent, HumanAgent
from edupra_core.models import TDGammon, TDGammonCNN
from edupra_core.path import ensure_exists
from flask import Flask, request
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS
from typing import Any
import gym
import os


model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise ValueError("MODEL_PATH environment variable must be set.")
ensure_exists(model_path)


hidden_units = int(os.environ.get("MODEL_HIDDEN_UNITS", "40"))
model_type = os.environ.get("MODEL_TYPE", "nn")
is_nn_model = model_type == "nn"
ai = (
    TDGammon(hidden_units=hidden_units, lr=0.1, lamda=None, init_weights=False)
    if is_nn_model
    else TDGammonCNN(lr=0.0001)
)
ai.load(
    checkpoint_path=model_path,
    optimizer=None,
    eligibility_traces=False,
)


redis_url = os.environ.get("REDIS_URL")
state_repository = (
    RedisStateRepository(redis_url) if redis_url else DictStateRepository()
)


app = Flask(__name__)


def init_state() -> GameState:
    agents = {BLACK: TDAgent(BLACK, net=ai), WHITE: HumanAgent(WHITE)}
    env = gym.make(
        "gym_backgammon:backgammon-v0"
        if is_nn_model
        else "gym_backgammon:backgammon-pixel-v0"
    )
    return GameState(agents, env)


def handle_start(uid: str) -> Response:
    state = init_state()

    message = "\nNew game started\n"
    commands = []

    if state.game_started:
        message = "The game is already started. To start a new game, type 'new game'\n"
        commands.append("new game")

    else:
        state.game_finished = False
        state.game_started = True

        agent_color, state.first_roll, *_ = state.env.reset()
        state.agent = state.agents[agent_color]

        if agent_color == WHITE:
            message += f"{COLORS[state.agent.color]} Starts first | Roll={(abs(state.first_roll[0]), abs(state.first_roll[1]))} | Run 'move (src/target)'\n"  # type: ignore
            commands.extend(state.env.get_valid_actions(state.first_roll))
            state.roll = state.first_roll

        else:
            opponent = state.agents[agent_color]
            message += f"{COLORS[opponent.color]} Starts first | Roll={(abs(state.first_roll[0]), abs(state.first_roll[1]))}\n"  # type: ignore

            if state.first_roll:
                roll = state.first_roll
                state.first_roll = None
            else:
                roll = opponent.roll_dice()

            actions = state.env.get_valid_actions(roll)
            action = opponent.choose_best_action(actions, state.env)
            message += f"{COLORS[opponent.color]} | Roll={roll} | Action={action} | Run 'roll'\n"
            commands.extend(["roll", "new game"])
            _ = state.env.step(action)

            agent_color = state.env.get_opponent_agent()
            state.agent = state.agents[agent_color]

    state_repository[uid] = state
    return Response(message=message, state=state.env.game.state, actions=list(commands))


def handle_roll(uid: str) -> Response:
    state = state_repository[uid]

    message = ""
    commands = []

    if state.roll is not None:
        message += f"You have already rolled the dice {(abs(state.roll[0]), abs(state.roll[1]))}. Run 'move (src/target)'\n"
        actions = state.env.get_valid_actions(state.roll)
        if len(actions) == 0:
            commands.append("start")
        else:
            commands.extend(list(actions))

    elif state.game_finished:
        message += "The game is finished. Type 'Start' to start a new game\n"
        commands.append("start")

    elif not state.game_started:
        message += "The game is not started. Type 'start' to start a new game\n"
        commands.append("start")

    else:
        state.roll = state.agent.roll_dice()  # type: ignore
        message += f"{COLORS[state.agent.color]} | Roll={(abs(state.roll[0]), abs(state.roll[1]))} | Run 'move (src/target)'\n"  # type: ignore
        actions = state.env.get_valid_actions(state.roll)
        commands.extend(list(actions))

        if len(actions) == 0:
            message += "You cannot move\n"

            agent_color = state.env.get_opponent_agent()
            opponent = state.agents[agent_color]

            roll = opponent.roll_dice()

            actions = state.env.get_valid_actions(roll)
            action = opponent.choose_best_action(actions, state.env)
            message += f"{COLORS[opponent.color]} | Roll={roll} | Action={action}\n"
            *_, done, _ = state.env.step(action)

            if done:
                winner = state.env.game.get_winner()
                message += f"Game Finished!!! {COLORS[winner]} wins \n"
                commands.append("new game")
                state.game_finished = True
            else:
                agent_color = state.env.get_opponent_agent()
                state.agent = state.agents[agent_color]
                state.roll = None
                commands.extend(["roll", "new game"])

    state_repository[uid] = state
    return Response(message=message, state=state.env.game.state, actions=list(commands))


def handle_move(uid: str, command: str) -> Response:
    state = state_repository[uid]

    message = ""
    commands = []

    if state.roll is None:
        message += "You must roll the dice first\n"
        commands = state.last_commands

    elif state.game_finished:
        message += "The game is finished. Type 'new game' to start a new game\n"
        commands.append("new game")

    else:
        try:
            action = command.split()[1]
            action = action.split(",")
            play = []
            is_bar = False

            for move in action:
                move = move.replace("(", "")
                move = move.replace(")", "")
                s, t = move.split("/")

                if s == "BAR" or s == "bar":
                    play.append(("bar", int(t)))
                    is_bar = True
                else:
                    play.append((int(s), int(t)))

            if is_bar:
                action = tuple(play)
            else:
                action = tuple(sorted(play, reverse=True))

        except Exception:
            message += "Error during parsing move\n"
            commands = state.last_commands

        else:
            actions = state.env.get_valid_actions(state.roll)

            if action not in actions:
                message += (
                    f"Illegal move | Roll={(abs(state.roll[0]), abs(state.roll[1]))}\n"
                )
            else:
                message += f"{COLORS[state.agent.color]} | Roll={(abs(state.roll[0]), abs(state.roll[1]))} | Action={action}\n"  # type: ignore
                *_, done, _ = state.env.step(action)

                if done:
                    winner = state.env.game.get_winner()
                    message += f"Game Finished!!! {COLORS[winner]} wins\n"
                    commands.append("new game")
                    state.game_finished = True

                else:
                    agent_color = state.env.get_opponent_agent()
                    opponent = state.agents[agent_color]

                    roll = opponent.roll_dice()
                    actions = state.env.get_valid_actions(roll)
                    action = opponent.choose_best_action(actions, state.env)

                    message += (
                        f"{COLORS[opponent.color]} | Roll={roll} | Action={action}\n"
                    )
                    *_, done, _ = state.env.step(action)

                    if done:
                        winner = state.env.game.get_winner()
                        message += f"Game Finished!!! {COLORS[winner]} wins\n"
                        commands.append("new game")
                        state.game_finished = True

                    else:
                        commands.extend(["roll", "new game"])
                        agent_color = state.env.get_opponent_agent()
                        state.agent = state.agents[agent_color]
                        state.roll = None

    state_repository[uid] = state
    return Response(message=message, state=state.env.game.state, actions=list(commands))


@app.route("/", methods=["POST"])
def make_turn() -> Any:
    if "UID" not in request.headers:
        return BaseResponse(message="UID header must be provided.").dict(), 400

    uid = request.headers["UID"]
    command = Request(**request.json).command

    if command == "start" or command == "new game":
        return handle_start(uid).dict()

    if command == "roll":
        return handle_roll(uid).dict()

    if "move" in command:
        return handle_move(uid, command).dict()

    return BaseResponse(message=f'Unknown operation "{command}').dict(), 400


def main() -> None:
    app.run(host="0.0.0.0", port=7314)


if __name__ == "__main__":
    main()
