import random
import time
from itertools import count
from random import randint, choice

import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

random.seed(0)


class Agent:
    def __init__(self, color):
        self.color = color
        self.name = "Agent({})".format(COLORS[color])

    def roll_dice(self):
        return (
            (-randint(1, 6), -randint(1, 6))
            if self.color == WHITE
            else (randint(1, 6), randint(1, 6))
        )

    def choose_best_action(self, actions, env):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = "RandomAgent({})".format(COLORS[color])

    def choose_best_action(self, actions, env):
        return choice(list(actions)) if actions else None


class HumanAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = "HumanAgent({})".format(COLORS[color])

    def choose_best_action(self, actions=None, env=None):
        pass


class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = "TDAgent({})".format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)

                env.game.restore_state(state)

            best_action_index = (
                int(np.argmax(values))
                if self.color == WHITE
                else int(np.argmin(values))
            )
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action


class TDAgentGNU(TDAgent):
    def __init__(self, color, net, gnubg_interface):
        super().__init__(color, net)
        self.gnubg_interface = gnubg_interface

    def roll_dice(self):
        gnubg = self.gnubg_interface.send_command("roll")
        return self.handle_opponent_move(gnubg)

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            game = env.game
            values = [0.0] * len(actions)
            state = game.save_state()

            for i, action in enumerate(actions):
                game.execute_play(self.color, action)
                opponent = game.get_opponent(self.color)
                observation = (
                    game.get_board_features(opponent)
                    if env.model_type == "nn"
                    else env.render(mode="state_pixels")
                )
                values[i] = self.net(observation)
                game.restore_state(state)

            best_action_index = (
                int(np.argmax(values))
                if self.color == WHITE
                else int(np.argmin(values))
            )
            best_action = list(actions)[best_action_index]

        return best_action

    def handle_opponent_move(self, gnubg):
        previous_agent = gnubg.agent
        if previous_agent == self.color:
            return gnubg
        else:
            while previous_agent != self.color and gnubg.winner is None:
                if gnubg.double:
                    gnubg = self.gnubg_interface.send_command("take")
                else:
                    gnubg = self.gnubg_interface.send_command("roll")
                previous_agent = gnubg.agent
            return gnubg


def evaluate_agents(agents, env, n_episodes):
    wins = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):
        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        t = time.time()

        for i in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                if winner is not None:
                    wins[agent.color] += 1
                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print(
                    "EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
                        episode + 1,
                        winner,
                        i,
                        agents[WHITE].name,
                        wins[WHITE],
                        (wins[WHITE] / tot) * 100,
                        agents[BLACK].name,
                        wins[BLACK],
                        (wins[BLACK] / tot) * 100,
                        time.time() - t,
                    )
                )
                break

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next
    return wins
