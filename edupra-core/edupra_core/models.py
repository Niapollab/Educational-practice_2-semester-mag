import datetime
import random
import time
from itertools import count

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from gym_backgammon.envs.backgammon import BLACK, WHITE

from .agents import RandomAgent, TDAgent, evaluate_agents


class BaseModel(keras.Model):
    def __init__(self, lr, lamda, seed=123):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.lamda = lamda
        self.start_episode = 0

        self.eligibility_traces = None
        self.optimizer = None

        tf.random.set_seed(seed)
        random.seed(seed)

    def update_weights(self, p, p_next):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def init_eligibility_traces(self):
        self.eligibility_traces = [
            tf.zeros(weights.shape)
            for weights in self.trainable_weights
        ]

    def checkpoint(self, checkpoint_path, step, name_experiment):
        path = checkpoint_path + "/{}_{}_{}.tar".format(
            name_experiment,
            datetime.datetime.now().strftime("%Y%m%d_%H%M_%S_%f"),
            step + 1,
        )
        self.save_weights(path)
        with open(path + '_metadata.pkl', 'wb') as f:
            import pickle
            pickle.dump({
                "step": step + 1,
                "eligibility": self.eligibility_traces if self.eligibility_traces else [],
            }, f)
        print("\nCheckpoint saved: {}".format(path))

    def load(self, checkpoint_path, optimizer=None, eligibility_traces=None):
        import pickle
        self.load_weights(checkpoint_path)
        with open(checkpoint_path + '_metadata.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
        self.start_episode = checkpoint["step"]

        if eligibility_traces is not None:
            self.eligibility_traces = checkpoint["eligibility"]

    def train_agent(
        self,
        env,
        n_episodes,
        save_path=None,
        eligibility=False,
        save_step=0,
        name_experiment="",
    ):
        start_episode = self.start_episode
        n_episodes += start_episode

        wins = {WHITE: 0, BLACK: 0}
        network = self

        agents = {
            WHITE: TDAgent(WHITE, net=network),
            BLACK: TDAgent(BLACK, net=network),
        }

        durations = []
        steps = 0
        start_training = time.time()

        for episode in range(start_episode, n_episodes):
            if eligibility:
                self.init_eligibility_traces()

            agent_color, first_roll, observation = env.reset()
            agent = agents[agent_color]

            t = time.time()

            for i in count():
                if first_roll:
                    roll = first_roll
                    first_roll = None
                else:
                    roll = agent.roll_dice()

                p = self(observation)

                actions = env.get_valid_actions(roll)
                action = agent.choose_best_action(actions, env)
                observation_next, reward, done, winner = env.step(action)
                p_next = self(observation_next)

                if done:
                    if winner is not None:
                        loss = self.update_weights(p, reward)

                        wins[agent.color] += 1

                    tot = sum(wins.values())
                    tot = tot if tot > 0 else 1

                    print(
                        "Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
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

                    durations.append(time.time() - t)
                    steps += i
                    break
                else:
                    _ = self.update_weights(p, p_next)

                agent_color = env.get_opponent_agent()
                agent = agents[agent_color]

                observation = observation_next

            if (
                save_path
                and save_step > 0
                and episode > 0
                and (episode + 1) % save_step == 0
            ):
                self.checkpoint(
                    checkpoint_path=save_path,
                    step=episode,
                    name_experiment=name_experiment,
                )
                agents_to_evaluate = {
                    WHITE: TDAgent(WHITE, net=network),
                    BLACK: RandomAgent(BLACK),
                }
                evaluate_agents(agents_to_evaluate, env, n_episodes=20)
                print()

        print(
            "\nAverage duration per game: {} seconds".format(
                round(sum(durations) / n_episodes, 3)
            )
        )
        print(
            "Average game length: {} plays | Total Duration: {}".format(
                round(steps / n_episodes, 2),
                datetime.timedelta(seconds=int(time.time() - start_training)),
            )
        )

        if save_path:
            self.checkpoint(
                checkpoint_path=save_path,
                step=n_episodes - 1,
                name_experiment=name_experiment,
            )

            with open("{}/comments.txt".format(save_path), "a") as file:
                file.write(
                    "Average duration per game: {} seconds".format(
                        round(sum(durations) / n_episodes, 3)
                    )
                )
                file.write(
                    "\nAverage game length: {} plays | Total Duration: {}".format(
                        round(steps / n_episodes, 2),
                        datetime.timedelta(seconds=int(time.time() - start_training)),
                    )
                )

        env.close()

    def compare_with(self, env, n_episodes, other=None):
        agents_to_evaluate = {
            WHITE: TDAgent(WHITE, net=self),
            BLACK: TDAgent(BLACK, net=other) if other else RandomAgent(BLACK),
        }
        return evaluate_agents(agents_to_evaluate, env, n_episodes)


class TDGammon(BaseModel):
    def __init__(
        self,
        hidden_units,
        lr,
        lamda,
        init_weights,
        seed=123,
        input_units=198,
        output_units=1,
    ):
        super(TDGammon, self).__init__(lr, lamda, seed=seed)

        tf.random.set_seed(seed)

        self.hidden = layers.Dense(hidden_units, activation='sigmoid')
        self.output_layer = layers.Dense(output_units, activation='sigmoid')

        # Build model
        self.build((None, input_units))

        if init_weights:
            self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
                if layer.use_bias:
                    layer.bias.assign(tf.zeros_like(layer.bias))

    def call(self, inputs):
        return self.forward(inputs)

    def forward(self, x):
        x = tf.convert_to_tensor(np.array(x), dtype=tf.float32)
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

    def update_weights(self, p, p_next):
        with tf.GradientTape() as tape:
            tape.watch(p)
            loss = p

        gradients = tape.gradient(loss, self.trainable_weights)

        td_error = p_next - p

        for i, (weights, grad) in enumerate(zip(self.trainable_weights, gradients)):
            if grad is not None:
                self.eligibility_traces[i] = (
                    self.lamda * self.eligibility_traces[i] + grad
                )

                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.assign(new_weights)

        return td_error
