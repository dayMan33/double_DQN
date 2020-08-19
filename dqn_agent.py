import numpy as np
import pandas as pd
from double_dqn.double_dqn import DoubleDQN
from double_dqn.experience_replay import ExperienceReplay
from tensorflow.keras.models import load_model, model_from_json
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


class DQNAgent:
    def __init__(self, env, net_update_rate: int = 25, exploration_rate: float = 1.0,
                 exploration_decay: float = 0.00005):
        # set hyper parameters
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.net_updating_rate = net_update_rate

        # set environment
        self.env = env
        self.state_shape = env.get_state_shape()
        self.action_shape = env.get_action_shape()

        # the number of experience per batch for batch learning
        # Experience Replay for batch learning
        self.exp_rep = ExperienceReplay()

        # Deep Q Network
        self.net = None

    def set_model(self, model):
        """ Sets the model the agent is used to train. Receives a compiled tf Model with
            input_shape = env.observation_space and output_shape = env.action_s pace"""
        self.net = DoubleDQN(model)

    def get_action(self, state: np.ndarray, eps=0) -> int:
        """Given a state returns a random action with probability eps, and argmax(q_net(state)) with probability 1-eps.
           (only legal actions are considered)"""
        if self.net is None:
            raise NotImplementedError(
                'agent.get_action called before model was not initiated.\n Please set the agent\'s model'
                ' using the set_model method. You can access the state and action shapes using '
                'agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        legal_actions = self.env.get_legal_actions(state)

        if np.random.random() >= eps:  # Exploitation

            # Calculate the Q-value of each action
            q_values = self.net.predict(state[np.newaxis, ...], np.expand_dims(legal_actions, 0))

            # Make sure we only choose between available actions
            legal_actions = np.logical_and(legal_actions, q_values == np.max(q_values))

        return np.random.choice(np.flatnonzero(legal_actions))

    def update_net(self, batch_size: int):
        """ if there are more than batch_size experiences, Optimizes the network's weights using the Double-Q-learning
         algorithm with a batch of experiences, else returns"""
        if self.exp_rep.get_num() < batch_size:
            return
        batch = self.exp_rep.get_batch(batch_size)
        self.net.fit(*batch)

    def train(self, episodes: int, path: str, checkpoint_rate=100, batch_size: int = 64,
              exp_decay_func=lambda exp_rate, exp_decay, i: 0.01 + (exp_rate - 0.01) * np.exp(exp_decay * (i + 1)),
              show_progress=False):
        """
        Runs a training session for the agent
        :param episodes: number of episodes to train.
        :param path: a path to a directory where the trained weights will be saved.
        :param batch_size: number of experiences to learn from in each net_update.
        """
        if self.net is None:
            raise NotImplementedError(
                'agent.train called before model was not initiated.\n Please set the agent\'s model'
                ' using the set_model method. You can access the state and action shapes using '
                'agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        # set hyper parameters
        exploration_rate = self.exploration_rate
        total_rewards = []
        # start training
        for episode in tqdm(range(episodes)):
            state = self.env.reset()  # Reset the environment for a new episode
            step, episode_reward = 0, 0
            run = True
            # Run until max actions is reached or episode has ended
            while run:

                step += 1
                # choose a current action using epsilon greedy exploration
                action = self.get_action(state, exploration_rate)

                # apply the chosen action to the environment and observe the next_state and reward
                obs = self.env.step(action)
                next_state, reward, is_terminal = obs[:3]
                episode_reward += reward

                # Add experience to memory
                self.exp_rep.add(state, action, reward, next_state, self.env.get_legal_actions(state), is_terminal)

                # Optimize the DoubleQ-net
                self.update_net(batch_size)

                if is_terminal:  # The action taken led to a  terminal state
                    run = False

                if (step % self.net_updating_rate) == 0 and step > 0:
                    # update target network
                    self.net.align_target_model()
                state = next_state

            # Update total_rewards to keep track of progress
            total_rewards.append(episode_reward)
            # Update target network at the end of the episode
            self.net.align_target_model()
            # Update exploration rate -
            exploration_rate = exp_decay_func(exploration_rate, self.exploration_decay, episode)

            if episode % checkpoint_rate == 0 and self.exp_rep.get_num() > batch_size:
                self.save_weights(os.path.join(path, f'episode_{episode}_weights'))

                if show_progress:  # Plot a moving average of last 10 episodes
                    self.plot_progress(total_rewards)

        # update the agents exploration rate in case more training is needed.
        self.exploration_rate = exploration_rate

        # saves the total_rewards as csv file to the path specified.
        with open(os.path.join(path, 'rewards.csv'), 'w') as reward_file:
            rewards = pd.DataFrame(total_rewards)
            rewards.to_csv(reward_file)
        self.save_weights(os.path.join(path, 'final_weights'))

    def plot_progress(self, total_rewards):
        w = np.ones(10) / 10
        moving_average = np.convolve(total_rewards, w, mode='valid')
        plt.plot(np.arange(len(moving_average)), moving_average)
        plt.title('Moving average of rewards across episodes')
        plt.xlabel('episodes')
        plt.ylabel('average reward over last 10 episodes')
        plt.show()

    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    # Handles saving\loading the model as explained here: https://www.tensorflow.org/guide/keras/save_and_serialize
    def load_weights(self, path):
        self.net.load_weights(path)

    def save_weights(self, path):
        self.net.save_weights(path)

    def save_model(self, path):
        if self.net is None:
            raise NotImplementedError('agent.save_model was called before model was not initiated.\n Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        self.net.save_model(path)

    def load_model(self, path):
        model = load_model(path)
        self.set_model(model)

    def to_json(self, **kwargs):
        if self.net is None:
            raise NotImplementedError('agent.to_json was called before model was not initiated.\n Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        return self.net.to_json(**kwargs)

    def from_json(self, json_config):
        model = model_from_json(json_config)
        self.set_model(model)
