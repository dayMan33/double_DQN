from tensorflow.keras import Model
from tensorflow.keras.models import clone_model, load_model, model_from_json
import numpy as np


class DoubleDQN:
    def __init__(self, model: Model, discount: float = 0.95):
        """
        C-tor
        :param model: A compiled tensorflow  model
        :param discount: Determines how future rewards are evaluated in comparison to immediate rewards.
        """
        self.discount = discount
        self.q_net = None
        self.target_net = None
        self._set_model(model)

    def align_target_model(self) -> None:
        """
        Sets the target net weights to be the same as the q-net.
        """
        self.target_net.set_weights(self.q_net.get_weights())

    def predict(self, states: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
        """
        Given a state and legal actions that can be taken from that step, calculates the Q-values of (state, action) for
        each action that can be taken (illegal actions are evaluated as 0). Also works for a batch of states.
        :param states: a batch of states with shape (batch_size, state_shape)
        :param legal_actions: a batch of boolean vectors representing the legal actions from each step.
        :return: A numpy.ndarray representing the Estimated Q-values of all actions that can be taken from the specified
                state
        """
        q_values = self.q_net.predict(states)

        # setting q_values of illegal actions to lowest possible value to make sure they are not chosen
        illegal = np.where(np.logical_not(legal_actions))
        q_values[illegal[0], illegal[1]] = -np.inf
        return q_values

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
            legal_actions: np.ndarray, terminal: np.ndarray) -> None:
        """
        Updates the net according to the Double Q-learning paradigm.
        :param states: A batch of states.
        :param actions: A batch of actions taken from the specified states.
        :param next_states: the observed next-states after taking the specified actions from the specified states
                (Expects None if the state was a terminal state).
        :param rewards: A batch of rewards given after taking soecified actions from specified states and transitioning
                to specified next_states.
        :param legal_actions: A batch of legal actions that are allowed from the specified states.
        """
        if self.q_net is None:
            raise NotImplementedError('model was not initiated')
        targets = self.predict(states, legal_actions)

        # Create mask for separating terminal states from non-terminal states.
        non_terminal_mask = np.where(np.logical_not(terminal))[0]

        # Set the targets in the actions taken to be the rewards. For non terminal states we will also add the Q-value
        # estimation of the next state
        targets[..., actions] = rewards

        # Calculate the expected sum of rewards based on the target-net and q-net
        t = self.target_net.predict(np.array(list(next_states[non_terminal_mask])))
        q = self.q_net.predict(np.array(list(next_states[non_terminal_mask])))

        # Double-Q learning paradigm. We choose the actions based on the q_net evaluation, but evaluate the values of
        # the state-action pair using the target net
        max_actions = np.argmax(q, axis=-1)
        estimated_values = t[np.arange(t.shape[0]), max_actions]

        # Set the target value of the non-terminal sates to be:
        # target(state, action) = Reward(state, action) + (discount * target_net(next_state, argmax(q_net(next_state))))
        targets[non_terminal_mask, actions[non_terminal_mask]] += self.discount * estimated_values

        # At this point the target is similar to the q-net prediction, except in the index corresponding to action
        # taken. In this index, the target value is just the reward if state is terminal, otherwise it is
        # reward + discount * Q(next_state, action), where Q(next_state, action) is evaluated using the Double
        # Q-learning algorithm. that is Q(next_state, action) = target_net(next_state)[argmax(q_net(next_state))]

        self.q_net.fit(states, targets, epochs=1, verbose=0)

    def _set_model(self, model: Model):
        """
        Sets the inner model to the specified model
        :param model: A compiled tf model.
        :return:
        """
        self.q_net = model
        self.target_net = clone_model(self.q_net)

    # This handles saving\loading the model as explained here:
    # https://www.tensorflow.org/guide/keras/save_and_serialize (Ctrl+Left_click to open)

    def load_weights(self, path):
        self.q_net.load_weights(path)
        self.target_net = clone_model(self.q_net)

    def save_weights(self, path):
        self.q_net.save_weights(path)

    def to_json(self, **kwargs):
        return self.q_net.to_json(**kwargs)

    def from_json(self, json_config):
        self._set_model(model_from_json(json_config))

    def save_model(self, path):
        self.q_net.save(path)

    def load_model(self, path):
        self._set_model(load_model(path))
