import abc


class DQNEnv:
    message = 'Method not implemented, Please implement this method'

    def __init__(self):
        self.state = None

    @abc.abstractmethod
    def reset(self):
        """ Resets the environmentand returns the initial state """
        raise NotImplementedError(DQNEnv.message)

    @abc.abstractmethod
    def step(self, action):
        """ Advances the state of the environment by taking action. Returns the next state and the reward achieved from
            taking specified action from self.state as well as a boolean parameter indicating whether this episode
            should be terminated."""
        raise NotImplementedError(DQNEnv.message)

    @abc.abstractmethod
    def get_legal_actions(self, state):
        """ Returns a boolean vector representing the legal actions that can be taken from specified state"""
        raise NotImplementedError(DQNEnv.message)

    @abc.abstractmethod
    def get_state_shape(self):
        """Returns the shape of a state in the environment. This should be used to set the network input shape"""
        raise NotImplementedError(DQNEnv.message)

    @abc.abstractmethod
    def get_action_shape(self):
        """Returns the shape of the environment's action space. This should be used to set the network output shape"""
        raise NotImplementedError(DQNEnv.message)
