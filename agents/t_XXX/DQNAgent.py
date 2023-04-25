from Azul.azul_model import AzulState, GameRule
import Azul.azul_utils as utils
import numpy as np
import tensorflow as tf 
import random
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
MEMORY_CAPACITY = 10000
NUM_EPISODES = 10000
class DQNAgent:
        
    def init(self, _id):
        self.id = _id # The agent needs to remember its own id
        self.model = self.build_model()
        self.target_model = self.build_model()
        # double-ended queue used in the replay buffer of the DQN algorithm. It stores the transitions
        # experienced by the agent, which consist of state, action, reward, next state and whether the next state is terminal
        self.memory = []
        self.game_rule = GameRule(NUM_PLAYERS)
        self.azul_state = AzulState()



    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_dim=25, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # the append method is used to add transition into memory , and when the memory is full,
    # the oldest memory is removed
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.pop(0)
    def get_features(state):
        # Extract relevant features for the Azul game state

        # Initialize the feature vector with zeroes
        # Features included:
        # 1. The number of tiles in each colour that have not been placed on the game board (5)
        # 2. The number of tiles in each colour that each player has already placed individually
        # on their game board (10)
        # 3. The current score for each player (2)
        # 4. The id of the current player (1)

        # Initialize the feature vector with zeros
        features = np.zeros((NUM_COLOR + NUM_COLOR * NUM_PLAYERS + NUM_PLAYERS + 2, ))

        # Add the number of tiles in each colour that have not yet been placed on the game board

        # add available tiles from the factory first
        for factory in state.factories: 
            for colour in utils.Tile:
                features[colour] = factory.tiles[colour]
                # factory from state.factories
                # state.centre_pool.tiles
                # factory.tiles
        # also add from the centre
        for colour in utils.Tile:
            features[colour] += state.centre_pool.tiles

        
        # add the number of tiles of each player that have been placed on the game board
        # access through self.agents
        # self.lines_tile[line] (AgentState)
        # self.lines_number[line]
        for player in range(NUM_PLAYERS):
            for colour2 in range(NUM_COLOR):
                features[colour + player * NUM_COLOR + colour2] 
                pass
    def get_player_tiles(self):
        agents = self.azul_state.agents
        # initialize a numpy array of size 5 with zeroes
        
        for i in range(NUM_PLAYERS):
            player = agents[i]




    def act(self, state):
        if np.random.rand() <= EPSILON:
            return self.game_rule.getLegalActions(state, self.id)
        # TODO: need to change this
        q_values = self.model.predict(state.reshape(1, -1))[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        samples = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        q_values = self.model.predict(states)
        target_q_values = self.target_model.predict(next_states)
        max_q_values = np.max(target_q_values, axis=1)
        target_q_values[dones, actions] = rewards[dones] + DISCOUNT_FACTOR * max_q_values[dones]
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

    def train(self):
        pass