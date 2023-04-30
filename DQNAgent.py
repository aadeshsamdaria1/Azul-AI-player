from tensorflow import keras
from DQNTrain import ActionEncoder
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils
import numpy as np
import os
import h5py
import tensorflow as tf
import random
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
GRID_SIZE = 5
NUM_FACTORIES = 5
LEARNING_RATE = 0.01
EPSILON_VALUE = 0.2
class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.epsilon = EPSILON_VALUE
        self.learning_rate = LEARNING_RATE
        # Create a dictionary to map layer names in the HDF5 file to layers in the model
        self.action_encoder = ActionEncoder()
        self.action_encoder.map_action()
        self.game_rule = GameRule(NUM_PLAYERS)
        self.validAction = self.game_rule.validAction
        weights = {}
        target_layer = list()
        self.model = self.build_model()
        for layer in self.model.layers:
            target_layer.append(layer)
        file_path = "policymodel.h5"
        with h5py.File(file_path, 'r') as f:
            print("Hello")
            try:
                count = 0
                for layer_name in f.keys():
                    layer_name = "/" + layer_name + "/" + layer_name
                    group = f.get(layer_name)
                    # print("layer name: ", layer_name, "Kernel: ", group['kernel:0'][()])
                    print("Shape: ", np.shape(group['kernel:0'][()]))
                    weights = [group['kernel:0'][()], group['bias:0'][()]]
                    # print("Target layer: ", target_layer)
                    try:
                        target_layer[count].set_weights(weights)
                        # print("Target layer weight: ", target_layer[count].get_weights())
                        # print("Weight: ", weights)
                    except:
                        print("Update error")
                    count += 1

            except:
                print("Please ignore this")
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(150, input_shape=(114,), activation='relu'))
        model.add(tf.keras.layers.Dense(150, activation='relu'))
        model.add(tf.keras.layers.Dense(150, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    # include build model and get features
    def get_features(self, state):
        # Extract relevant features for the Azul game state

        # Initialize the feature vector with zeroes
        # Features included:
        # 1. The number of tiles in each colour that have not been placed on the game board (5 * 5)
        # 2. The number of tiles in each colour that each player has already placed individually
        # on their game board (10)
        # 3. The current score for each player (2)
        # 4. The id of the current player (1)

        # Initialize the feature vector with zeros
        # 5 + 5 * 2 + 2 + 1
        features = np.zeros((NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + GRID_SIZE * GRID_SIZE + GRID_SIZE + 2, ))
        #print("Actual size: ", len(features))
        # Add the number of tiles in each colour that have not yet been placed on the game board

        # add available tiles from the factory first
        # get the total tiles from all factories for later use
        factory_tile_count = 0
        # there are five factories in a five player game
        for factory_num, factory in enumerate(state.factories): 
            # there are 5 types of tiles
            for colour in utils.Tile:
                features[factory_num * NUM_COLOR + colour] = factory.tiles[colour]
                factory_tile_count += factory.tiles[colour]
                # factory from state.factories
                # state.centre_pool.tiles
                # factory.tiles
        # also add from the centre
        for colour in utils.Tile:
            features[NUM_FACTORIES * NUM_COLOR + colour] += state.centre_pool.tiles[colour]

        # add the number of tiles of each player that have been placed on the game board
        # access through self.agents
        # self.lines_tile[line] (AgentState)
        # self.lines_number[line]
        # size: NUM_PLAYERS * NUM_COLOR * GRID_SIZE
        players_filled_tiles = self.get_player_tiles(state)
        features[NUM_FACTORIES * NUM_COLOR  + NUM_COLOR : NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR] = players_filled_tiles

        # Add the current score of each player
        # self.score in agent state
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + i]  = player.score
        
        # Add the agent's wall grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + i * GRID_SIZE + j] = state.agents[self.id].grid_state[i][j]
        # Add the current round in the game
        # AgentState.agent_trace.round_scores
        current_agent = state.agents[self.id]

        # records the scores in each round
        # if the length of round_scores is 5, it means 5 rounds have been completed and 
        # we are currently in the sixth round
        round_scores = current_agent.agent_trace.round_scores
        current_round = len(round_scores) + 1
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + GRID_SIZE * GRID_SIZE + GRID_SIZE] = current_round
        # Add the current player's id
        #print("Accessed index: ", NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1)
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + GRID_SIZE * GRID_SIZE + GRID_SIZE + 1] = self.id
        return features
    # TODO: may be wrong
    def get_player_tiles(self, state):
        # initialize a numpy array of size (NUM_PLAYERS * NUM_COLOR * GRID_SIZE)
        player_filled_tile = np.zeros((NUM_PLAYERS * NUM_COLOR * GRID_SIZE, ))
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            # go through each line of the board
            for j in range(GRID_SIZE):
                # if the tile is at least filled with one colour
                if(player.lines_tile[j] != -1):
                    tile_color = player.lines_tile[j]
                    num_filled = player.lines_number[j]
                    # update the player_filled_tile array with the specific tile colour information
                    player_filled_tile[i * (NUM_COLOR * GRID_SIZE) + tile_color * GRID_SIZE + j] = num_filled
        return player_filled_tile
    
    def SelectAction(self, legal_actions, game_state):
        # if np.random.rand() <= self.epsilon:
        #         return random.choice(legal_actions)
        max_action = random.choice(legal_actions)
        features = self.get_features(game_state)
        features = features.reshape(1, -1)
        q_values = self.model.predict(features, verbose = 0)[0]
        max_q_val = float("-inf")
        # need to account for end action
        for action in legal_actions:
            # too many values to unpack
            if isinstance(action, tuple):
                action_type = action[0]
                id = action[1]
                tg = action[2]
                tile_type = tg.tile_type
                num_tiles = tg.number
                pattern_line_dest = tg.pattern_line_dest
                if pattern_line_dest >= 0:
                    num_to_pattern_line = tg.num_to_pattern_line
                    num_to_floor_line = tg.num_to_floor_line
                    index = self.action_encoder.action_dict[(tile_type, pattern_line_dest, num_to_pattern_line)]
                    # get this index from the q-values
                    q_val = q_values[index]
                    if q_val > max_q_val:
                        max_action = (action_type, id, tg)
        return max_action
        
    

    
agent = myAgent(0)