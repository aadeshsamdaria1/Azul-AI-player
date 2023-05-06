from tensorflow import keras
from DQNTrain import ActionEncoder
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils
import numpy as np
import os
import h5py
import tensorflow as tf
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
GRID_SIZE = 5
NUM_FACTORIES = 5
LEARNING_RATE = 0.01

class myAgent():
    def __init__(self, _id):
        self.id = _id
        file_path1 = "policymodel1.h5"
        self.learning_rate = LEARNING_RATE
        # Create a dictionary to map layer names in the HDF5 file to layers in the model
        self.action_encoder = ActionEncoder()
        self.action_encoder.map_action()
        self.game_rule = GameRule(NUM_PLAYERS)
        file_path2 = "policymodel2.h5"
        weights = {}
        target_layer = list()
        self.model = self.build_model()
        for layer in self.model.layers:
            target_layer.append(layer)
        with h5py.File(file_path1, 'r') as f:
            try:
                count = 0
                for layer_name in f.keys():
                    layer_name = "/" + layer_name + "/" + layer_name
                    group = f.get(layer_name)
                    # print("layer name: ", layer_name, "Kernel: ", group['kernel:0'][()])
                    weights = [group['kernel:0'][()], group['bias:0'][()]]
                    target_layer[count].set_weights(weights)
                    # print("Target layer weight: ", target_layer[count].get_weights())
                    # print("Weight: ", weights)
                    count += 1

            except:
                print("Please ignore this")
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape = (84,), activation = 'relu'))
        model.add(tf.keras.layers.Dense(12, activation = 'relu'))
        model.add(tf.keras.layers.Dense(self.action_encoder.num_actions, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    def get_features(self, state):
        features = np.zeros((NUM_PLAYERS * NUM_COLOR * GRID_SIZE +
                            NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 2, ))
        factory_tile_count = 0
        for factory_num, factory in enumerate(state.factories):
            for colour in utils.Tile:
                features[factory_num * NUM_COLOR +
                    colour] = factory.tiles[colour]
                factory_tile_count += factory.tiles[colour]
        for colour in utils.Tile:
            features[NUM_FACTORIES * NUM_COLOR +
                colour] += state.centre_pool.tiles[colour]
        players_filled_tiles = self.get_player_tiles(state)
        features[NUM_FACTORIES * NUM_COLOR + NUM_COLOR: NUM_PLAYERS * NUM_COLOR *
            GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR] = players_filled_tiles
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE +
                NUM_FACTORIES * NUM_COLOR + NUM_COLOR + i] = player.score
        current_agent = state.agents[self.id]
        round_scores = current_agent.agent_trace.round_scores
        current_round = len(round_scores) + 1
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES *
            NUM_COLOR + NUM_COLOR + NUM_PLAYERS] = current_round
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES *
            NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1] = self.id
        return features

    def get_player_tiles(self, state):
        player_filled_tile = np.zeros((NUM_PLAYERS * NUM_COLOR * GRID_SIZE, ))
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            for j in range(GRID_SIZE):
                if(player.lines_tile[j] != -1):
                    tile_color = player.lines_tile[j]
                    num_filled = player.lines_number[j]
                    # update the player_filled_tile array with the specific tile colour information
                    player_filled_tile[i * (NUM_COLOR * GRID_SIZE) +
                                            tile_color * GRID_SIZE + j] = num_filled
        return player_filled_tile

    def SelectAction(self, actions, game_state):
      legal_actions = self.game_rule.getLegalActions(game_state, self.id)
      for action in legal_actions:
        if action == "ENDROUND":
                return action

      features = self.get_features(game_state)
      features = features.reshape(1, -1)
      q_values = self.model.predict(features, verbose=0)[0]
      max_q_val = float("-inf")
        # need to account for end action
      for action in legal_actions:
            action_type = action[0]
            id = action[1]
            tg = action[2]
            tile_type = tg.tile_type
            num_tiles = tg.number
            pattern_line_dest = tg.pattern_line_dest
            # print("Pattern line dest: ", pattern_line_dest)
            num_to_pattern_line = tg.num_to_pattern_line
            num_to_floor_line = tg.num_to_floor_line
            index = self.action_encoder.action_dict[(
                tile_type, pattern_line_dest, num_to_pattern_line)]
            # get this index from the q-values
            q_val = q_values[index]
            if q_val > max_q_val:
                max_action = (action_type, id, tg)
      return max_action