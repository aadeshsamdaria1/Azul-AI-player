from tensorflow import keras 
from DQNTrain import ActionEncoder
from Azul.azul_model import AzulState, GameRule, AzulGameRule
import Azul.azul_utils as utils
import numpy as np
import os
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
class DQNPlayer():
    def __init__(self, _id):
        self.id = _id
        self.action_encoder = ActionEncoder()
        self.action_encoder.map_action()
        self.game_rule = GameRule()
        file_path1 = "policymodel1.h5"
        file_path2 = "policymodel1.h5"
        if os.path.exists(file_path1): 
            self.model = keras.model.load(file_path1)
        if os.path.exists(file_path2):
            self.model - keras.model.load(file_path2)

    pass
    def get_features(self, state):
        features = np.zeros((NUM_COLOR + NUM_COLOR * NUM_PLAYERS + NUM_PLAYERS + 2, ))
        factory_tile_count = 0
        for factory in state.factories: 
            for colour in utils.Tile:
                features[colour] = factory.tiles[colour]
                factory_tile_count += factory.tiles[colour]
        for colour in utils.Tile:
            features[colour] += state.centre_pool.tiles[colour]
        players_filled_tiles = self.get_player_tiles(state)
        features[NUM_COLOR : NUM_PLAYERS * NUM_COLOR + NUM_COLOR] = players_filled_tiles
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + i]  = player.score
        current_agent = state.agents[self.id]
        round_scores = current_agent.agent_trace.round_scores
        current_round = len(round_scores) + 1
        features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS] = current_round
        features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1] = self.id
        return features
    def SelectAction(self,actions,game_state):
        # returns one action at a time
        legal_actions = self.game_rule.getLegalActions(game_state, self.id)
        features = self.get_features(game_state)
        features = features.reshape(1, -1)
        q_values = self.model.predict(features)[0]
        max_q_val = float("-inf")
        if len(legal_actions) == 1:
            return legal_actions
        for action in legal_actions:
            action_type = action[0]
            id = action[1]
            tg = action[2]
            tile_type = tg.tile_type
            num_tiles = tg.number
            pattern_line_dest = tg.pattern_line_dest
            num_to_pattern_line = tg.num_to_pattern_line
            num_to_floor_line = tg.num_to_floor_line
            index = self.action_encoder.action_dict[(action_type, id, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line)]
            q_val = q_values[index]
            if q_val > max_q_val:
                max_q_index = index
        flattened_action = self.action_encoder.reverse_action_dict[max_q_index]
        action_type, id, tile_type, num_avail, pattern_line_dest, num_to_pattern_line, num_to_floor_line = flattened_action
        tg =  utils.TileGrab()
        tg.tile_type = tile_type
        tg.number = num_avail
        tg.num_to_floor_line = num_to_floor_line
        tg.num_to_pattern_line = num_to_pattern_line
        tg.pattern_line_dest = pattern_line_dest
        return(action_type, id, tg)
