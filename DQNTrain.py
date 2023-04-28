
from Azul.azul_model import AzulState, GameRule, AzulGameRule
import Azul.azul_utils as utils
import numpy as np
import tensorflow as tf 
import random
from template import Agent as DummyAgent
import copy
from   func_timeout import func_timeout, FunctionTimedOut

THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
GRID_SIZE = 5
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.1
MIN_EPSILON = 0.01
MEMORY_CAPACITY = 10000
NUM_EPISODES = 10000
SAVE_FREQUENCY = 10
# In general, a typical game of Azul between two skilled players 
# can take anywhere from 20 to 40 per player
MAX_MOVES = 50
GAMMA = 0.955
NUM_FACTORIES = 5
# DEEPQ LEARNING ALGORITHM
# 1. INITIALIZE REPLAY MEMORY CAPACITY
# 2. INITIALIZE THE NETWORK WITH RANDOM WEIGHTS
# 3. FOR EACH EPISODE:
#   1. Initialize the starting state
#   2. For each time step:
#       1. SELECT AN ACTION
#       2. EXECUTE THE SELECTED ACTION IN THE EMULATOR
#       3. OBSERVE THE REWARD AND THE NEXT STATE
#       4. STORE EXPERIENCE IN REPLAY MEMORY
#       5. SELECT RANDOM BATCH FROM THE REPLAY MEMORY
#       6. PREPROCESS STATES FROM BATCH
#       7. PASS BATCH OF PREPROCESSED STATES TO THE POLICY NETWORK
#       8. CALCULATE LOSS BETWEEN OUTPUT Q-VALUES AND TARGET Q-VALUES
#       9. GRADIENT DESCENT UPDATES WEIGHTS IN THE POLICY NETWORK TO MINIMIZE LOSS
class ActionEncoder:
    # missing one element, num_avail
    # In a 2-player game of Azul, the number of tiles that each type can occur is at most 5, since there are
    # only 20 tiles of each type in a game
    def map_action(self):
        self.action_dict = {}
        index = 0  
        for factory_id in range(NUM_FACTORIES):
            for tile_type in utils.Tile:
                for num_tiles in range(6):
                    for pattern_line_dest in range(GRID_SIZE):
                        for num_to_pattern_line in range(1, GRID_SIZE + 1):
                            for num_to_floor_line in range(num_to_pattern_line, 5):
                                self.action_dict[(factory_id, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line)] = index
                                index += 1
        # include taking from centre actions
        for tile_type in utils.Tile:
            for num_tiles in range(1,21):
                for pattern_line_dest in range(GRID_SIZE):
                        for num_to_pattern_line in range(1, GRID_SIZE + 1):
                            for num_to_floor_line in range(num_to_pattern_line, 5):
                                self.action_dict[(-1, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line)] = index
                                index += 1
        # Get the total number of possible actions
        self.num_actions = len(self.action_dict)

    def one_hot_encoding(self):
        # Loop over all the possible actions and creating a one-hot encoding for that action
        self.actions = []
        for factory_id in range(NUM_FACTORIES):
            for tile_type in utils.Tile:
                for num_tiles in range(6):
                    for pattern_line_dest in range(GRID_SIZE):
                            for num_to_pattern_line in range(1, GRID_SIZE + 1):
                                for num_to_floor_line in range(num_to_pattern_line, 5):
                                    action = (factory_id, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line)
                                    action_index = self.action_dict[action]
                                    one_hot = np.zeros(self.num_actions)
                                    one_hot[action_index] = 1
                                    self.actions.append(one_hot)
        # include taking from centre actions
        for tile_type in utils.Tile:
            for num_tiles in range(6):
                for pattern_line_dest in range(GRID_SIZE):
                    for num_to_pattern_line in range(1, GRID_SIZE + 1):
                        for num_to_floor_line in range(num_to_pattern_line, 5):
                            action = (-1, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line)
                            action_index = self.action_dict[action]
                            one_hot = np.zeros(self.num_actions)
                            one_hot[action_index] = 1
                            self.actions.append(one_hot)

            
    def one_hot_to_actualaction(self, one_hot_encoding):
        # one hot encoding is an np array of zeroes
        index = np.argmax(one_hot_encoding)
        for key, value in self.action_dict.items():
            if value == index:
                flattened_action = key
            pass
        id, tile_type, num_avail, pattern_line_dest, num_to_pattern_line, num_to_floor_line = flattened_action
        tg =  utils.TileGrab()
        tg.tile_type = tile_type
        tg.number = num_avail
        tg.num_to_floor_line = num_to_floor_line
        tg.num_to_pattern_line = num_to_pattern_line
        tg.pattern_line_dest = pattern_line_dest


class DQNAgent:
        # game_rule.current_game_state
    def __init__(self, _id, _all_possible_actions):
        self.id = _id # The agent needs to remember its own id
        # double-ended queue used in the replay buffer of the DQN algorithm. It stores the transitions
        # experienced by the agent, which consist of state, action, reward, next state and whether the next state is terminal
        self.memory = []
        self.gamma =  GAMMA
        # learning rate is not used
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = MIN_EPSILON
        self.game_rule = AzulGameRule(NUM_PLAYERS)
        # getlegalactions need gamestate
        # gamestate is azul_state
        self.all_possible_actions = _all_possible_actions
        self.num_actions = len(self.all_possible_actions)
        # policy network
        self.model = self.build_model(self.num_actions)
        # target network
        self.target_model = self.build_model(self.num_actions)
        self.action_encoder = ActionEncoder()
        self.action_encoder.map_action()
        self.action_encoder.one_hot_encoding()
        # initialize the weights from both policy network and target network



    def build_model(self, num_actions):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(19,), activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(self.action_encoder.num_actions, kernel_initializer='random_normal')
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
    def get_features(self, state):
        # Extract relevant features for the Azul game state

        # Initialize the feature vector with zeroes
        # Features included:
        # 1. The number of tiles in each colour that have not been placed on the game board (5)
        # 2. The number of tiles in each colour that each player has already placed individually
        # on their game board (10)
        # 3. The current score for each player (2)
        # 4. The id of the current player (1)

        # Initialize the feature vector with zeros
        # 5 + 5 * 2 + 2 + 1
        features = np.zeros((NUM_COLOR + NUM_COLOR * NUM_PLAYERS + NUM_PLAYERS + 2, ))
        #print("Actual size: ", len(features))
        # Add the number of tiles in each colour that have not yet been placed on the game board

        # add available tiles from the factory first
        # get the total tiles from all factories for later use
        factory_tile_count = 0
        for factory in state.factories: 
            for colour in utils.Tile:
                features[colour] = factory.tiles[colour]
                factory_tile_count += factory.tiles[colour]
                # factory from state.factories
                # state.centre_pool.tiles
                # factory.tiles
        # also add from the centre
        for colour in utils.Tile:
            features[colour] += state.centre_pool.tiles[colour]

        # add the number of tiles of each player that have been placed on the game board
        # access through self.agents
        # self.lines_tile[line] (AgentState)
        # self.lines_number[line]
        # 0 .. 4 av from fac
        # index 5 -> index 14 (inclusive)
        # NUM_PLAYERS * NUM_COLOR + NUM_COLOR = 2 * 5 + 5
        players_filled_tiles = self.get_player_tiles(state)
        features[NUM_COLOR : NUM_PLAYERS * NUM_COLOR + NUM_COLOR] = players_filled_tiles

        # Add the current score of each player
        # self.score in agent state
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + i]  = player.score
        # Add the current round in the game
        # AgentState.agent_trace.round_scores
        current_agent = state.agents[self.id]

        # records the scores in each round
        # if the length of round_scores is 5, it means 5 rounds have been completed and 
        # we are currently in the sixth round
        round_scores = current_agent.agent_trace.round_scores
        current_round = len(round_scores) + 1
        features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS] = current_round
        # Add the current player's id
        #print("Accessed index: ", NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1)
        features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1] = self.id
        return features
    # TODO: may be wrong
    def get_player_tiles(self, state):
        # initialize a numpy array of size 5 with zeroes
        player_filled_tile = np.zeros((NUM_PLAYERS * NUM_COLOR, ))
        for i in range(NUM_PLAYERS):
            player = state.agents[i]
            # go through each line of the board
            for j in range(GRID_SIZE):
                # if the tile is at least filled with one colour
                if(player.lines_tile[j] != -1):
                    tile_color = player.lines_tile[j]
                    num_filled = player.lines_number[j]
                    player_filled_tile[i * NUM_COLOR + tile_color] = num_filled
        return player_filled_tile


    # act function is responsible for selecting an action to take in the current state. 
    # It does this by either selecting a random action with a certain probabilty (exploration), 
    # or selecting the action with the highest Q-value as determined by the neural network (exploitation)
    def act(self, state):
        # This line of code implements the epsilon greedy policy
        # The function np.random.rand() generates a random number between 0 and 1
        # and epsilon is the probability of choosing a random action instead of an action
        # with the highest Q-value

        # If the random number is less than or equal to EPSILON, the agent will explore the game space
        # by occassionally selecting random actions 
        
        # Get legal actions
        legal_actions = self.game_rule.getLegalActions(state, self.id)
        #print("Legal actions: ", legal_actions)
        if np.random.rand() <= EPSILON:
            return random.choice(legal_actions)
        # TODO: need to change this
        # Extract relevant features for the Azul game state
        features = self.get_features(state)
        # Reshape the feature vector to be used as an input to the model
        features = features.reshape(1, -1)
        # Use the model to get Q-values for the current state
        q_values = self.model.predict(features)[0]
        # set q-values for unavailable actions to very low values
        for i in range(self.num_actions):
            if self.all_possible_actions[i] not in legal_actions:
                q_values[i] = -9999
        # choose an action with the highest Q-value
        highest_q_index = np.argmax(q_values)
        return self.all_possible_actions[highest_q_index]

    # replay function is responsible for training the neural network based on a batch of experiences 
    # sampled randomly from the agent's memory. This is done in order to update the Q-values of the neural 
    # network and improve the agent's policy. 
    # TODO : check this
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        input_data = np.array([data[0] for data in minibatch])
        target_data = self.model.predict(input_data)
        print("Done")
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            print("Next state shape: ", next_state.shape)
            # target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            if not done: 
                next_target = np.max(self.model.predict(np.array([next_state]))[0])
                print("Next target predicted")
                target = reward + self.gamma * next_target
                
            else: 
                target = reward
            action_index = self.all_possible_actions.index(action)
            target_data[i][action_index] = target
            # need to get the index of this action 
        self.model.fit(input_data, target_data, epochs=1, verbose=0)
        self.update_target_model()
    def load(self, name):
        # load the model weights from a file
        self.model.load_weights(name)
    def save(self, name):
        # save the model weights to a file
        self.model.save_weights(name)
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def reward_function(self, state, successor_state, is_timed_out):
        # Reward for completing a row or a column
        # both state and successor_state are Azul state
        prev_score = state.agents[self.id].GetCompletedRows() + state.agents[self.id].GetCompletedColumns()
        curr_score = successor_state.agents[self.id].GetCompletedRows() + state.agents[self.id].GetCompletedColumns()
        # get the total number of actions taken so far in the game
        num_moves = sum(len(round_actions) for round_actions in successor_state.agents[self.id].agent_trace.actions)
        reward = 0
        if curr_score > prev_score:
            reward += 10
        else:
            reward += 1
        if num_moves > MAX_MOVES:
            reward += -10
        # Reward for placing the tile in the correct position
        if is_timed_out:
            reward -= 10
       

        return reward
    def get_all_actions(self):
         self.game_rule = AzulGameRule(NUM_PLAYERS)
         all_possible_actions = self.game_rule.getLegalActions(self.game_rule.initialGameState(),0)
         return all_possible_actions
    def train(self):
        all_possible_actions = self.get_all_actions()
        self.agent1 = DQNAgent(0, all_possible_actions)
        self.agent2 = DQNAgent(1, all_possible_actions)
        for episode in range(NUM_EPISODES):
                game = AdvancedGame(self.agent1, self.agent2)
                game._run()
                if self.agent1.epsilon > self.agent1.epsilon_min:
                    self.agent1.decay_epsilon()
                if self.agent2.epsilon > self.agent2.epsilon_min:
                    self.agent2.decay_epsilon()
                if episode % SAVE_FREQUENCY == 0:
                    # save the policy for both agents
                    self.agent1.save("policymodel1.h5")
                    self.agent2.save("policymodel2.h5")
                    


 
        
    
    
class AdvancedGame:
    def __init__(self, _agent1, _agent2):
        self.game_rule = AzulGameRule(NUM_PLAYERS)
        self.agent1 = _agent1
        self.agent2 = _agent2
        self.game_master = DummyAgent(NUM_PLAYERS) # Used for signalling rounds in Azul
        self.time_limit = 1
        # An agent state is initialised if there is a azulstate
        # but how does it connect with the agent defined
        self.time_limit = 1
     
        self.agent_list = list()
        self.agent_list.append(self.agent1)
        self.agent_list.append(self.agent2)
    def _run(self):
       

        # run one episode
        # use self-play strategy
        # define the two agents
        state = self.game_rule.current_game_state
        agent_turn = 0
        print("Start game")
        # start round for all players
        for player in state.agents:
            player.agent_trace.StartRound()
        done = False
        while not done:
            current_agent = self.agent_list[agent_turn]
            # give one second limit to choose action
            is_timed_out = False
            try: 
                action = func_timeout(self.time_limit, current_agent.act, args= (state,))
            except: 
                action = current_agent.act(state)
                is_timed_out = True
            if action == "ENDROUND":
                print("End of round")
                for player in state.agents:
                    player.agent_trace.StartRound()
                for fd in state.factories:
                    state.InitialiseFactory(fd)
                for tile in utils.Tile:
                    state.centre_pool.tiles[tile] = 0
            # successor state
            #self.game_rule.update(action)
            # print(len(state.agents))
            # print("Agent turn: ", agent_turn)
            # print("Agent trace: ", state.agents[agent_turn].agent_trace.actions)
            next_state = self.game_rule.generateSuccessor(state,action,agent_turn)
            reward = current_agent.reward_function(state, next_state, is_timed_out)
            # features should be extracted from state and successor_state
            # before putting them into memory
            curr_feature = current_agent.get_features(state)
            next_feature = current_agent.get_features(next_state)
            current_agent.remember(curr_feature, action, reward, next_feature, done)
            current_agent.replay()
            # update the state variable
            state = next_state
            # update the agent id
            agent_turn = (agent_turn + 1) % NUM_PLAYERS
            
            done = self.game_rule.gameEnds()
            # calScore
        agent_1_complete_reward = self.game_rule.calScore(state,0)
        agent_2_complete_reward = self.game_rule.calScore(state, 1)
        return [agent_1_complete_reward, agent_2_complete_reward]
# class ActualDQNAgent:

      
if __name__ == "__main__":  
    # testing for one episode
    game_rule = AzulGameRule(NUM_PLAYERS)
    all_possible_actions = game_rule.getLegalActions(game_rule.initialGameState(),0)
    # print(all_possible_actions)
    agent1 = DQNAgent(0, all_possible_actions)
    agent2 = DQNAgent(1, all_possible_actions)
    game = AdvancedGame(agent1, agent2)
    game._run()


        