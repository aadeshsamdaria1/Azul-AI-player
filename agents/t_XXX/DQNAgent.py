from Azul.azul_model import AzulState, GameRule
import Azul.azul_utils as utils
import numpy as np
import tensorflow as tf 
import random
from agents.generic import random
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
MIN_EPSILON = 0.01
MEMORY_CAPACITY = 10000
NUM_EPISODES = 10000
# In general, a typical game of Azul between two skilled players 
# can take anywhere from 20 to 40 per player
MAX_MOVES = 50
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

class DQNTrainModel:
        # game_rule.current_game_state
    def init(self, _id):
        self.id = _id # The agent needs to remember its own id
        # double-ended queue used in the replay buffer of the DQN algorithm. It stores the transitions
        # experienced by the agent, which consist of state, action, reward, next state and whether the next state is terminal
        self.memory = []
        self.game_rule = GameRule(NUM_PLAYERS)
        self.azul_state = self.game_rule.initialGameState()
        # getlegalactions need gamestate
        # gamestate is azul_state
        self.all_possible_actions = self.game_rule.getLegalActions(self.azul_state, self.id)
        self.num_actions = len(self.all_possible_actions)
        # policy network
        self.model = self.build_model(self.num_actions)
        # target network
        self.target_model = self.build_model(self.num_actions)
        # initialize the weights from both policy network and target network



    def build_model(self, num_actions):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_dim=18, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_normal'),
            tf.keras.layers.Dense(num_actions, kernel_initializer='random_normal')
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
        features = np.zeros((NUM_COLOR + NUM_COLOR * NUM_PLAYERS + NUM_PLAYERS + 1, ))

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
            features[colour] += state.centre_pool.tiles

        # add the number of tiles of each player that have been placed on the game board
        # access through self.agents
        # self.lines_tile[line] (AgentState)
        # self.lines_number[line]
        # 0 .. 4 av from fac
        # index 5 -> index 14 (inclusive)
        # NUM_PLAYERS * NUM_COLOR + NUM_COLOR = 2 * 5 + 5
        players_filled_tiles = self.get_player_tiles()
        features[NUM_COLOR : NUM_PLAYERS * NUM_COLOR + NUM_COLOR] = players_filled_tiles

        # Add the current score of each player
        # self.score in agent state
        for i in range(NUM_PLAYERS):
            player = self.agents[i]
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
        features[NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1] = self.id
        return features
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
        if np.random.rand() <= EPSILON:
            return np.random.choice(legal_actions)
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



class DQNTrainFinalPhase: 
    def __init__(self):
        self.dt = DQNTrainModel()
        self.game_rule =  GameRule(NUM_PLAYERS)
    def train(self):
        for episode in range(NUM_EPISODES):
            # intialise everything needed for training
            
            # define the starting state
            state = self.dt.azul_state()
            total_reward = 0
            done = False
            while not done:
                # select action
                action = dt.act()
                # there is no done, but action can be STARTROUND or ENDROUND
                if action == "ENDROUND":
                    done = True 
                # emulate the action in env
                # TOD0: missing agent id (when training), should i simulate states in 
                # both player and enemy
                successor_state = self.dt.game_rule.generateSuccessor(state, action)
                reward = self.reward_function(successor_state)
                # features should be extracted from state and successor_state
                # before putting them into memory
                curr_feature = self.dt.get_features(state)
                next_feature = self.dt.get_features(successor_state)
                self.dt.remember(curr_feature, action, reward, next_feature, done)
                total_reward += reward
                # move to the next state, how? 
                # both current state and next state are azul states

                # get the reward of this successor state
                pass
    def reward_function(self, state, successor_state):
        # Reward for completing a row or a column
        prev_score = state.GetCompletedRows() + state.GetCompletedColumns()
        curr_score = successor_state.GetCompletedRows() + state.GetCompletedColumns()
        # get the total number of actions taken so far in the game
        num_moves = sum(len(round_actions) for round_actions in successor_state.agent_trace.actions)
        if curr_score > prev_score:
            reward = 10
        
        elif num_moves > MAX_MOVES:
            reward = -10
        # Reward for placing the tile in the correct position
        else:
            reward = 1
        return reward
    
class AdvancedGame:
    def __init__(self):
        self.game_rule = GameRule(NUM_PLAYERS)
        self.game_master = DummyAgent()
        self.random_agent = random.myAgent(0)
        self.DQN_agent = DQNTrainModel(1)
        self.agent_list = list()
        self.agent_list.append(self.random_agent)
        self.agent_list.append(self.DQN_agent)
        self.agents = self.agent_list
        self.time_limit = 1
    def _run(self):
        # run one episode
        # define the two agents
       
        while not self.game_rule.gameEnds:
            agent_index = self.game_rule.getCurrentAgentIndex()
            agent = self.agents[agent_index] if agent_index < len(self.agents) else self.gamemaster
            game_state = self.game_rule.current_game_state
            actions = self.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            gs_copy = copy.deepcopy(game_state)
            selected = func_timeout(self.time_limit, agent.SelectMove(), args=(actions_copy, gs_copy))
            if(agent_index == 0):
                pass
           

        