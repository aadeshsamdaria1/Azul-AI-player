
from Azul.azul_model import AzulState, GameRule, AzulGameRule
import Azul.azul_utils as utils
import numpy as np
import tensorflow as tf 
import random
from template import Agent as DummyAgent
import copy
import time
from   func_timeout import func_timeout, FunctionTimedOut
import os
import contextlib
import matplotlib.pyplot as plt
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
GRID_SIZE = 5
BATCH_SIZE = 1000
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.001
MIN_EPSILON = 0.01
MEMORY_CAPACITY = 10000
NUM_EPISODES = 1000
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
        # for abstraction, just get pattern_line_dest, num_to_pattern_line and num_to_floor_line
        for pattern_line_dest in range(-1, GRID_SIZE):
            for num_to_pattern_line in range(0, GRID_SIZE + 1):
                    for num_to_floor_line in range(0, GRID_SIZE + 3):
                        self.action_dict[(pattern_line_dest, num_to_pattern_line, num_to_floor_line)] = index
                        index += 1

        # Get the total number of possible actions
        print("Size of action space: ", len(self.action_dict))
        with open("action_space.txt","w") as f:
            f.write(str(self.action_dict))
        self.num_actions = len(self.action_dict)

class DQNAgent:
        # game_rule.current_game_state
    def __init__(self, _id):
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
        # getlegalactions need gamestate
        # gamestate is azul_state
        # policy network
        self.action_encoder = ActionEncoder()
        self.action_encoder.map_action()
        # with open("action_dict.txt", "w") as f:
        #     f.write(str(self.action_encoder.action_dict))
        self.model = self.build_model()
        # target network
        self.target_model = self.build_model()
        # initialize the weights from both policy network and target network
        # collects the rewards from each episode
        self.episode_total_reward = 0
        self.rewards = []


    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape = (86,), activation = 'relu'))
        model.add(tf.keras.layers.Dense(12, activation = 'relu'))
        model.add(tf.keras.layers.Dense(self.action_encoder.num_actions, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # the append method is used to add transition into memory , and when the memory is full,
    # the oldest memory is removed
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_CAPACITY:
            popped_memory = self.memory.pop(0)
            print("Popped memory: ", popped_memory)
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
        features = np.zeros((NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 4, ))
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
        # Add the current round in the game
        # AgentState.agent_trace.round_scores
        current_agent = state.agents[self.id]

        # records the scores in each round
        # if the length of round_scores is 5, it means 5 rounds have been completed and 
        # we are currently in the sixth round
        round_scores = current_agent.agent_trace.round_scores
        current_round = len(round_scores) + 1
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS] = current_round
        # Add the current player's id
        #print("Accessed index: ", NUM_PLAYERS * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1)
        features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 1] = self.id

        # extract the number of tiles taken in the previous action in the game state
        try: 
            previous_action = current_agent.agent_trace.actions[-1][-1]
            if(len(previous_action) > 1):
                tg = previous_action[2]
                number = tg.number
                tile_type = tg.tile_type
                features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 2] = number
                fearures[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 3] = tile_type
            else:
                features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 2] = 0
                features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 3] = -1
        except:
            features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 2] = 0
            features[NUM_PLAYERS * NUM_COLOR * GRID_SIZE + NUM_FACTORIES * NUM_COLOR + NUM_COLOR + NUM_PLAYERS + 3] = -1
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


    # act function is responsible for selecting an action to take in the current state. 
    # It does this by either selecting a random action with a certain probabilty (exploration), 
    # or selecting the action with the highest Q-value as determined by the neural network (exploitation)
    def act(self, state, game_rule):
        # This line of code implements the epsilon greedy policy
        # The function np.random.rand() generates a random number between 0 and 1
        # and epsilon is the probability of choosing a random action instead of an action
        # with the highest Q-value

        # If the random number is less than or equal to EPSILON, the agent will explore the game space
        # by occassionally selecting random actions 
        # Get legal actions
        legal_actions = game_rule.getLegalActions(state, self.id)
        for action in legal_actions:
            if action == "ENDROUND":
                return action
            
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)
        print("Finally random action is not chosen")
        # TODO: need to change this
        # Extract relevant features for the Azul game state
        features = self.get_features(state)
        # Reshape the feature vector to be used as an input to the model
        features = features.reshape(1, -1)
        # Use the model to get Q-values for the current state
        q_values = self.model.predict(features)[0]
        # set q-values for unavailable actions to very low values
        
        # TODO: test this part 
        # flatten the legal actions
        
        max_q_val = float("-inf")
        # need to account for end action
        for action in legal_actions:
            # too many values to unpack
            if len(action) == 3:
                action_type = action[0]
                id = action[1]
                try:
                    tg = action[2]
                except:
                    print("error caught")
                    print("Action: ", action)
                tile_type = tg.tile_type
                num_tiles = tg.number
                pattern_line_dest = tg.pattern_line_dest
                num_to_pattern_line = tg.num_to_pattern_line
                num_to_floor_line = tg.num_to_floor_line
                index = self.action_encoder.action_dict[(pattern_line_dest, num_to_pattern_line, num_to_floor_line)]
                # get this index from the q-values
                q_val = q_values[index]
                if q_val > max_q_val:
                    max_q_index = index
                    max_action = (action_type, id, tg)
        return max_action

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
        next_states = np.array([data[3] for data in minibatch])
        actions = [data[1] for data in minibatch]
        rewards = [data[2] for data in minibatch]
        dones = [data[4] for data in minibatch]
        next_q_values = self.target_model.predict(next_states)
        for i in range(len(minibatch)):
            target = rewards[i]
            if isinstance(actions[i], str):
                print("There should not be any strings here")
                continue
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])
            try:
                target_data[i][actions[i]] = target
            # need to get the index of this action 
            except:
                print("Tile type: ", num_tiles)
                # write the batch with error actions into a file to check
                with open("action_batch_error.txt", "w") as f:
                    f.write(str(actions))
                print("Action with error: ", (action_type, id, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line))
                print("Is action type end: ", actions[i][0] == 'ENDROUND')
                # continue
                exit()
        self.model.fit(input_data, target_data, epochs=1, verbose=0)
        self.update_target_model()
        # predict in batches

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
        for i in range(len(state.agents[self.id].floor)):
            reward += state.agents[self.id].floor[i] * state.agents[self.id].FLOOR_SCORES[i]
        if curr_score > prev_score:
            reward += (curr_score - prev_score) * 5
        # penalty for wasted tiles
        else:
            reward += 1
        if num_moves > MAX_MOVES:
            reward += -10
        # Reward for placing the tile in the correct position
        if is_timed_out:
            reward -= 10
        return reward
    def train(self):
        # with open(os.devnull, 'w') as devnull:
        #     with contextlib.redirect_stdout(devnull):
                self.epsilons = []
                self.agent1 = DQNAgent(0)
                self.agent2 = DQNAgent(1)
                wins_agent1 = 0
                wins_agent2 = 0
                start_time = time.time()
                self.agent1_win_rate_over_windows = []
                self.agent2_win_rate_over_windows = []
                for episode in range(NUM_EPISODES):
                        game = AdvancedGame(self.agent1, self.agent2)
                        agents_complete_reward = game._run()
                        agent_1_reward = agents_complete_reward[0]
                        agent_2_reward = agents_complete_reward[1]
                        print("Agent 1 reward: ", agent_1_reward)
                        print("Agent 2 reward: ", agent_2_reward)
                        if agent_1_reward > agent_2_reward:
                                wins_agent1 += 1
                        else:
                            wins_agent2 += 1
                        if episode % 5 == 0:
                            self.agent1_win_rate_over_windows.append(wins_agent1 / 5)
                            self.agent2_win_rate_over_windows.append(wins_agent2 / 5)
                            wins_agent1 = 0
                            wins_agent2 = 0
                        if self.agent1.epsilon > self.agent1.epsilon_min:
                            self.agent1.decay_epsilon()
                            self.epsilons.append(self.agent1.epsilon)
                        if self.agent2.epsilon > self.agent2.epsilon_min:
                            self.agent2.decay_epsilon()
                        if episode % SAVE_FREQUENCY == 0:
                            # save the policy for both agents
                            self.agent1.save("policymodel1.h5")
                            self.agent2.save("policymodel2.h5")
                if wins_agent1 >= wins_agent2:
                    # specify the path of the file
                    file_path = "policymodel2.h5"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print("File deleted successfully")
                else:
                    file_path = "policymodel1.h5"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print("File deleted successfully")     
                # End the timer and count the elapsed time
                elapsed_time = time.time() - start_time   
                print(elapsed_time)     
                self.agent1_win_rate_over_windows.pop(0)
                self.agent1_win_rate_over_windows.append(wins_agent1/5)

                self.agent2_win_rate_over_windows.pop(0)
                self.agent2_win_rate_over_windows.append(wins_agent2/5)


 
        
    
    
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
        # print the factory display at the start of each game
        # for i in range(len(state.factories)):
        #     for tile in utils.Tile:
        #         print("Factory: ", i, " Tile colour: ",tile, state.factories[i].tiles[tile])
        agent_turn = 0
        first_player_token  = 0
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
                action = func_timeout(self.time_limit, current_agent.act, args= (state, self.game_rule))
            except: 
                action = current_agent.act(state, self.game_rule)
                is_timed_out = True
            # successor state
            #self.game_rule.update(action)
            # print(len(state.agents))
            # print("Agent turn: ", agent_turn)
            # print("Agent trace: ", state.agents[agent_turn].agent_trace.actions)

            
            if isinstance(action, tuple):
                tg = action[2]
                pattern_line_dest = tg.pattern_line_dest
                num_to_pattern_line = tg.num_to_pattern_line
                num_to_floor_line = tg.num_to_floor_line
                try:
                    action_index = current_agent.action_encoder.action_dict[(pattern_line_dest, num_to_pattern_line, num_to_floor_line)]
                    # when the action is end round, the next state generated is scoring the
                    # rounds for each state agent and add the tiles to the used bag
                    next_state = self.game_rule.generateSuccessor(state,action,agent_turn)
                    reward = current_agent.reward_function(state, next_state, is_timed_out)
                    current_agent.episode_total_reward += reward
                    # features should be extracted from state and successor_state
                    # before putting them into memory
                    curr_feature = current_agent.get_features(state)
                    next_feature = current_agent.get_features(next_state)
                except:
                    # print("Action key pattern: ", (pattern_line_dest, num_to_pattern_line, num_to_floor_line))
                    legal_actions = self.game_rule.getLegalActions(state,current_agent.id)
                    for action_2 in legal_actions:
                        if len(action_2) == 3:
                            tile_type = action_2[2].tile_type
                            pattern_line_dest = action_2[2].pattern_line_dest
                            num_to_pattern_line = action_2[2].num_to_pattern_line
                            num_to_floor_line = action_2[2].num_to_floor_line
                            if tile_type > 0 and num_to_floor_line <= 7: 
                                action = action_2
                                action_index = current_agent.action_encoder.action_dict[(pattern_line_dest, num_to_pattern_line, num_to_floor_line)]
                                print("Tile type: ", action[2].tile_type)
                                next_state = self.game_rule.generateSuccessor(state,action,agent_turn)
                                reward = current_agent.reward_function(state, next_state, is_timed_out)
                                current_agent.episode_total_reward += reward
                                next_feature = current_agent.get_features(next_state)
                                print("Other action key pattern: ", (pattern_line_dest, num_to_pattern_line, num_to_floor_line))
                        else:
                            print("Other actions here: ", action)
                            action = action_2
                            next_state = self.game_rule.generateSuccessor(state,action,agent_turn)
                            reward = current_agent.reward_function(state, next_state, is_timed_out)
                            current_agent.episode_total_reward += reward
                            next_feature = current_agent.get_features(next_state)
                    
                if action_index:
                    current_agent.remember(curr_feature, action_index, reward, next_feature, done)
                else:
                    return 
            current_agent.replay()
            # update the state variable
            state = next_state
            if action == "ENDROUND":
                print("End of round")
                for player in state.agents:
                    player.agent_trace.StartRound()
                for fd in state.factories:
                    state.InitialiseFactory(fd)
                for tile in utils.Tile:
                    state.centre_pool.tiles[tile] = 0
                first_player_token = (first_player_token + 1) % NUM_PLAYERS
            # update the agent id
            if action == "ENDROUND":
                agent_turn = first_player_token
            else:
                agent_turn = (agent_turn + 1) % NUM_PLAYERS
            
            done = self.game_rule.gameEnds()
            # calScore
        print("End game")
        self.agent1.rewards.append(self.agent1.episode_total_reward)
        self.agent2.rewards.append(self.agent2.episode_total_reward)
        # renew the episode_total_reward
        self.agent1.episode_total_reward = 0
        self.agent2.episode_total_reward = 0
        agent_1_complete_reward = self.game_rule.calScore(state,0)
        agent_2_complete_reward = self.game_rule.calScore(state, 1)
        return [agent_1_complete_reward, agent_2_complete_reward]
# class ActualDQNAgent:

      
if __name__ == "__main__":  
    # testing for one episode
    # game_rule = AzulGameRule(NUM_PLAYERS)
    # agent1 = DQNAgent(0)
    # agent2 = DQNAgent(1)
    # game = AdvancedGame(agent1, agent2)
    # game._run()

    dqn_train = DQNAgent(3)
    dqn_train.train()
    # get agent1 rewards aross episodes
    agent1_rewards = dqn_train.agent1.rewards
    agent2_rewards = dqn_train.agent2.rewards
    ag1_wr = dqn_train.agent1_win_rate_over_windows
    ag2_wr = dqn_train.agent2_win_rate_over_windows

    # # Organize data into arrays
    episode_nums = [i for i in range(NUM_EPISODES)]
    episode_intervals = list(range(5,NUM_EPISODES + 1,5))
    figure, axis = plt.subplots(2,2)
    axis[0,0].plot(episode_nums, agent1_rewards)
    # Naming: Y vs X axis
    axis[0,0].set_title("Agent 1 rewards vs episodes nums")

    axis[0,1].plot(episode_nums, agent2_rewards)
    axis[0,1].set_title("Agent 2 rewards vs episode nums")

    axis[1,0].plot(episode_intervals,ag1_wr)
    axis[1,0].set_title("Agent 1 win rates vs episode interval")

    axis[1,1].plot(episode_intervals,ag2_wr)
    axis[1,1].set_title("Agent 2 win rates vs episode interval")

    # Combine all operations and display
    plt.show()





        