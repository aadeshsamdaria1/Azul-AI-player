import Azul.azul_utils as utils
import numpy as np
import tensorflow as tf 
import random
from template import Agent as DummyAgent
import matplotlib.pyplot as plt
import random, copy, time
from   template     import GameState
from   func_timeout import func_timeout, FunctionTimedOut
from   template     import Agent as DummyAgent
import DQNTrain
from agents.generic.random import myAgent as RA
from Azul.azul_model import AzulGameRule
THINK_TIME = 0.9
NUM_PLAYERS = 2
NUM_COLOR = 5
GRID_SIZE = 5
BATCH_SIZE = 100
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.01
MIN_EPSILON = 0.01
MEMORY_CAPACITY = 3000
NUM_EPISODES = 30
SAVE_FREQUENCY = 10
FREEDOM = False  
WARMUP  = 15
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
        for tile in utils.Tile:
            for pattern_line_dest in range(GRID_SIZE):
                for num_to_pattern_line in range(0, GRID_SIZE + 1):
                            self.action_dict[(tile, pattern_line_dest, num_to_pattern_line)] = index
                            index += 1

        # Get the total number of possible actions
        # print("Size of action space: ", len(self.action_dict))
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
        model.add(tf.keras.layers.Dense(150, input_shape=(114,), activation='relu'))
        model.add(tf.keras.layers.Dense(150, activation='relu'))
        model.add(tf.keras.layers.Dense(150, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # the append method is used to add transition into memory , and when the memory is full,
    # the oldest memory is removed
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_CAPACITY:
            popped_memory = self.memory.pop(0)
            # print("Popped memory: ", popped_memory)
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
            if np.random.rand() <= self.epsilon:
                return random.choice(legal_actions)
            # TODO: need to change this
            # Extract relevant features for the Azul game state
            features = self.get_features(game_state)
            # Reshape the feature vector to be used as an input to the model
            features = features.reshape(1, -1)
            # Use the model to get Q-values for the current state
            print("Q max act")
            q_values = self.model.predict(features, verbose = 0)[0]
            # set q-values for unavailable actions to very low values
        
            # TODO: test this part 
            # flatten the legal actions
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
                    num_to_pattern_line = tg.num_to_pattern_line
                    num_to_floor_line = tg.num_to_floor_line
                    index = self.action_encoder.action_dict[(tile_type, pattern_line_dest, num_to_pattern_line)]
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
        target_data = self.model.predict(input_data, verbose = 0)
        next_states = np.array([data[3] for data in minibatch])
        actions = [data[1] for data in minibatch]
        rewards = [data[2] for data in minibatch]
        dones = [data[4] for data in minibatch]
        next_q_values = self.target_model.predict(next_states, verbose = 0)
        for i in range(len(minibatch)):
            target = rewards[i]
            if isinstance(actions[i], str):
                continue
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])
            try:
                tg = actions[i][2]
                tile_type = tg.tile_type
                pattern_line_dest = tg.pattern_line_dest
                num_to_pattern_line = tg.num_to_pattern_line
                # num_to_floor_line = tg.num_to_floor_line
                action_index = self.action_encoder.action_dict[(tile_type, pattern_line_dest,num_to_pattern_line)]
                target_data[i][action_index] = target
            # need to get the index of this action 
            except:
                # print("Action with error: ", (action_type, id, tile_type, num_tiles, pattern_line_dest, num_to_pattern_line, num_to_floor_line))
                # print("Is action type end: ", actions[i][0] == 'ENDROUND')
                continue
                # exit()
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

    def reward_function(self, state, action, successor_state):
        # Get the number of completed pattern lines in the previous state and current state
        # penalize against action that places in the floor line
        reward = 0
        if isinstance(action, tuple):
            tg = action[2]
            num_to_floor_line = tg.num_to_floor_line
            reward -= (num_to_floor_line) * 5
            # get the agent state in the previous state
            state_completed_lines = self.get_completed_patternlines(state)
            nextstate_completed_lines = self.get_completed_patternlines(successor_state)
            if nextstate_completed_lines > state_completed_lines:
                reward += (nextstate_completed_lines - state_completed_lines) * 5
            small_reward = self.compare_states_patternline(state, successor_state)
            reward += small_reward
            lt_r = self.long_term_reward(state, action, successor_state)
            reward += lt_r
        return reward
    def long_term_reward(self, state, action, successor_state):
        # get the pattern line number of the action
        if isinstance(action, tuple):
            tg = action[2]
            row = tg.pattern_line_dest
            tile_type = tg.tile_type
            wall_col = int(state.agents[self.id].grid_scheme[row][tile_type])
            lt_r = 0
            # the long term reward is only possible, if at least one of the target tile
            # is placed on the pattern line
            # for that row and column in the wall grid, check how many filled slots are there
            if tg.num_to_pattern_line > 0:
                for i in range(GRID_SIZE):
                    # same column (across row)
                    if state.agents[self.id].grid_state[i][wall_col]:
                        # get difference in row
                        # do not consider the current row
                        if i != row: 
                            diff_row = abs(row - i)
                            lt_r += (GRID_SIZE - diff_row) * 10
                for j in range(GRID_SIZE):
                    # same row (across column)
                    if state.agents[self.id].grid_state[row][j]:
                        # more reward
                        if j != wall_col:
                            diff_col = abs(wall_col - 1)
                            lt_r += (GRID_SIZE - diff_col) * 20
            return lt_r
        

                       

    def compare_states_patternline(self, state, next_state):
          # give a small reward if this state fills in more in the same pattern line
          state_lines_number = state.agents[self.id].lines_number
           # state_lines_tile = state.agents[self.id].lines_tile
          nextstate_lines_number = next_state.agents[self.id].lines_number
          # nextstate_lines_tile = next_state.agents[self.id].lines_tile
          small_reward = 0
          for i in range(GRID_SIZE):
              if nextstate_lines_number[i] > state_lines_number[i]:
                  small_reward += 1
          return small_reward
              

    def get_completed_patternlines(self, state):
        state_lines_number = state.agents[self.id].lines_number
        state_lines_tile = state.agents[self.id].lines_tile
        completed_lines = 0
        for i in range(GRID_SIZE):
                if state_lines_tile[i] != -1 and state_lines_number == i + 1:
                    completed_lines += 1
        return completed_lines
    
      
class Game:
    def __init__(self, GameRule,
                 agent_list, 
                 num_of_agent,
                 learner_id, seed=1, 
                 time_limit=1, 
                 warning_limit=3, 
                 displayer = None, 
                 agents_namelist = ["Alice","Bob"],
                 interactive=False):
        
        self.seed = seed
        random.seed(self.seed)
        self.seed_list = [random.randint(0,1e10) for _ in range(1000)]
        self.seed_idx = 0
        self.learner_id = learner_id

        # Make sure we are forming a valid game, and that agent
        # id's range from 0 to N-1, where N is the number of agents.
        # assert(len(agent_list) <= 4)
        # assert(len(agent_list) > 1)
        i = 0
        for plyr in agent_list:
            # print("Player id: ", plyr.id)
            # print("Agent list pos: ", i)
            assert(plyr.id == i)    
            i += 1

        self.game_rule = GameRule(num_of_agent)
        self.gamemaster = DummyAgent(num_of_agent) #GM/template agent used by some games (e.g. Azul, for signalling rounds).
        self.valid_action = self.game_rule.validAction
        self.agents = agent_list
        self.agents_namelist = agents_namelist
        self.time_limit = time_limit
        self.warning_limit = warning_limit
        self.warnings = [0]*len(agent_list)
        self.warning_positions = []
        self.displayer = displayer
        if self.displayer is not None:
            self.displayer.InitDisplayer(self)
        self.interactive = interactive

    def _EndGame(self,num_of_agent,history, isTimeOut = True, id = None):
        history.update({"seed":self.seed,
                        "num_of_agent":num_of_agent,
                        "agents_namelist":self.agents_namelist,
                        "warning_positions":self.warning_positions,
                        "warning_limit":self.warning_limit})
        history["scores"]= {i:0 for i in range(num_of_agent)}
        if isTimeOut:
            history["scores"][id] = -1
        else:
            for i in range(num_of_agent):
                history["scores"].update({i:self.game_rule.calScore(self.game_rule.current_game_state,i)})

        if self.displayer is not None:
            self.displayer.EndGame(self.game_rule.current_game_state,history["scores"])
        return history

    def Run(self):
        history = {"actions":[]}
        action_counter = 0
        while not self.game_rule.gameEnds():
            agent_index = self.game_rule.getCurrentAgentIndex()
            agent = self.agents[agent_index] if agent_index < len(self.agents) else self.gamemaster
            game_state = self.game_rule.current_game_state
            game_state.agent_to_move = agent_index
            actions = self.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            gs_copy = copy.deepcopy(game_state)
            
            # Delete all specified attributes in the agent state copies, if this isn't a perfect information game.
            if self.game_rule.private_information:
                delattr(gs_copy.deck, 'cards') # Upcoming cards cannot be observed.
                for i in range(len(gs_copy.agents)):
                    if gs_copy.agents[i].id != agent_index:
                        for attr in self.game_rule.private_information:
                            delattr(gs_copy.agents[i], attr)
            
            #Before updating the game, if this is the first move, allow the displayer an initial update.
            #This is used by some games to run simple pre-game animations.
            if action_counter==1 and self.displayer is not None:
                self.displayer._DisplayState(self.game_rule.current_game_state)
                        
            #If interactive mode, update displayer and obtain action via user input.
            if self.interactive and agent_index==1:
                self.displayer._DisplayState(self.game_rule.current_game_state)
                selected = self.displayer.user_input(actions_copy)
                
            else:
                #If freedom is given to agents, let them return any action in any time period, at the risk of breaking 
                #the simulation. This can be useful for debugging purposes.
                exception = None
                if FREEDOM:
                    selected = agent.SelectAction(actions_copy, gs_copy)
                else:
                    #"Gamemaster" agent has an agent index equal to the number of player agents in the game.
                    #If the gamemaster acts (e.g. to start or end a round in Azul), let it do so uninhibited.
                    #Else, allow player agent to select action within a time limit. 
                    #- If it times out, display TimeOutWarning. 
                    #- If it returns an illegal move, display IllegalWarning.
                    #  - Illegal move checked by self.validaction(), if implemented by the game being run.
                    #  - Else, look for move in actions list by equality according to Python.
                    #If this is the agent's first turn, allow warmup time.
                    try: 
                        # TODO : selected action
                        selected = func_timeout(WARMUP if action_counter < len(self.agents) else self.time_limit, 
                                                agent.SelectAction,args=(actions_copy, gs_copy))
                    except FunctionTimedOut:
                        selected = "timeout"
                    except Exception as e:
                        exception = e
                        
                    if agent_index != self.game_rule.num_of_agent:
                        if selected != "timeout":
                            if self.valid_action:
                                if not self.valid_action(selected, actions):
                                    print("Illegal warning")
                                    selected = "illegal"
                            elif not selected in actions:
                                selected = "illegal"
                            
                        if selected in ["timeout", "illegal"]:
                            self.warnings[agent_index] += 1
                            self.warning_positions.append((agent_index,action_counter))
                            if self.displayer is not None:
                                if selected=="timeout":
                                    self.displayer.TimeOutWarning(self,agent_index)
                                else:
                                    print("This action is illegal")
                                    self.displayer.IllegalWarning(self,agent_index,exception)                        
                            selected = random.choice(actions)

                
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1
            history["actions"].append({action_counter:{"agent_id":self.game_rule.current_agent_index,"action":selected}})
            action_counter += 1
            
            # TODO: state update is here
            self.game_rule.update(selected)
            random.seed(self.seed_list[self.seed_idx])
            if agent.id == self.learner_id:
                reward = agent.reward_function(gs_copy, selected, self.game_rule.current_game_state)
                curr_feature = agent.get_features(gs_copy)
                next_feature = agent.get_features(self.game_rule.current_game_state)
                if (isinstance(selected, tuple)):
                    print("Reward: ", reward)
                    agent.remember(curr_feature, selected, reward, next_feature, self.game_rule.gameEnds())
                agent.replay()

            self.seed_idx += 1

            if self.displayer is not None:
                self.displayer.ExcuteAction(agent_index,selected, self.game_rule.current_game_state)
            if (agent_index != self.game_rule.num_of_agent) and (self.warnings[agent_index] == self.warning_limit):
                history = self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=True,id=agent_index)
                return history
                
        # Score agent bonuses
        print("End of game reached")
        return self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=False)
            

if __name__ == "__main__":
    # NUM_PLAYERS = 2
    # agent_list = list()
    # agent_1 = RA(0)
    # LEARNER_ID = 1
    # agent_2 = DQNAgent(LEARNER_ID)
    # agent_list.append(agent_1)
    # agent_list.append(agent_2)
    # game = Game(AzulGameRule, agent_list, NUM_PLAYERS, LEARNER_ID)
    # game.Run()
    def train(): 
        NUM_PLAYERS = 2
        random_agent = RA(0)
        deepq_agent = DQNAgent(1)
        current_number_of_episodes = 1
        # make sure random seed is traceable
        random_seed = 1
        random.seed(random_seed)
        seed_list = [random.randint(0,1e10) for _ in range(1000)]
        seed_idx = 0
        for i in range(NUM_EPISODES // 2):
              agent_list = list()
              agent_list.append(random_agent)
              agent_list.append(deepq_agent)
              seed = seed_list[i]
              game = Game(AzulGameRule, agent_list, NUM_PLAYERS, deepq_agent.id, seed)
              output = game.Run()
              print("Episode num: ", current_number_of_episodes)
              end_of_game_score = output["scores"]
              if i % SAVE_FREQUENCY == 0:
                  deepq_agent.save("policymodel.h5")
              current_number_of_episodes += 1
        for j in range(NUM_EPISODES // 2, NUM_EPISODES):
              agent_list = list()
              deepq_agent.id = 0
              random_agent.id = 1
              agent_list.append(deepq_agent)
              agent_list.append(random_agent)
              # change swap id for both agents
              seed  = seed_list[j]
              game = Game(AzulGameRule, agent_list, NUM_PLAYERS, deepq_agent.id, seed)
              output = game.Run()
              end_of_game_score = output["scores"]
              if j % SAVE_FREQUENCY == 0:
                  deepq_agent.save("policymodel.h5")
              print("Episode num: ", current_number_of_episodes)
              current_number_of_episodes += 1
    train()
   

      






        