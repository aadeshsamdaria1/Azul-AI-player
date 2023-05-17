import random
import math
import time, random
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState as State
from copy import deepcopy
from collections import deque
from typing import List
from template import Agent

THINKTIME   = 0.9
NUM_PLAYERS = 2
EXPLORATION_FACTOR = 0.3
class MinimaxAgent():
    # Constructor for the class
    def __init__(self, _id):
        self.id = _id # Agent ID
        self.game_rule = GameRule(NUM_PLAYERS) # Create an instance of the Azul game rules

    # Method to evaluate a given state and return a score
    def evaluate(self, state):
        # Get the scores of each player from the given state
        agent1_score,_ = state.agents[self.id].ScoreRound()
        agent2_score,_ = state.agents[self.id*-1 + 1].ScoreRound()
        # Calculate and return the difference between the scores of the two players and the difference between the scores of the current agent and the other agent
        diff_score = (self.game_rule.calScore(state, self.id) - self.game_rule.calScore(state, self.id*-1 + 1)) + (agent1_score - agent2_score)
        return diff_score
    
    def minimax(self, state, action, depth, alpha, beta, maximizing=True):

        # Base case for recursion - if depth reaches 0, return the evaluation of the current state
        if depth == 0:
            return self.evaluate(deepcopy(state))

        # Maximizing player case 
        if maximizing:
            v = -math.inf
            # For each legal action for the current agent, generate a successor state and call minimax recursively with a depth reduced by 1, and maximizing set to False (since the next turn will be the other player's)
            for action in self.game_rule.getLegalActions(state, self.id):
                try:
                    successor = self.game_rule.generateSuccessor(state, action, self.id)
                    v_successor = self.minimax(successor, action, depth - 1, alpha, beta, False)    
                    v = max(v, v_successor) # Update v to be the maximum of v and the evaluation of the successor state
                    alpha = max(alpha, v) # Update alpha to be the maximum of alpha and v
                    # If beta is less than or equal to alpha, break the loop since the minimum value that the other player can force will already be less than or equal to v
                    if beta <= alpha:
                        break
                except:
                    pass
            return v
        # Minimizing player case
        else:
            v = math.inf
            # For each legal action for the other agent, generate a successor state and call minimax recursively with a depth reduced by 1, and maximizing set to True (since the next turn will be the current player's)
            for action in self.game_rule.getLegalActions(state, self.id*-1 + 1):
                try:
                    successor = self.game_rule.generateSuccessor(state, action, self.id*-1 + 1)
                    v_successor = self.minimax(successor, action, depth - 1, alpha, beta, True)
                    v = min(v, v_successor) # Update v to be the minimum of v and the evaluation of the successor state
                    beta = min(beta, v) # Update beta to be the minimum of beta and v
                    # If beta is less than or equal to alpha, break the loop since the minimum value that the other player can force will already be less than or equal to v
                    if beta <= alpha:
                        break
                except:
                    pass
            return v
    
    def SelectAction(self, actions, rootstate):
        try:
            max_val = -math.inf
            best_action = None
            for action in actions: # Loop over each action and evaluate the successor state using minimax algorithm
                try:
                    successor = self.game_rule.generateSuccessor(deepcopy(rootstate), action, self.id)
                    # Call minimax with depth=2, alpha=-infinity, beta=infinity, and maximizing=True, since the current player is trying to maximize their score
                    v = self.minimax(successor, action, depth=2, alpha=-math.inf, beta=math.inf, maximizing=True)
                    # Update max_val and best_action if a higher value is found
                    if v > max_val:
                        max_val = v
                        best_action = action
                except:
                    pass
            # If no best action is found, choose a random action from the list of available actions otherwise return the best action
            if best_action is None:
                return random.choice(actions)
            else:
                return best_action
        except:
            # In case of an error, choose a random action from the list of available actions
            return random.choice(actions)
        
class Node:
    def __init__(self, state: State, action, player_id, parent=None):
        self.state: State = state
        self.parent: Node = parent
        self.children: List(Node) = []
        self.visits = 0
        self.action = action
        self.q_value = 0
        self.sum_of_squares = 0
        self.player_id = player_id

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.q_value += reward
        self.sum_of_squares += reward**2

    def fully_expanded(self):
        return all(child.visits > 0 for child in self.children)
    
    def best_q_value_child(self):
        return max(self.children, key=lambda child_node: child_node.q_value, default=None)
    
    def best_ucb_child(self):
        return max(self.children, key=lambda child_node: self.ucb_score(child_node), default=None)
    
    def best_ucb1_tuned_child(self):
        return max(self.children, key=lambda child_node: self.ucb1_tuned_score(child_node), default=None)

    def ucb_score(self, child_node):
        if child_node.visits == 0:
            return float('-inf')
        exploitation_term = child_node.q_value / child_node.visits
        exploration_term = math.sqrt(math.log(self.visits) / child_node.visits)
        return exploitation_term + EXPLORATION_FACTOR * exploration_term
    
    def ucb1_tuned_score(self, child_node):
        if child_node.visits == 0:
            return float('inf')

        exploitation_term = child_node.q_value / child_node.visits

        variance = (child_node.sum_of_squares / child_node.visits) - (child_node.q_value / child_node.visits)**2
        variance_term = min(1/4, variance + math.sqrt((2*math.log(self.visits)) / child_node.visits))

        exploration_term = math.sqrt((math.log(self.visits) / child_node.visits) * variance_term)

        return exploitation_term + EXPLORATION_FACTOR * exploration_term

class MCTS:
    def __init__(self, _id, root: Node):
        self.id = _id
        self.exploration_constant = EXPLORATION_FACTOR
        self.minimax_agent = MinimaxAgent(self.id)
        self.root = root

    def Search(self):

        begin_time = time.time()

        while time.time() - begin_time < THINKTIME:

            ### SELECTION ###
            expand_node = self.Selection(self.root)


            ### EXPANSION ###
            subset_size = 10
            actions = []
            if expand_node.state.TilesRemaining():
                children = self.Expansion(expand_node) 
                if len(children) > subset_size:
                    subset = random.sample(children, subset_size)
                else:
                     subset = children    
                for child in subset:
                    actions.append(child.action)
                action = self.minimax_agent.SelectAction(actions, expand_node.state)
                for child in subset:
                    if child.action == action:
                        expand_node = child


            ### SIMULATION ###
            rewards, move_counts = self.Simulation(expand_node)

            ### BACKPROPAGATION ###
            self.Backpropagation(expand_node, rewards)


        best_child = self.root.best_q_value_child()
        return best_child.action


    def Selection(self, root: Node):
        node = root
        while len(node.children): ### not node.fully_expanded():
            node = node.best_ucb_child()
        return node

    def Expansion(self, node: Node):
        opponent_id = 1 - node.player_id
        actions = self.minimax_agent.game_rule.getLegalActions(node.state, node.player_id)
        for action in actions:
            game_state = deepcopy(node.state)
            successor = self.minimax_agent.game_rule.generateSuccessor(game_state, action, node.player_id)
            node.children.append(Node(successor, action, opponent_id, node))


        return node.children
    
    def Simulation(self, node: Node):
        move_count = 0
        player_id = node.player_id
        game_state = node.state
        while game_state.TilesRemaining():
            actions = self.minimax_agent.game_rule.getLegalActions(game_state, player_id)
            action = self.minimax_agent.SelectAction(actions, game_state)
            game_state = self.minimax_agent.game_rule.generateSuccessor(game_state, action, player_id)
            player_id = 1 - player_id
            move_count += 1
        return self.evaluate_score(game_state, player_id), move_count

    def choose_best_action(self, game_state: State, actions, player_id):
        best_action = None
        best_score = float('-inf')
        for action in actions:
            successor = self.minimax_agent.game_rule.generateSuccessor(deepcopy(game_state), action, player_id)
            score = self.minimax_agent.game_rule.calScore(successor, player_id)
            if score > best_score:
                best_score = score
                best_action = action
        if not best_action:
            return random.choice(actions)
        return best_action


    def Backpropagation(self, node: Node, reward):
        parent = node
        while parent:
            parent.update(reward)
            parent = parent.parent

    def evaluate_score(self, game_state: State, agent_id):
        return self.minimax_agent.game_rule.calScore(game_state, agent_id) 
    
class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)

    def SelectAction(self, actions, rootstate):
        root_node = Node(rootstate, None, self.id, None)
        monte_carlo = MCTS(self.id, root_node)
        best_move = monte_carlo.Search()
        if not best_move:
            return random.choice(actions)
        return best_move