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
EXPLORATION_FACTOR = 1

class Node:
    def __init__(self, state: State, action, player_id, parent=None):
        self.state: State = state
        self.parent: Node = parent
        self.children: List(Node) = []
        self.visits = 0
        self.action = action
        self.q_value = 0
        self.player_id = player_id

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.q_value += reward

    def fully_expanded(self):
        return all(child.visits > 0 for child in self.children)
    
    def best_q_value_child(self):
        return max(self.children, key=lambda child_node: child_node.q_value, default=None)
    
    def best_ucb_child(self):
        return max(self.children, key=lambda child_node: self.ucb_score(child_node), default=None)

    def ucb_score(self, child_node):
        if child_node.visits == 0:
            return float('-inf')
        exploitation_term = child_node.q_value / child_node.visits
        exploration_term = math.sqrt(math.log(self.visits) / child_node.visits)
        return exploitation_term + EXPLORATION_FACTOR * exploration_term

class MCTS:
    def __init__(self, root: Node):
        self.exploration_constant = EXPLORATION_FACTOR
        self.game_rule = GameRule(NUM_PLAYERS)
        self.root = root

    def Search(self):

        begin_time = time.time()

        while time.time() - begin_time < THINKTIME:

            ### SELECTION ###
            expand_node = self.Selection(self.root)


            ### EXPANSION ###
            if expand_node.state.TilesRemaining():
                children = self.Expansion(expand_node)
                expand_node = random.choice(children)


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
        actions = self.game_rule.getLegalActions(node.state, node.player_id)
        for action in actions:
            game_state = deepcopy(node.state)
            successor = self.game_rule.generateSuccessor(game_state, action, node.player_id)
            node.children.append(Node(successor, action, opponent_id, node))


        return node.children

    def Simulation(self, node: Node):
        move_count = 0
        player_id = node.player_id
        game_state = node.state
        while game_state.TilesRemaining():
            actions = self.game_rule.getLegalActions(game_state, player_id)
            action = random.choice(actions)
            game_state = self.game_rule.generateSuccessor(game_state, action, player_id)
            player_id = 1-player_id
            move_count += 1
        
        return self.evaluate_score(game_state, player_id), move_count

    def Backpropagation(self, node: Node, reward):
        parent = node
        while parent:
            parent.update(reward)
            parent = parent.parent

    def evaluate_score(self, game_state: State, agent_id):
        return self.game_rule.calScore(game_state, agent_id) 
    
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)

    def SelectAction(self, actions, rootstate):
        root_node = Node(rootstate, None, self.id, None)
        monte_carlo = MCTS(root_node)
        best_move = monte_carlo.Search()
        if not best_move:
            return random.choice(actions)
        return best_move