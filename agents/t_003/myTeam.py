import random
import math
import time, random
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState as State
from copy import deepcopy
from collections import deque
from typing import List
from Azul.azul_utils import Tile

from template import Agent

THINKTIME   = 0.9
NUM_PLAYERS = 2
EXPLORATION_FACTOR = 0.5

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
    def __init__(self, root: Node):
        self.exploration_constant = EXPLORATION_FACTOR
        self.game_rule = GameRule(NUM_PLAYERS)
        self.root = root
        self.transposition_table = {}

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
        while len(node.children):
            # if self.hash(node.state) in self.transposition_table:
            #     return self.transposition_table[self.hash(node.state)]
            # else:
            node = node.best_ucb_child()
        return node

    def Expansion(self, node: Node):
        opponent_id = 1 - node.player_id
        actions = self.game_rule.getLegalActions(node.state, node.player_id)
        actions = self.simplify_action_space(actions)
        for action in actions:
            game_state = deepcopy(node.state)
            successor = self.game_rule.generateSuccessor(game_state, action, node.player_id)
            node.children.append(Node(successor, action, opponent_id, node))
            # self.transposition_table[self.hash(successor)] = node.children[-1]


        return node.children
    
    def Simulation(self, node: Node):
        move_count = 0
        player_id = node.player_id
        game_state = node.state
        while game_state.TilesRemaining():
            actions = self.game_rule.getLegalActions(game_state, player_id)
            action = self.choose_best_action(game_state, actions, player_id)
            game_state = self.game_rule.generateSuccessor(game_state, action, player_id)
            player_id = 1 - player_id
            move_count += 1
        return self.evaluate_score(game_state, player_id), move_count

    def choose_best_action(self, game_state: State, actions, player_id):
        best_action = None
        best_score = float('-inf')
        for action in actions:
            successor = self.game_rule.generateSuccessor(deepcopy(game_state), action, player_id)
            score = self.game_rule.calScore(successor, player_id)
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
        goal = self.calculate_bonus(self, game_state: State, agent_id)
        penalty = self.calculate_penalty(game_state, agent_id)

        return self.game_rule.calScore(game_state, agent_id) - penalty
    
    def hash(self, state):
        # Placeholder hash function that will convert GameState into hash for the Transposition Table
        return hash(str(state))
    
    def simplify_action_space(self, moves):

        def too_many_to_floor_line(move, limit):
            return move[2].num_to_floor_line >= limit

        def picking_one_to_higher_lines(move):
            return move[2].pattern_line_dest > 2 and move[2].num_to_pattern_line in [0, 1, 2]

        def no_pattern_line(move):
            return move[2].num_to_pattern_line == 0

        simplified_moves = []
        for move in moves:
            if len(moves) > 50 and (picking_one_to_higher_lines(move) or too_many_to_floor_line(move, 1)):
                continue
            if len(moves) > 30 and too_many_to_floor_line(move, 2):
                continue
            if len(moves) > 10 and no_pattern_line(move):
                continue
            if len(moves) > 5 and too_many_to_floor_line(move, 3):
                continue

            simplified_moves.append(move)
                
        return simplified_moves
    
    def calculate_penalty(self, game_state: State, agent_id):
        penalty_mapping = {
            "incomplete_line": 1,
            "near_completion": 0.5,
            "single_tile": 0.5,
            "last_two_rows": 0.25,
            "last_row": 0.5,
            "same_color": 0.5  
        }

        agent_state = game_state.agents[agent_id]
        penalty = 0

        color_counts = {color: 0 for color in Tile} 

        # Punish unfinished pattern line
        for i in range(0, 5):
            line_color = agent_state.lines_tile[i]
            line_count = agent_state.lines_number[i]
            is_incomplete = line_count > 0 and line_count < i+1
            is_near_completion = i > 0 and line_count == i
            is_single_tile = i > 0 and line_count == 1

            if line_color != -1:  # -1 means no tile in the line
                color_counts[line_color] += 1

            if is_incomplete:
                penalty += penalty_mapping["incomplete_line"]
            if is_near_completion:
                penalty += penalty_mapping["near_completion"]
            if is_single_tile:
                penalty += penalty_mapping["single_tile"]

                # Additional penalty if 1 tile added to last 2 rows
                if i > 3:
                    penalty += penalty_mapping["last_two_rows"]
                if i > 4:
                    penalty += penalty_mapping["last_row"]

        # Add penalties for same color in multiple rows
        for color, count in color_counts.items():
            if count > 1:
                penalty += (count - 1) * penalty_mapping["same_color"]

        return penalty

    def calculate_bonus(self, game_state: State, agent_id):
        pass



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