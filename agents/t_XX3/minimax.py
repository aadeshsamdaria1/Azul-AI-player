import math
import random
import traceback
from Azul.azul_model import *
from copy import deepcopy


NUM_PLAYERS = 2

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = AzulGameRule(NUM_PLAYERS)
        self.azul_state = AzulState(NUM_PLAYERS)
        self.agent_state = self.azul_state.AgentState

    def evaluate(self, state):
        agent1_score,_ = state.agents[self.id].ScoreRound()
        agent2_score,_ = state.agents[self.id*-1 + 1].ScoreRound()
        return (self.game_rule.calScore(state, self.id) - self.game_rule.calScore(state, self.id*-1 + 1)) + (agent1_score - agent2_score)
    
    def minimax(self, state, action, depth, alpha, beta, maximizing=True):

        # base case 
        if depth == 0:
            return self.evaluate(state)

        # maximizing player case 
        if maximizing:
            v = -math.inf
            for action in self.game_rule.getLegalActions(state, self.id):
                try:
                    successor = self.game_rule.generateSuccessor(state, action, self.id)
                    v_successor = self.minimax(successor, action, depth - 1, alpha, beta, False)
                    v = max(v, v_successor)
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                except:
                    pass
            return v
        # minimizing player case
        else:
            v = math.inf
            for action in self.game_rule.getLegalActions(state, self.id*-1 + 1):
                try:
                    successor = self.game_rule.generateSuccessor(state, action, self.id*-1 + 1)
                    v_successor = self.minimax(successor, action, depth - 1, alpha, beta, True)
                    v = min(v, v_successor)
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
                except:
                    pass
            return v
    
    def SelectAction(self, actions, rootstate):
        try:
            max_val = -math.inf
            best_action = None  
            for action in actions:
                try:
                    successor = self.game_rule.generateSuccessor(rootstate, action, self.id)
                    v = self.minimax(successor, action, depth=2, alpha=-math.inf, beta=math.inf, maximizing=True)
                    if v > max_val:
                        max_val = v
                        best_action = action
                except:
                    pass
            if best_action is None:
                return random.choice(actions)
            else:
                return best_action
        except:
            random.choice(actions)

# command to run the program
#  python general_game_runner.py -g Azul -a agents.generic.random,agents.t_XX3.minimax -p