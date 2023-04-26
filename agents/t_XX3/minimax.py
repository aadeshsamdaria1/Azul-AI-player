import math
import random
from Azul.azul_model import AzulGameRule as GameRule

NUM_PLAYERS = 2

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)

    def evaluate(self, state):
        score = state.agents[self.id].score
        return score

    def min_value(self, state, depth, alpha, beta):
        if self.game_rule.isTerminal(state) or depth == 0:
            return self.evaluate(state)
        
        v = math.inf
        for action in self.game_rule.getLegalActions(state, 1 - self.id):
            successor = self.game_rule.generateSuccessor(state, action, 1 - self.id)
            v = min(v, self.max_value(successor, depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def max_value(self, state, depth, alpha, beta):
        if self.game_rule.isTerminal(state) or depth == 0:
            return self.evaluate(state)
        
        v = -math.inf
        for action in self.game_rule.getLegalActions(state, self.id):
            successor = self.game_rule.generateSuccessor(state, action, self.id)
            v = max(v, self.min_value(successor, depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def SelectAction(self, actions, rootstate):
        max_val = -math.inf
        best_action = None
        
        for action in self.game_rule.getLegalActions(rootstate, self.id):
            successor = self.game_rule.generateSuccessor(rootstate, action, self.id)
            v = self.min_value(successor, depth=2, alpha=-math.inf, beta=math.inf)
            if v > max_val:
                max_val = v
                best_action = action
                
        if best_action is None:
            return random.choice(actions)
        else:
            return best_action
