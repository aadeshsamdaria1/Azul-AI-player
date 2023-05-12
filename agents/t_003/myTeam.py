import math
import random
from Azul.azul_model import *
from copy import deepcopy

NUM_PLAYERS = 2

class myAgent():
    # Constructor for the class
    def __init__(self, _id):
        self.id = _id # Agent ID
        self.game_rule = AzulGameRule(NUM_PLAYERS) # Create an instance of the Azul game rules

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