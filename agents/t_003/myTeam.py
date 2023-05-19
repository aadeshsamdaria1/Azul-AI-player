import math
import random
from Azul.azul_model import *
from copy import deepcopy
import collections
import time
THINK_TIME = 0.9
NUM_PLAYERS = 2
GRID_SIZE = 5


class myAgent():
    # Constructor for the class
    def __init__(self, _id):
        print("--Minimax agent loaded--")
        self.id = _id  # Agent ID
        # Create an instance of the Azul game rules
        self.game_rule = AzulGameRule(NUM_PLAYERS)
        self.transition_table = {}

    def get_remainder(self, player_state, line_idx, num_to_line):
        remainder = 0

        if player_state.lines_tile[line_idx] != -1:
            num_exist = player_state.lines_number[line_idx]
            remainder = line_idx + 1 - (num_exist + num_to_line)

        else:
            assert player_state.lines_number[line_idx] == 0
            remainder = line_idx + 1 - num_to_line

        return remainder

    def get_bag(self, game_state):
        # get unused tiles
        bag_dic = collections.defaultdict(int)

        for tile in game_state.bag:
            bag_dic[tile] += 1

        for factory in game_state.factories:
            for tile in range(5):
                bag_dic[tile] += factory.tiles[tile]

        for tile in range(5):
            bag_dic[tile] += game_state.centre_pool.tiles[tile]

        return bag_dic

    def get_bonus(self, bonus_unit, game_state, player_state, round_num, flag):
        bag_dic = self.get_bag(game_state)
        estimated_bonus = 0
        # go through each row and column of the wall grid
        for i in range(5):
            wall_grid_filled = 0
            vacant_unit = collections.defaultdict(int)
            for j in range(5):
                if flag == 'row':
                    # row slower (outer)
                    row_index, column_index = i, j
                    # grid scheme[wallgridrow][tile_color] = wallgridcolumn
                    # [[0. 1. 2. 3. 4.]
                    # [1. 2. 3. 4. 0.]
                    # [2. 3. 4. 0. 1.]
                    # [3. 4. 0. 1. 2.]
                    # [4. 0. 1. 2. 3.]]
                    tile_type = numpy.where(
                        player_state.grid_scheme[i] == j)[0]
                elif flag == 'col':
                    row_index, column_index = j, i
                    # col slower (outer)
                    tile_type = numpy.where(
                        player_state.grid_scheme[j] == i)[0]
                else:  # set
                    # completed set is when 5 tiles of the same colour is filled on the wall grid
                    row_index, column_index = j, int(
                        player_state.grid_scheme[j][i])
                    # check for tile (slower)
                    tile_type = i

                if player_state.grid_state[row_index][column_index] == 1:
                    # if a particular wall grid is filled in
                    wall_grid_filled += 1
                elif player_state.grid_state[row_index][column_index] == 0:
                    # if a wall grid is not filled in
                    # if the row in the pattern line currently has that tile type
                    # check how many are filled
                    left = player_state.lines_number[row_index] if player_state.lines_tile[row_index] == tile_type else 0
                    # record how many to fill
                    vacant_unit[int(tile_type)] += row_index + 1 - left

            feasible = all(
                tile in bag_dic and vacant_unit[tile] <= bag_dic[tile] for tile in vacant_unit)

            if wall_grid_filled >= round_num and feasible:
                # bonus unit specifies whether bonus given for a row or column
                # number of filled grids / 5 * bonus
                # estimated bonus is only greater than 0 when there is enough tiles in the bag to fill
                estimated_bonus += wall_grid_filled * bonus_unit/5
        # prioritize future rounds
        estimated_bonus = estimated_bonus*0.9**(4-round_num)
        return estimated_bonus
    # long term reward

    def get_future_bonus(self, game_state, player_state, round_num):
        row_score = self.get_bonus(
            2, game_state, player_state, round_num, 'row')
        column_score = self.get_bonus(
            7, game_state, player_state, round_num, 'col')
        set_score = self.get_bonus(
            10, game_state, player_state, round_num, 'set')
        return row_score + column_score + set_score

    # Method to evaluate a given state and return a score
    def evaluate(self, state):
        # Get the scores of each player from the given state
        round_num = (4 - len(state.bag) // 20)
        plr_state = state.agents[self.id]
        enemy_state = state.agents[self.id*-1 + 1]
        agent1_score, _ = plr_state.ScoreRound()
        agent2_score, _ = enemy_state.ScoreRound()
        player_bonus = self.get_future_bonus(state, plr_state, round_num)
        opponent_bonus = self.get_future_bonus(state, enemy_state, round_num)
        player_escore = plr_state.EndOfGameScore()
        opponent_escore = enemy_state.EndOfGameScore()
        diff_score = (agent1_score - agent2_score) + player_bonus - \
            opponent_bonus + player_escore - opponent_escore
        return diff_score

    def minimax(self, game_state, depth, alpha, beta, maximizing=True):
        action_threshold = 6
        # base case
        if depth == 0 or self.game_rule.gameEnds() or game_state.TilesRemaining() == 0:
            V = self.evaluate(game_state)
            return (None, V)

        # maximizing player case
        if maximizing:
            value = -math.inf
            moves = self.game_rule.getLegalActions(game_state, self.id)
            best_move = moves[0]
            move_dict = {}
            plr_state = game_state.agents[self.id]

            # filtering some unplausible actions
            if len(moves) > 6:
                for move in moves:
                    if move[2].num_to_floor_line > 1 or move[2].pattern_line_dest == -1:
                        continue

                    tile_type = move[2].tile_type
                    p_dest = move[2].pattern_line_dest
                    num_to_line = move[2].num_to_pattern_line
                    floor = move[2].num_to_floor_line
                    remainder = self.get_remainder(
                        plr_state, p_dest, num_to_line)
                    bag_dict = self.get_bag(game_state)
                    unnecessary = remainder + floor + bag_dict[tile_type]
                    numoffset = move[2].num_to_pattern_line - \
                        move[2].num_to_floor_line

                    if (tile_type, p_dest) not in move_dict or numoffset > move_dict[(tile_type, p_dest)][0]:
                        move_dict[(tile_type, p_dest)] = (
                            numoffset, unnecessary, move)

                moves = [v[2] for k, v in sorted(
                    move_dict.items(), key=lambda item: item[1][1])][:action_threshold]

            for move in moves:
                game_state_copy = copy.deepcopy(game_state)
                succ_state = self.game_rule.generateSuccessor(
                    game_state_copy, move, self.id)
                new_value = self.minimax(
                    succ_state, depth-1, alpha, beta,  False)[1]
                if new_value > value:
                    value = new_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return best_move, value

        # minimizing player case
        else:
            value = math.inf
            moves = self.game_rule.getLegalActions(game_state, self.id*-1 + 1)
            best_move = moves[0]
            enemy_state = game_state.agents[self.id*-1 + 1]
            move_dict = {}

            # filtering some unplausible actions
            if len(moves) > 6:
                for move in moves:
                    if move[2].num_to_floor_line > 1 or move[2].pattern_line_dest == -1:
                        continue

                    tile_type = move[2].tile_type
                    p_dest = move[2].pattern_line_dest
                    num_to_line = move[2].num_to_pattern_line
                    floor = move[2].num_to_floor_line
                    remainder = self.get_remainder(
                        enemy_state, p_dest, num_to_line)
                    bag_dict = self.get_bag(game_state)
                    unnecessary = remainder + floor + bag_dict[tile_type]
                    numoffset = move[2].num_to_pattern_line - \
                        move[2].num_to_floor_line

                    if (tile_type, p_dest) not in move_dict or numoffset > move_dict[(tile_type, p_dest)][0]:
                        move_dict[(tile_type, p_dest)] = (
                            numoffset, unnecessary, move)

                moves = [v[2] for k, v in sorted(
                    move_dict.items(), key=lambda item: item[1][1])][:action_threshold]

            for move in moves:
                game_state_copy = copy.deepcopy(game_state)

                succ_state = self.game_rule.generateSuccessor(
                    game_state_copy, move, self.id*-1 + 1)
                new_value = self.minimax(
                    succ_state, depth-1, alpha, beta, True)[1]
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break

            return best_move, value

    def SelectAction(self, moves, game_state):
        depth = 4

        if len(moves) > 55:
            depth = 3
        elif len(moves) > 10:
            depth = 4
        else:
            depth = 5
        move = self.minimax(game_state, depth, -math.inf, math.inf, True)[0]

        return move
