# Written by Michelle Blom, 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from utils import *
import math
import copy
import collections
from Azul.azul_model import *
NUM_PLAYERS = 2


class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = AzulGameRule(NUM_PLAYERS)

    def get_remainder(self, ps, line_idx, num_to_line):
        remainder = 0

        if ps.lines_tile[line_idx] != -1:
            num_exist = ps.lines_number[line_idx]
            remainder = line_idx + 1 - (num_exist + num_to_line)

        else:
            assert ps.lines_number[line_idx] == 0
            remainder = line_idx + 1 - num_to_line

        return remainder

    def get_estimated_bonus(self, game_state, player_state, round_num):
        row_score = self.get_bonus(
            2, game_state, player_state, round_num, 'row')
        column_score = self.get_bonus(
            7, game_state, player_state, round_num, 'col')
        set_score = self.get_bonus(
            10, game_state, player_state, round_num, 'set')
        return row_score + column_score + set_score

    def get_bag(self, game_state):
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

        for i in range(5):
            each_unit = 0
            vacant_unit = collections.defaultdict(int)
            for j in range(5):
                if flag == 'row':
                    row_index = i
                    column_index = j
                    tile_type = numpy.where(
                        player_state.grid_scheme[i] == j)[0]
                elif flag == 'col':
                    row_index = j
                    column_index = i
                    tile_type = numpy.where(
                        player_state.grid_scheme[j] == i)[0]
                else:  # set
                    row_index = j
                    column_index = int(player_state.grid_scheme[j][i])
                    tile_type = i

                if player_state.grid_state[row_index][column_index] == 1:
                    each_unit += 1
                elif player_state.grid_state[row_index][column_index] == 0:
                    left = 0
                    if player_state.lines_tile[row_index] == tile_type:
                        left = player_state.lines_number[row_index]
                    vacant_unit[int(tile_type)] += row_index + 1 - left

            feasible = True

            for tile in vacant_unit.keys():
                if not tile in bag_dic.keys() or vacant_unit[tile] > bag_dic[tile]:
                    feasible = False

            if each_unit >= round_num and feasible:
                estimated_bonus += each_unit * bonus_unit/5
        # prioritize future pos
        estimated_bonus = estimated_bonus*0.9**(4-round_num)
        return estimated_bonus

    def get_grid_tileCnt(self, game_state):
        player_grid_cnt = 0
        enemy_grid_cnt = 0

        plr_state = game_state.agents[self.id]
        enemy_state = game_state.agents[self.id*-1 + 1]

        player_tile_exist = 0
        enemy_tile_exist = 0

        for i in range(plr_state.GRID_SIZE):
            tt = plr_state.lines_tile[i]
            player_tile_exist += plr_state.lines_number[i]

            for j in range(5):
                if plr_state.grid_state[i][j] != 0:
                    player_grid_cnt += 1

        for i in range(enemy_state.GRID_SIZE):
            tt = enemy_state.lines_tile[i]
            enemy_tile_exist += enemy_state.lines_number[i]

            for j in range(5):
                if enemy_state.grid_state[i][j] != 0:
                    enemy_grid_cnt += 1

        return (player_grid_cnt - enemy_grid_cnt), (player_tile_exist - enemy_tile_exist)

    def evaluate(self, game_state):
        """
        Simple evaluation of game state using player scores 
        """

        round_num = (4 - len(game_state.bag) // 20)
        game_state_eval = copy.deepcopy(game_state)

        enemy_id = self.id*-1 + 1
        plr_state = game_state_eval.agents[self.id]
        enemy_state = game_state_eval.agents[enemy_id]

        game_state_eval.ExecuteEndOfRound()
        grid_tile_cnt_diff, tile_exist_diff = self.get_grid_tileCnt(
            game_state_eval)

        #player_score_change = plr_state.player_trace.round_scores[-1]
        #enemy_score_change = enemy_state.player_trace.round_scores[-1]
        player_score = game_state_eval.agents[self.id].score
        enemy_score = game_state_eval.agents[enemy_id].score

        player_bonus = self.get_estimated_bonus(
            game_state_eval, plr_state, round_num)
        opponent_bonus = self.get_estimated_bonus(
            game_state_eval, enemy_state, round_num)

        return (player_score - enemy_score) + player_bonus - opponent_bonus + grid_tile_cnt_diff - tile_exist_diff

    def get_action_threshold(self, moves):
        action_threshold = 7

        return action_threshold

    def minimax(self, game_state, depth, alpha, beta, maximizing=True):

        # check terminal state
        is_terminal = False
        for plr_state in game_state.agents:
            if plr_state.GetCompletedRows() > 0:
                is_terminal = True
                break

        # reached end of round but not end of game
        is_round_end = False
        if not is_terminal and game_state.TilesRemaining() == 0:
            is_round_end = True

        # base case
        if depth == 0 or is_terminal or is_round_end:
            V = self.evaluate(game_state)
            return (None, V)

        # maximizing player case
        if maximizing:
            value = -math.inf
            moves = self.game_rule.getLegalActions(game_state, self.id)
            best_move = moves[0]
            move_dict = {}
            plr_state = game_state.agents[self.id]
            action_threshold = self.get_action_threshold(moves)

            # filtering some unplausible actions
            if len(moves) > 7:
                for move in moves:
                    if move[2].num_to_floor_line > 1 or move[2].pattern_line_dest == -1:
                        continue

                    tile_type = move[2].tile_type
                    p_dest = move[2].pattern_line_dest
                    num_to_line = move[2].num_to_pattern_line
                    floor = move[2].num_to_floor_line
                    remainder = self.get_remainder(
                        plr_state, p_dest, num_to_line)

                    unnecessary = remainder + floor
                    numoffset = move[2].num_to_pattern_line - \
                        move[2].num_to_floor_line

                    if (tile_type, p_dest) not in move_dict or numoffset > move_dict[(tile_type, p_dest)][0]:
                        move_dict[(tile_type, p_dest)] = (
                            numoffset, unnecessary, move)

                moves = [v[2] for k, v in sorted(
                    move_dict.items(), key=lambda item: item[1][1])][:action_threshold]

            for move in moves:
                game_state_copy = copy.deepcopy(game_state)
                self.game_rule.generateSuccessor(
                    game_state_copy, move, self.id)
                new_value = self.minimax(
                    game_state_copy, depth-1, alpha, beta,  False)[1]
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
            action_threshold = self.get_action_threshold(moves)

            # filtering some unplausible actions
            if len(moves) > 7:
                for move in moves:
                    if move[2].num_to_floor_line > 1 or move[2].pattern_line_dest == -1:
                        continue

                    tile_type = move[2].tile_type
                    p_dest = move[2].pattern_line_dest
                    num_to_line = move[2].num_to_pattern_line
                    floor = move[2].num_to_floor_line
                    remainder = self.get_remainder(
                        enemy_state, p_dest, num_to_line)

                    unnecessary = remainder + floor
                    numoffset = move[2].num_to_pattern_line - \
                        move[2].num_to_floor_line

                    if (tile_type, p_dest) not in move_dict or numoffset > move_dict[(tile_type, p_dest)][0]:
                        move_dict[(tile_type, p_dest)] = (
                            numoffset, unnecessary, move)

                moves = [v[2] for k, v in sorted(
                    move_dict.items(), key=lambda item: item[1][1])][:action_threshold]

            for move in moves:
                game_state_copy = copy.deepcopy(game_state)

                self.game_rule.generateSuccessor(
                    game_state_copy, move, self.id*-1 + 1)
                new_value = self.minimax(
                    game_state_copy, depth-1, alpha, beta, True)[1]
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break

            return best_move, value

    def SelectAction(self, moves, game_state):

        plr_state = game_state.agents[self.id]

        depth = 4

        if len(moves) > 55:
            depth = 3
        elif len(moves) > 10:
            depth = 4
        else:
            depth = 5

        move, minimax_score = self.minimax(
            game_state, depth, -math.inf, math.inf, True)

        return move
