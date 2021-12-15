import sys

from codes.types import Player
from codes.types import Move
from codes.types import Point
from codes.agents.human import Human
from codes.agents.zero_agent import ZeroAgent
from codes.agents.mcts_agent import MCTSAgent
from codes.encoders.zero_encoder import ZeroEncoder
from codes import utils
from codes import types

DEBUG = False


class TestMCTS:
    def __init__(self):
        game_name = "Omok"
        rule_name = "FreeRule"
        agent_version = 16
        num_threads = 1
        white_agent_file_name = utils.get_agent_filename(game_name, agent_version)
        # encoder = ZeroEncoder(types.board_size_dict[game_name])
        # loaded_agent = MCTSAgent(encoder)
        loaded_agent = ZeroAgent.load_agent(white_agent_file_name, 'cuda:1', num_threads)
        self.players = {
            # Player.black: Human(),
            Player.white: Human(),
            Player.black: loaded_agent,
            # Player.white: loaded_agent
        }
        loaded_agent.noise = False
        loaded_agent.rounds_per_move = 500

        self.game_state_constructor = utils.get_game_state_constructor(game_name)
        self.rule_constructor = utils.get_rule_constructor(game_name, rule_name)

        self.black_moves = [[4, 3], [4, 4], [4, 5], [4, 6]]
        self.white_moves = [[0, 0], [0, 2], [0, 4], [0, 6]]

        self.black_moves2 = [[0, 0], [0, 2], [0, 4], [0, 6]]
        self.white_moves2 = [[4, 2], [4, 3], [4, 5], [4, 6]]

        self.black_moves3 = [[0, 0], [0, 2], [1, 6]]
        self.white_moves3 = [[4, 2], [4, 3], [4, 4]]

        self.test_cases = [True, True, True]

    def run(self):
        if(self.test_cases[0]):
            game_state = self.game_state_constructor.new_game(self.rule_constructor)

            for black_move, white_move in zip(self.black_moves, self.white_moves):
                move = Move(Point(*black_move))
                game_state = game_state.apply_move(move)

                move = Move(Point(*white_move))
                game_state = game_state.apply_move(move)

            # print(chr(27) + "[2J")
            utils.print_board(game_state.board)

            next_move, visit_counts = self.players[Player.black].select_move(game_state)
            game_state = game_state.apply_move(next_move)
            print(next_move.point)

            # print(chr(27) + "[2J")
            utils.print_visit_count(visit_counts)
            utils.print_board(game_state.board)

        # test 2
        if(self.test_cases[1]):
            game_state = self.game_state_constructor.new_game(self.rule_constructor)

            for black_move, white_move in zip(self.black_moves2, self.white_moves2):
                move = Move(Point(*black_move))
                game_state = game_state.apply_move(move)

                move = Move(Point(*white_move))
                game_state = game_state.apply_move(move)

            # print(chr(27) + "[2J")
            utils.print_board(game_state.board)

            next_move, visit_counts = self.players[Player.black].select_move(game_state)
            game_state = game_state.apply_move(next_move)

            # print(chr(27) + "[2J")
            utils.print_visit_count(visit_counts)
            utils.print_board(game_state.board)

        # test 3
        if(self.test_cases[2]):
            game_state = self.game_state_constructor.new_game(self.rule_constructor)

            for black_move, white_move in zip(self.black_moves3, self.white_moves3):
                move = Move(Point(*black_move))
                game_state = game_state.apply_move(move)

                move = Move(Point(*white_move))
                game_state = game_state.apply_move(move)

            # print(chr(27) + "[2J")
            utils.print_board(game_state.board)

            next_move, visit_counts = self.players[Player.black].select_move(game_state)
            game_state = game_state.apply_move(next_move)
            print(next_move.point)

            # print(chr(27) + "[2J")
            utils.print_visit_count(visit_counts)
            utils.print_board(game_state.board)


if __name__ == '__main__':
    test = TestMCTS()
    test.run()
