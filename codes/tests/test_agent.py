from game_types import Player
from game_types import Move
from game_types import Point
from agents.human import Human
from agents.zero_agent import ZeroAgent
import utils
import game_types as game_types

DEBUG = False


class TestMCTS:
    def __init__(self):
        game_type = "MiniOmok"
        rule_type = "OmokFreeRule"

        agent_version = 13
        num_threads = 2
        agent_file_name = utils.get_agent_filename(game_type, agent_version)
        loaded_agent = ZeroAgent.load_agent(agent_file_name, 'cuda:1', num_threads, False)
        loaded_agent.simulations_per_move = 300

        self.players = {
            Player.black: loaded_agent,
            Player.white: Human(),
        }

        self.board_size = game_types.board_size_dict[game_type]

        self.game_state_constructor = utils.get_game_state_constructor(game_type)
        self.rule_constructor = utils.get_rule_constructor(game_type, rule_type)
        self.rule = self.rule_constructor(self.board_size)

        self.black_moves1 = [[4, 3], [4, 4], [4, 5], [4, 6]]
        self.white_moves1 = [[0, 0], [0, 2], [0, 4], [0, 6]]

        self.black_moves2 = [[0, 0], [0, 2], [0, 4], [0, 6]]
        self.white_moves2 = [[4, 2], [4, 3], [4, 5], [4, 6]]

        self.black_moves3 = [[0, 0], [0, 2], [1, 6]]
        self.white_moves3 = [[4, 2], [4, 3], [4, 4]]

        self.test_cases = [True, True, True]

    def simulation(self, black_moves, white_moves):
        game_state = self.game_state_constructor.new_game(self.board_size, self.rule)

        for black_move, white_move in zip(black_moves, white_moves):
            move = Move(Point(*black_move))
            game_state = game_state.apply_move(move)

            move = Move(Point(*white_move))
            game_state = game_state.apply_move(move)

        utils.print_board(game_state.board)

        next_move, visit_counts = self.players[Player.black].select_move(game_state)
        game_state = game_state.apply_move(next_move)
        print(next_move.point)

        utils.print_visit_count(visit_counts)
        utils.print_board(game_state.board)

    def run(self):
        if(self.test_cases[0]):
            self.simulation(self.black_moves1, self.white_moves1)

        if(self.test_cases[1]):
            self.simulation(self.black_moves2, self.white_moves2)

        if(self.test_cases[2]):
            self.simulation(self.black_moves3, self.white_moves3)


if __name__ == '__main__':
    test = TestMCTS()
    test.run()
