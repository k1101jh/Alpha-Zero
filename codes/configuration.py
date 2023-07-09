from enum import Enum, auto

import os
import importlib


class GameType(Enum):
    TICTACTOE = auto()
    MINI_OMOK = auto()
    OMOK = auto()


class RuleType(Enum):
    TICTACTOE_BASE = auto()
    OMOK_BASE = auto()
    OMOK_RENJU = auto()


class EncoderType(Enum):
    ZERO_ENCODER = auto()


class ModelType(Enum):
    ALPHA_ZERO = auto()
    KATAGO = auto()


game_state_constructor_dict = {
    GameType.TICTACTOE: "games.tic_tac_toe.tic_tac_toe_game_state.TicTacToeGameState",
    GameType.MINI_OMOK: "games.omok.omok_game_state.OmokGameState",
    GameType.OMOK: "games.omok.omok_game_state.OmokGameState",
    # "Othello": "OthelloGameState"
}

rule_constructor_dict = {
    RuleType.TICTACTOE_BASE: "games.tic_tac_toe.tic_tac_toe_rule.TicTacToeRule",
    RuleType.OMOK_BASE: "games.omok.omok_rule.OmokFreeRule",
    RuleType.OMOK_RENJU: "games.omok.omok_rule.OmokRenjuRule",
}

encoder_constructor_dict = {
    EncoderType.ZERO_ENCODER: "encoders.zero_encoder.ZeroEncoder"
}

model_dict = {
    
}

board_size_dict = {
    GameType.TICTACTOE: 3,
    GameType.MINI_OMOK: 9,
    GameType.OMOK: 15,
    # "Othello": 8,
}


def get_constructor(string):
    parent_name, child_name = string.rsplit('.', 1)
    module = importlib.import_module(parent_name)
    constructor = getattr(module, child_name)
    return constructor


def get_game_state_constructor(game_type):
    return get_constructor(game_state_constructor_dict[game_type])


def get_rule_constructor(rule_type):
    return get_constructor(rule_constructor_dict[rule_type])


def get_encoder(encoder_type):
    return get_constructor(encoder_constructor_dict[encoder_type])


class Configuration:
    def __init__(self, game_type, rule_type, encoder_type=EncoderType.ZERO_ENCODER,
                 epochs=2000, test_term=20, learning_rate=4e-4, batch_size=512,
                 num_devices=1, num_processes=4, num_threads=1,
                 num_simulate_games=4, num_test_games=10,
                 max_memory_size=300000, simulations_per_move=500):
        """ Game Train & Test configuration
        Train configurations only work when training

        Args:
            game_type (GameType): Game to simulation.
            rule_type (RuleType): Game Rule type.
            encoder_type (EncoderType): Encoder type.
            epochs (int, optional): Number of epochs to train. Defaults to 2000.
            test_term (int, optional): Test model every {test_term} epochs. Defaults to 20.
            learning_rate (double, optional): Learning rate. Defaults to 4e-4.
            batch_size (int, optional): Batch Size. Defaults to 512
            num_devices (int, optional): Number of devices. Defaults to 1.
            num_processes (int, optional): Number of processes. Defaults to 4.
            num_threads (int, optional): Number of threads of each process. This can makes MCTS process faster, but not fully implemented.
                                         Defaults to 1.
            num_simulate_games (int, optional): Number of games to simulate simultaneously. Defaults to 4.
            num_test_games (int, optional): Number of test games. Defaults to 10.
            max_memory_size (int, optional): Maximum number of stored game experiences. Defaults to 300000.
            simulations_per_move (int, optional): Number of simulation games to decide one action. Defaults to 500.
        """
        assert(encoder_type in encoder_constructor_dict)
        assert(num_test_games % 2 == 0)
        self.game_type = game_type
        self.rule_type = rule_type
        self.encoder_type = encoder_type
        self.board_size = board_size_dict[game_type]
        
        self.epochs = epochs
        self.test_term = test_term
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.num_devices = num_devices
        self.num_processes = num_processes if num_processes != -1 else os.cpu_count()
        self.num_threads = num_threads
        self.num_simulate_games = num_simulate_games if num_simulate_games != -1 else os.cpu_count()
        self.num_test_games = num_test_games
        
        self.max_memory_size = max_memory_size
        self.simulations_per_move = simulations_per_move
