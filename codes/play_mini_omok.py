import argparse
import os
import torch

from games.game_components import Player
from agents.human import Human
from agents.zero_agent import ZeroAgent
from ui.cui import CUI
from ui.gui import GUI

from configuration import GameType, RuleType, Configuration

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play 9x9 omok.')
    parser.add_argument('-n', '--numai', dest='numai', type=int, default=2, choices=[0, 1, 2], help='number of AI agents.')
    parser.add_argument('-a', '--aifirst', dest='aifirst', action='store_true', help='if numai is 1, ai plays first.')
    parser.add_argument('-c', '--cui', dest='cui', action='store_true', help='play game on cui.')
    
    args = parser.parse_args()
    
    # config = Configuration(
    #     GameType.MINI_OMOK,
    #     RuleType.OMOK_BASE,
    # )
    
    # config = Configuration(
    #     GameType.TICTACTOE,
    #     RuleType.TICTACTOE_BASE,
    # )

    config = Configuration(
        GameType.OMOK,
        RuleType.OMOK_BASE,
        simulations_per_move=500,
    )

    sample_models = {
        GameType.TICTACTOE: './sample_models/TICTACTOE/model.pth',
        GameType.MINI_OMOK: './sample_models/MINI_OMOK/model.pth',
        GameType.OMOK: './trained_models/OMOK/-v7.pth'
    }
    
    ui_type = "GUI"
    if args.cui:
        ui_type = "CUI"
    num_ai = args.numai
    ai_first = args.aifirst

    # AI settings
    agent_file_name = sample_models[config.game_type]

    if num_ai == 0:
        black_player = Human()
        white_player = Human()
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        loaded_agent = ZeroAgent.load_agent(agent_file_name, device, config.simulations_per_move, config.num_threads, False)
        
        if num_ai == 1:
            if ai_first is True:
                black_player = loaded_agent
                white_player = Human()
            else:
                black_player = Human()
                white_player = loaded_agent
        else:
            black_player = loaded_agent
            white_player = loaded_agent

    players = {
        Player.black: black_player,
        Player.white: white_player,
    }

    if ui_type == "GUI" or ui_type == "gui":
        ui = GUI(config, players)
        ui.run()
    else:
        ui = CUI(config, players)
        ui.run()
