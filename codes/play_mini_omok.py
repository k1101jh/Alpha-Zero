import argparse

import torch

from games.game_types import Player
from agents.human import Human
from agents.zero_agent import ZeroAgent
from ui.cui import CUI
from ui.gui import GUI


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play 9x9 omok.')
    parser.add_argument('-n', '--numai', dest='numai', type=int, default=2, choices=[0, 1, 2], help='number of AI agents.')
    parser.add_argument('-a', '--aifirst', dest='aifirst', action='store_true', help='if numai is 1, ai plays first.')
    parser.add_argument('-c', '--cui', dest='cui', action='store_true', help='play game on cui.')
    
    args = parser.parse_args()
    
    game_type = "MiniOmok"
    rule_type = "OmokFreeRule"
    ui_type = "GUI"
    if args.cui:
        ui_type = 'CUI'
    num_ai = args.numai
    ai_first = args.aifirst

    # AI settings
    agent_file_name = './sample_models/MiniOmok/model.pth'
    num_threads = 1
    simulations_per_move = 500

    if num_ai == 0:
        black_player = Human()
        white_player = Human()
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        loaded_agent = ZeroAgent.load_agent(agent_file_name, device, num_threads, False)
        loaded_agent.simulations_per_move = simulations_per_move
        
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
        ui = GUI(game_type, rule_type, players)
        ui.run()
    else:
        ui = CUI(game_type, rule_type, players)
        ui.run()
