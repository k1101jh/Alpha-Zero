import argparse

from codes.game_types import Player
from codes.agents.human import Human
from codes.agents.zero_agent import ZeroAgent
from codes.ui.cui import CUI
from codes.ui.gui import GUI
from codes.utils import get_agent_filename

if __name__ == '__main__':
    # game list: "TicTacToe", "MiniOmok"(9 x 9), , "Omok"(15 x 15)
    # ui list: "GUI", "CUI"
    game_type = "MiniOmok"
    rule_type = "FreeRule"
    ui_type = "GUI"
    num_ai = 1
    ai_first = False
    
    # AI settings
    simulations_per_move = 200
    agent_version = 136
    num_threads = 2

    if num_ai == 0:
        black_player = Human()
        white_player = Human()
    else:
        white_agent_file_name = get_agent_filename(game_type, agent_version)
        loaded_agent = ZeroAgent.load_agent(white_agent_file_name, 'cuda:0', num_threads, False)
        loaded_agent.simulations_per_move = simulations_per_move
        
        if num_ai == 1:
            if ai_first == True:
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
