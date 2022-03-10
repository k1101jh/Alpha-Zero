from games.game_types import Player
from agents.human import Human
from agents.zero_agent import ZeroAgent
from ui.cui import CUI
from ui.gui import GUI
from utils import get_agent_filename


if __name__ == '__main__':
    # game list: "TicTacToe", "MiniOmok"(9 x 9), "Omok"(15 x 15)
    # ui list: "GUI", "CUI"
    # game_type = "MiniOmok"
    
    # game_type = "Othello"
    # rule_type = "OthelloRule"
    game_type = "MiniOmok"
    rule_type = "OmokFreeRule"
    ui_type = "CUI"
    num_ai = 2
    ai_first = False

    # AI settings
    agent_version = 29
    num_threads = 1
    simulations_per_move = 350

    if num_ai == 0:
        black_player = Human()
        white_player = Human()
    else:
        agent_file_name = get_agent_filename(game_type, agent_version)
        # agent_file_name = get_agent_filename(game_type, agent_version, postfix='1thread500sim1000epoch/')
        loaded_agent = ZeroAgent.load_agent(agent_file_name, 'cuda:1', num_threads, False)
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
