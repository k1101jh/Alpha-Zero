import os
from games.game_components import Player
from agents.human import Human
from agents.zero_agent import ZeroAgent
from ui.cui import CUI
from ui.gui import GUI
from utils import get_agent_filename
from configuration import Configuration, GameType, RuleType

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


if __name__ == '__main__':
    config = Configuration(
        GameType.OMOK,
        RuleType.OMOK_BASE,
        simulations_per_move=500
    )
    ui_type = "GUI"
    num_ai = 2
    ai_first = False

    # AI settings
    agent_version = 16

    if num_ai == 0:
        black_player = Human()
        white_player = Human()
    else:
        agent_file_name = get_agent_filename(config.game_type, agent_version)
        loaded_agent = ZeroAgent.load_agent(agent_file_name, 'cuda:0', config.simulations_per_move, config.num_threads, False)
        
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
