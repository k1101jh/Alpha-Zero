from codes.types import Player
from codes.agents.human import Human
from codes.agents.zero_agent import ZeroAgent
from codes.ui.cui import CUI
from codes.ui.gui import GUI
from codes.utils import get_agent_filename

if __name__ == '__main__':
    game_type = "MiniOmok"
    rule_type = "FreeRule"
    is_only_human = False

    if not is_only_human:
        agent_version = 136
        num_threads = 2
        white_agent_file_name = get_agent_filename(game_type, agent_version, postfix='T1/')
        loaded_agent = ZeroAgent.load_agent(white_agent_file_name, 'cuda:1', num_threads)
        loaded_agent.noise = False
        loaded_agent.rounds_per_move = 400

    players = {
        Player.black: Human(),
        # Player.white: Human(),
        # Player.black: loaded_agent,
        Player.white: loaded_agent
    }

    ui = GUI(game_type, rule_type, players)
    ui.run()
