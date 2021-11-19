import asyncio
from codes.types import Player
from codes.agents.human import Human
from codes.agents.zero_agent import ZeroAgent
from codes.ui.cui import CUI
from codes.utils import get_agent_filename
from codes.utils import get_game_state_constructor
from codes.utils import get_rule_constructor

if __name__ == '__main__':
    game_name = "Omok"
    rule_name = "FreeRule"
    agent_version = 11
    white_agent_file_name = get_agent_filename(game_name, agent_version)
    players = {
        # Player.black: Human(),
        # Player.white: Human(),
        Player.black: ZeroAgent.load_agent(white_agent_file_name, 'cuda:0'),
        Player.white: ZeroAgent.load_agent(white_agent_file_name, 'cuda:0')
    }
    game_state_constructor = get_game_state_constructor(game_name)
    rule_constructor = get_rule_constructor(game_name, rule_name)
    ui = CUI(game_state_constructor, rule_constructor, players)
    ui.run()
