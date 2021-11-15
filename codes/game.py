from codes.games.tic_tac_toe.game import TicTacToe
from codes.games.tic_tac_toe.rule import TicTacToeRule
from codes.types import Player
from codes.agents.human import Human
from codes.ui.cui import CUI

if __name__ == '__main__':
    players = {
        Player.black: Human(),
        Player.white: Human(),
    }
    game_constructor = TicTacToe
    rule_constructor = TicTacToeRule
    # game = TicTacToe(players)
    ui = CUI(game_constructor, rule_constructor, players)
    ui.run()
