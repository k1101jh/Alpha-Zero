from codes.games.tic_tac_toe.game import TicTacToe
from codes.types import Player
from codes.agents.human import Human
from codes.ui.cui import CUI

if __name__ == '__main__':
    players = {
        Player.black: Human(),
        Player.white: Human(),
    }
    game = TicTacToe
    # game = TicTacToe(players)
    ui = CUI(game, players)
    ui.run()
