import os
import copy
import time
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from codes.agents.zero_agent import ZeroAgent
from codes.encoders.zero_encoder import ZeroEncoder
from codes.networks.alpha_zero import AlphaZeroModel
from codes.experience import ExperienceCollector
from codes.experience import ExperienceDataset
from codes.ui.cui import CUI
from codes.game_types import board_size_dict
from codes.game_types import Player
from codes import utils


writer = SummaryWriter('../runs')

NUM_DEVICES = torch.cuda.device_count()
AGENT_NAME = "ZeroAgent"
GAME_TYPE = "MiniOmok"
RULE_TYPE = "FreeRule"
BOARD_SIZE = board_size_dict[GAME_TYPE]
MAX_MEMORY_SIZE = 40000
EPOCHS = 1000
TEST_TERM = 10
BATCH_SIZE = 512
ROUNDS_PER_MOVE = 300
NUM_SIMULATE_GAMES = 10
NUM_THREADS_PER_ROUND = 2
NUM_TEST_GAMES_HALF = 5
LEARNING_RATE = 3e-3
LOAD_AGENT_VERSION = 1


def train():
    mp.set_start_method('spawn')
    agent_version = LOAD_AGENT_VERSION
    load_agent_filename = utils.get_agent_filename(GAME_TYPE, LOAD_AGENT_VERSION)

    encoder = ZeroEncoder(BOARD_SIZE)

    # load or generate agent
    if os.path.exists(load_agent_filename):
        agent = ZeroAgent.load_agent(load_agent_filename, 'cuda:0', NUM_THREADS_PER_ROUND)
    else:
        model = AlphaZeroModel(encoder.shape()[0], num_blocks=5, board_size=BOARD_SIZE)
        agent = ZeroAgent(encoder, model, device='cuda:0',
                          rounds_per_move=ROUNDS_PER_MOVE,
                          num_threads_per_round=NUM_THREADS_PER_ROUND,
                          lr=LEARNING_RATE)
        agent_version = 0

    # generate experience memory
    memory = ExperienceDataset(BOARD_SIZE, encoder.num_planes, MAX_MEMORY_SIZE)

    start_epoch = agent.epoch
    for epoch in tqdm(range(start_epoch, EPOCHS), initial=start_epoch):
        num_black_wins = 0
        num_white_wins = 0
        num_draw = 0

        prev_agent = copy.deepcopy(agent)

        # generate experience
        game_results, collectors = generate_experience(NUM_SIMULATE_GAMES, GAME_TYPE, RULE_TYPE, agent, prev_agent, collect_exp=True)
        memory.add_experiences(collectors)

        # game statistics
        for winner in game_results:
            if winner is Player.black:
                num_black_wins += 1
            elif winner is Player.white:
                num_white_wins += 1
            else:
                num_draw += 1

        writer.add_scalars('game statistics',
                           {'Black Win Rate': num_black_wins / NUM_SIMULATE_GAMES,
                            'White Win Rate': num_white_wins / NUM_SIMULATE_GAMES,
                            'Draw Rate': num_draw / NUM_SIMULATE_GAMES},
                           epoch)

        # train agent
        loss, policy_loss, value_loss = agent.train(memory, BATCH_SIZE)
        writer.add_scalars('loss',
                           {'Policy Loss': policy_loss,
                            'Value Loss': value_loss,
                            'Epoch Loss': loss},
                           epoch)

        agent.add_num_simulated_games(NUM_SIMULATE_GAMES)
        agent.epoch = epoch

        if (epoch + 1) % TEST_TERM == 0:
            win_rate, draw_rate = evaluate_bot(GAME_TYPE, RULE_TYPE, agent, prev_agent)
            writer.add_scalars('evaluate results',
                               {'Win Rate': win_rate,
                                'Draw Rate': draw_rate},
                               epoch)

            agent_version += 1
            update_agent_filename = utils.get_agent_filename(GAME_TYPE, agent_version)
            agent.save_agent(update_agent_filename)

    writer.close()


def simulate_game(game_type, rule_type, black_agent, white_agent, collect_experience=False):
    start_time = time.time()

    players = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    ui = CUI(game_type, rule_type, players)

    if collect_experience is True:
        black_collector = ExperienceCollector()
        white_collector = ExperienceCollector()

        black_agent.set_collector(black_collector)
        white_agent.set_collector(white_collector)

        black_collector.begin_episode()
        white_collector.begin_episode()

        ui = CUI(game_type, rule_type, players)
        ui.run()
        game_result = ui.get_game().get_game_state().get_winner()

        # set reward
        if game_result == Player.black:
            black_collector.complete_episode(1)
            white_collector.complete_episode(-1)
        elif game_result == Player.white:
            black_collector.complete_episode(-1)
            white_collector.complete_episode(1)
        else:
            black_collector.complete_episode(0)
            white_collector.complete_episode(0)

        return_val = (game_result, (black_collector, white_collector))
    # evaluate game
    else:
        ui.run()
        game_result = ui.get_game().get_game_state().get_winner()
        return_val = (game_result, None)

    torch.cuda.empty_cache()
    print('simulation elapsed time:', time.time() - start_time)

    return return_val


def generate_experience(num_games, game_type, rule_type, black_agent, white_agent, collect_exp):
    start_time = time.time()

    collectors = []
    winners = []

    for game_cnt in range(num_games):
        print("Simulate games %d / %d..." %
              (game_cnt + 1, num_games))

        collectors = []
        winner, result_collectors = simulate_game(game_type, rule_type, black_agent, white_agent, collect_exp)

        winners.append(winner)
        if(collect_exp):
            collectors.append(result_collectors[0])
            collectors.append(result_collectors[1])

    print('\n%d games elapsed time: %d' % (num_games, time.time() - start_time))

    return winners, collectors


def evaluate_bot(game_type, rule_type, agent, prev_agent):
    num_win = 0
    num_lose = 0
    num_draw = 0
    eval_agent_color = Player.black
    black_agent, white_agent = agent, prev_agent

    for i in range(2):
        winners, _ = generate_experience(NUM_TEST_GAMES_HALF, game_type, rule_type, black_agent, white_agent, collect_exp=False)

        for winner in winners:
            print('Winner:', winner)
            if winner is eval_agent_color:
                num_win += 1
            elif winner is eval_agent_color.other:
                num_lose += 1
            else:
                num_draw += 1
        white_agent, black_agent = agent, prev_agent
        eval_agent_color = eval_agent_color.other

    # calc win rate, draw rate
    win_rate = num_win / NUM_TEST_GAMES_HALF
    draw_rate = num_draw / NUM_TEST_GAMES_HALF
    print('Win:', num_win, ' Lose:', num_lose, ' Draw:', num_draw, ' Win Rate:', win_rate)

    return win_rate, draw_rate


if __name__ == "__main__":
    train()
