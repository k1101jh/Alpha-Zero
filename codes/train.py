import os
import copy
import time
import importlib
import h5py
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from codes.agents.zero_agent import ZeroAgent
from codes.encoders.zero_encoder import ZeroEncoder
from codes.networks.alpha_zero import AlphaZeroModel
from codes.experience import ExperienceCollector
from codes.experience import ExperienceDataset
from codes.types import game_name_dict
from codes.types import board_sizes
from codes.types import Player
from codes import utils

writer = SummaryWriter('../runs')

NUM_DEVICES = torch.cuda.device_count()
AGENT_NAME = "ZeroAgent"
GAME_NAME = "TicTacToe"
RULE_NAME = "Rule"
BOARD_SIZE = board_sizes[GAME_NAME]
MAX_MEMORY_SIZE = 50000
EPOCHS = 100
BATCH_SIZE = 512
ROUNDS_PER_MOVE = 100
NUM_SIMULATE_GAMES = 12
NUM_TEST_GAMES = 24
NUM_MAX_PROCESSES = 12
LEARNING_RATE = 2e-3
LOAD_AGENT_VERSION = 1
SAVE_EPISODE = True
LOAD_SAVED_EPISODE = False
LOAD_EPISODE_VERSION = 1
SAVE_EPISODE_VERSION = 1
EPISODE_SAVE_PATH = utils.get_experience_filename(GAME_NAME, SAVE_EPISODE_VERSION)


def get_game_and_rule():
    game_module = importlib.import_module(f'codes.games.{game_name_dict[GAME_NAME]}.game')
    game_constructor = getattr(game_module, GAME_NAME)
    rule_module = importlib.import_module(f'codes.games.{game_name_dict[GAME_NAME]}.rule')
    rule_constructor = getattr(rule_module, GAME_NAME + RULE_NAME)

    return game_constructor, rule_constructor


def train():
    mp.set_start_method('spawn')
    agent_version = LOAD_AGENT_VERSION
    load_agent_filename = utils.get_agent_filename(GAME_NAME, LOAD_AGENT_VERSION)

    encoder = ZeroEncoder(BOARD_SIZE)
    memory = ExperienceDataset(BOARD_SIZE, encoder.num_planes, MAX_MEMORY_SIZE)
    game_constructor, rule_constructor = get_game_and_rule()
    # load or generate agent
    if os.path.exists(load_agent_filename):
        agent = ZeroAgent.load_agent(load_agent_filename, 'cpu')
    else:
        model = AlphaZeroModel(encoder.shape()[0], board_size=BOARD_SIZE)
        agent = ZeroAgent(encoder, model, device='cpu', rounds_per_move=ROUNDS_PER_MOVE)

    start_epoch = agent.epoch
    for epoch in tqdm(range(start_epoch, start_epoch + EPOCHS)):
        prev_agent = copy.deepcopy(agent)

        # generate experience
        collectors = generate_experience(game_constructor, rule_constructor, agent, prev_agent, collect_exp=True)
        memory.add_experiences(collectors)

        # train agent
        loss, policy_loss, value_loss = agent.train(memory, BATCH_SIZE)
        writer.add_scalars('loss',
                           {'Policy Loss': policy_loss,
                            'Value Loss': value_loss,
                            'Epoch Loss': loss},
                           epoch)

        agent.add_num_simulated_games(NUM_SIMULATE_GAMES)
        agent.epoch = epoch

        if (epoch + 1) % 10 == 0:
            win_rate, draw_rate = evaluate_bot(game_constructor, rule_constructor, agent, prev_agent)
            writer.add_scalars('win rate/draw rate',
                               {'Win Rate': win_rate,
                                'Draw Rate': draw_rate},
                               epoch)
            if(win_rate > 0.5 or draw_rate >= 0.95):
                agent_version += 1
                update_agent_filename = utils.get_agent_filename(GAME_NAME, agent_version)
                agent.save_agent(update_agent_filename)

    with h5py.File(EPISODE_SAVE_PATH, 'w') as episode_out:
        memory.serialize(episode_out)

    writer.close()


def simulate_game(game_constructor, rule_constructor, black_agent, white_agent, device_num, result_queue, collect_experience=False):
    start_time = time.time()

    players = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    game = game_constructor(rule_constructor, players)

    # set device
    black_agent.set_device(device_num)
    white_agent.set_device(device_num)

    if collect_experience is True:
        black_collector = ExperienceCollector()
        white_collector = ExperienceCollector()

        black_agent.set_collector(black_collector)
        white_agent.set_collector(white_collector)

        black_collector.begin_episode()
        white_collector.begin_episode()

        game.start()
        game.join()

        game_result = game.game_state.winner

        if game_result == Player.black:
            black_collector.complete_episode(1)
            white_collector.complete_episode(-1)
        elif game_result == Player.white:
            black_collector.complete_episode(-1)
            white_collector.complete_episode(1)
        else:
            black_collector.complete_episode(0)
            white_collector.complete_episode(0)

        result_queue.put(black_collector)
        result_queue.put(white_collector)
    # evaluate game
    else:
        game.start()
        game.join()

        game_result = game.game_state.winner
        result_queue.put(game_result)

    utils.print_board(game.game_state.board)
    utils.print_winner(game_result)

    torch.cuda.empty_cache()
    print('simulation elapsed time:', time.time() - start_time)


def generate_experience(game_constructor, rule_constructor, black_agent, white_agent, collect_exp):
    start_time = time.time()

    num_remain_games = NUM_SIMULATE_GAMES
    results = []

    while num_remain_games > 0:
        num_processes = min(NUM_MAX_PROCESSES, num_remain_games)
        num_complete_games = 0
        print("Simulate games %d / %d..." %
              ((NUM_SIMULATE_GAMES - num_remain_games), NUM_SIMULATE_GAMES))

        result_queue = mp.Queue()
        processes = []
        for process_num in range(num_processes):
            device_num = process_num % NUM_DEVICES

            p = mp.Process(target=simulate_game,
                           args=(game_constructor,
                                 rule_constructor,
                                 black_agent,
                                 white_agent,
                                 device_num,
                                 result_queue,
                                 collect_exp))
            p.daemon = True
            p.start()
            processes.append(p)

        while processes:
            running = any(p.is_alive() for p in processes)
            while not result_queue.empty():
                results.append(result_queue.get())
                num_complete_games += 1
            if not running:
                break

        num_remain_games = NUM_SIMULATE_GAMES - num_complete_games

    print('\n%d games elapsed time: %d' % (NUM_SIMULATE_GAMES, time.time() - start_time))

    return results


def evaluate_bot(game_constructor, rule_constructor, agent, prev_agent):
    num_win = 0
    num_lose = 0
    num_draw = 0
    eval_agent_color = Player.black
    black_agent, white_agent = agent, prev_agent

    for i in range(2):
        winners = generate_experience(game_constructor, rule_constructor, black_agent, white_agent, collect_exp=False)

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
    win_rate = num_win / NUM_TEST_GAMES
    draw_rate = num_draw / NUM_TEST_GAMES
    print('Win:', num_win, ' Lose:', num_lose, ' Draw:', num_draw, ' Win Rate:', win_rate)

    return win_rate, draw_rate


if __name__ == "__main__":
    train()
