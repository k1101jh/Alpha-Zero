from multiprocessing import Queue
import os
import copy
import time
from typing import Iterable, Tuple
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from agents.abstract_agent import AbstractAgent
from agents.zero_agent import ZeroAgent
from encoders.zero_encoder import ZeroEncoder
from games.game import Game
from networks.alpha_zero import AlphaZeroModel
from games.experience import ExperienceCollector
from games.experience import ExperienceDataset
from games.game_types import board_size_dict
from games.game_types import Player
import utils


writer = SummaryWriter('../runs')

NUM_DEVICES = torch.cuda.device_count()
AGENT_NAME = "ZeroAgent"
GAME_TYPE = "MiniOmok"
RULE_TYPE = "OmokFreeRule"
BOARD_SIZE = board_size_dict[GAME_TYPE]
MAX_MEMORY_SIZE = 30000
EPOCHS = 2000
TEST_TERM = 20
BATCH_SIZE = 512
SIMULATIONS_PER_MOVE = 500
NUM_MAX_PROCESSES = os.cpu_count()
NUM_SIMULATE_GAMES = os.cpu_count()
NUM_TEST_GAMES_HALF = os.cpu_count()
NUM_THREADS_PER_ROUND = 1
LEARNING_RATE = 4e-4
LOAD_AGENT_VERSION = 50


def train() -> None:
    mp.set_start_method('spawn')
    agent_version = LOAD_AGENT_VERSION
    load_agent_filename = utils.get_agent_filename(GAME_TYPE, LOAD_AGENT_VERSION)

    encoder = ZeroEncoder(BOARD_SIZE)

    # load or generate agent
    if os.path.exists(load_agent_filename):
        agent = ZeroAgent.load_agent(load_agent_filename, 'cpu',
                                     num_threads_per_round=NUM_THREADS_PER_ROUND,
                                     noise=True)
    else:
        model = AlphaZeroModel(encoder.shape()[0], num_blocks=8, board_size=BOARD_SIZE)
        agent = ZeroAgent(encoder, model, device='cpu',
                          simulations_per_move=SIMULATIONS_PER_MOVE,
                          num_threads_per_round=NUM_THREADS_PER_ROUND,
                          lr=LEARNING_RATE)
        agent_version = 0

    prev_agent = copy.deepcopy(agent)
    train_agent = copy.deepcopy(agent)
    train_agent.set_device('cuda:0')
    # generate experience memory
    memory = ExperienceDataset(BOARD_SIZE, encoder.num_planes, MAX_MEMORY_SIZE)

    start_epoch = agent.epoch
    for epoch in tqdm(range(start_epoch, EPOCHS), initial=start_epoch):
        num_black_wins = 0
        num_white_wins = 0
        num_draw = 0

        # generate experience
        game_results, collectors = generate_experience(NUM_SIMULATE_GAMES, GAME_TYPE, RULE_TYPE, agent, copy.deepcopy(agent), collect_exp=True)
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
        loss, policy_loss, value_loss = train_agent.train(memory, BATCH_SIZE)
        writer.add_scalars('loss',
                           {'Policy Loss': policy_loss,
                            'Value Loss': value_loss,
                            'Epoch Loss': loss},
                           epoch)
        agent.model.load_state_dict(train_agent.model.state_dict())
        agent.optimizer.load_state_dict(train_agent.optimizer.state_dict())
        agent.scheduler.load_state_dict(train_agent.scheduler.state_dict())
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
            prev_agent = copy.deepcopy(agent)

    writer.close()


def simulate_game(game_type: str, rule_type: str, black_agent: AbstractAgent, white_agent: AbstractAgent,
                  device_num: int, result_queue: Queue, collect_experience: bool = False) -> None:
    start_time = time.time()

    players = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    game = Game(game_type, rule_type, players)

    # set device
    black_agent.set_device(device_num)
    white_agent.set_device(device_num)

    if collect_experience is True:
        black_collector = ExperienceCollector(BOARD_SIZE)
        white_collector = ExperienceCollector(BOARD_SIZE)

        black_agent.set_collector(black_collector)
        white_agent.set_collector(white_collector)

        black_collector.begin_episode()
        white_collector.begin_episode()

        game.start()
        game.join()

        game_result = game.get_game_state().winner

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
        game.start()
        game.join()

        game_result = game.get_game_state().winner
        return_val = (game_result, None)

    utils.print_board(game.get_game_state().board)
    utils.print_winner(game_result)

    torch.cuda.empty_cache()
    print('simulation elapsed time:', time.time() - start_time)
    result_queue.put(return_val)


def generate_experience(num_games: int, game_type: str, rule_type: str,
                        black_agent: AbstractAgent, white_agent: AbstractAgent,
                        collect_exp: bool) -> Tuple[Iterable[Player], Iterable[ExperienceCollector]]:
    start_time = time.time()

    num_remain_games = num_games
    collectors = []
    winners = []

    while num_remain_games > 0:
        num_processes = min(NUM_MAX_PROCESSES, num_remain_games)
        print("Simulate games %d / %d..." %
              (num_games - num_remain_games, num_games))

        result_queue = mp.Queue()
        processes = []
        for process_num in range(num_processes):
            device_num = process_num % NUM_DEVICES

            p = mp.Process(target=simulate_game,
                           args=(game_type,
                                 rule_type,
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
                queue_item = result_queue.get()
                winner = queue_item[0]
                winners.append(winner)
                if collect_exp:
                    collectors.append(queue_item[1][0])
                    collectors.append(queue_item[1][1])
                num_remain_games -= 1
            if not running:
                break

    print('\n%d games elapsed time: %d' % (num_games, time.time() - start_time))

    return winners, collectors


def evaluate_bot(game_type: str, rule_type: str, agent: AbstractAgent, prev_agent: AbstractAgent) -> Tuple[float, float]:
    num_win = 0
    num_lose = 0
    num_draw = 0
    eval_agent_color = Player.black
    black_agent, white_agent = agent, prev_agent

    for _ in range(2):
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
    win_rate = num_win / (NUM_TEST_GAMES_HALF * 2)
    draw_rate = num_draw / (NUM_TEST_GAMES_HALF * 2)
    print('Win:', num_win, ' Lose:', num_lose, ' Draw:', num_draw, ' Win Rate:', win_rate)

    return win_rate, draw_rate


if __name__ == "__main__":
    train()
