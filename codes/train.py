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
from games.game import Game
from networks.alpha_zero import AlphaZeroModel
from games.experience import ExperienceCollector
from games.experience import ExperienceDataset
from games.game_components import Player
from configuration import Configuration
from configuration import GameType, RuleType, EncoderType, get_encoder
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

writer = SummaryWriter('runs')

AGENT_NAME = "ZeroAgent"
LOAD_AGENT_VERSION = 15

config = Configuration(
    # game_type=GameType.TICTACTOE,
    # rule_type=RuleType.TICTACTOE_BASE,
    game_type=GameType.OMOK,
    rule_type=RuleType.OMOK_BASE,
    encoder_type=EncoderType.ZERO_ENCODER,

    # training settings
    epochs=2000,
    test_term=50,
    learning_rate=4e-4,
    batch_size=512,
    
    # agent simulation settings
    num_devices=4,
    num_processes=16,
    num_threads=1,
    num_simulate_games=32,
    num_test_games=16,
    max_memory_size=20000,
    simulations_per_move=700,
)


def train() -> None:
    mp.set_start_method('spawn')
    agent_version = LOAD_AGENT_VERSION
    load_agent_filename = utils.get_agent_filename(config.game_type, LOAD_AGENT_VERSION)
    encoder = get_encoder(config.encoder_type)(config.board_size)

    # load or generate agent
    # if agent file exists, load agent
    if os.path.exists(load_agent_filename):
        agent = ZeroAgent.load_agent(load_agent_filename, 'cpu',
                                     simulations_per_move=config.simulations_per_move,
                                     num_threads_per_round=config.num_threads,
                                     noise=True)
    else:
        model = AlphaZeroModel(encoder.num_planes, num_blocks=10, board_size=config.board_size)
        agent = ZeroAgent(encoder, model, device='cpu',
                          simulations_per_move=config.simulations_per_move,
                          num_threads_per_round=config.num_threads,
                          lr=config.learning_rate)

        agent_version = 0

    prev_agent = copy.deepcopy(agent)
    train_agent = copy.deepcopy(agent)
    train_agent.set_device('cuda:3')
    # agents = []
    # prev_agents = []
    # for i in range(config.num_devices):
    #     prev_agent = copy.deepcopy(agent)
    #     prev_agent.set_device(f'cuda:{i % 4}')
    #     new_agent = copy.deepcopy(agent)
    #     new_agent.set_device(f'cuda:{i % 4}')
    #     agents.append(new_agent)
    #     prev_agents.append(prev_agent)
    
    # generate experience memory
    memory = ExperienceDataset(config.board_size, encoder.num_planes, config.max_memory_size)

    start_epoch = train_agent.epoch + 1
    for epoch in tqdm(range(start_epoch, config.epochs + 1), initial=start_epoch):
        num_black_wins = 0
        num_white_wins = 0
        num_draw = 0

        # generate experience
        # game_results, collectors = generate_experience(config, config.num_simulate_games, encoder, agent, copy.deepcopy(agent), collect_exp=True)
        game_results, collectors = generate_experience(config, config.num_simulate_games, encoder, agent, agent, collect_exp=True)
        memory.add_experiences(collectors)

        # game statistics
        for winner in game_results:
            if winner is Player.black:
                num_black_wins += 1
            elif winner is Player.white:
                num_white_wins += 1
            else:
                num_draw += 1

        num_actual_simulated_games = len(game_results)

        writer.add_scalars('game statistics',
                           {'Black Win Rate': num_black_wins / num_actual_simulated_games,
                            'White Win Rate': num_white_wins / num_actual_simulated_games,
                            'Draw Rate': num_draw / num_actual_simulated_games},
                           epoch)

        # train agent
        loss, policy_loss, value_loss = train_agent.train(memory, config.batch_size)
        writer.add_scalars('loss',
                           {'Policy Loss': policy_loss,
                            'Value Loss': value_loss,
                            'Epoch Loss': loss},
                           epoch)
        agent.model.load_state_dict(train_agent.model.state_dict())
        train_agent.add_num_simulated_games(num_actual_simulated_games)
        train_agent.epoch = epoch
        
        # for agent in agents[1:]:
        #     agent.model.load_state_dict(train_agent.model.state_dict())
        #     agent.optimizer.load_state_dict(train_agent.optimizer.state_dict())
        #     agent.scheduler.load_state_dict(train_agent.scheduler.state_dict())
        # agent.add_num_simulated_games(num_actual_simulated_games)
        # agent.epoch = epoch

        if epoch % config.test_term == 0:
            win_rate, draw_rate = evaluate_bot(config, encoder, agent, prev_agent)
            writer.add_scalars('evaluate results',
                               {'Win Rate': win_rate,
                                'Draw Rate': draw_rate},
                               epoch)

            agent_version += 1
            update_agent_filename = utils.get_agent_filename(config.game_type, agent_version)
            train_agent.save_agent(update_agent_filename)
            prev_agent.model.load_state_dict(train_agent.model.state_dict())

    writer.close()


def simulate_game(config: Configuration, encoder, black_agent: AbstractAgent, white_agent: AbstractAgent,
                  device_num: int, result_queue: Queue, collect_experience: bool = False) -> None:
    start_time = time.time()

    players = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    game = Game(config, encoder, collect_experience, players)
    
    black_agent.set_device(device_num)
    white_agent.set_device(device_num)

    game.start()
    game.join()

    game_result = game.get_game_state().winner
    
    return_val = (game_result, None)
    # Train 시
    if collect_experience is True:
        collectors = list(game.get_collectors().values())
        return_val = (game_result, collectors)

    utils.print_board(game.get_game_state().board)
    utils.print_winner(game_result)

    torch.cuda.empty_cache()
    print('simulation execution time:', time.time() - start_time)
    result_queue.put(return_val)


def generate_experience(config: Configuration, num_games, encoder,
                        black_agent, white_agent,
                        collect_exp: bool) -> Tuple[Iterable[Player], Iterable[ExperienceCollector]]:
    start_time = time.time()

    num_remain_games = num_games
    num_devices = config.num_devices
    collectors = []
    winners = []

    while num_remain_games > 0:
        num_processes = min(config.num_processes, num_remain_games)
        print("Simulate games %d / %d..." % (num_games - num_remain_games, num_games))

        result_queue = mp.Queue()
        processes = []
        for process_num in range(num_processes):
            device_num = process_num % num_devices

            p = mp.Process(target=simulate_game,
                           args=(config,
                                 encoder,
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
                    for collector in queue_item[1]:
                        collectors.append(collector)
                num_remain_games -= 1
            if not running:
                break
            
        time.sleep(2)

    print('\n%d games execution time: %d' % (num_games, time.time() - start_time))

    return winners, collectors


def evaluate_bot(config: Configuration, encoder, agent: AbstractAgent, prev_agent: AbstractAgent) -> Tuple[float, float]:
    num_win = 0
    num_lose = 0
    num_draw = 0
    eval_agent_color = Player.black
    black_agent, white_agent = agent, prev_agent

    for _ in range(2):
        winners, _ = generate_experience(config, config.num_test_games // 2, encoder, black_agent, white_agent, collect_exp=False)

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
    num_games = (num_win + num_lose + num_draw)
    win_rate = num_win / num_games
    draw_rate = num_draw / num_games
    print('Win:', num_win, ' Lose:', num_lose, ' Draw:', num_draw, ' Win Rate:', win_rate)

    return win_rate, draw_rate


if __name__ == "__main__":
    train()
