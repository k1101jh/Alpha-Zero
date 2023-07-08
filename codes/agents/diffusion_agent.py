# 참고한 코드: https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/master/code/dlgo/zero/agent.py
from typing import Dict, Iterable, Optional, Tuple, TypeVar
import numpy as np
import heapq
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from agents.abstract_agent import AbstractAgent
from encoders.zero_encoder import ZeroEncoder
from games.experience import ExperienceCollector, ExperienceDataset
from games.game_types import Move, Player
from games.abstract_game_state import AbstractGameState
from networks.vae import VAE


SelfZeroAgent = TypeVar("SelfZeroAgent", bound="DiffusionAgent")


class DiffusionAgent(AbstractAgent):
    def __init__(self, encoder: ZeroEncoder, model: nn.Module, device: str,
                 input_size: int = 3, beta_start: float = 1e-4, beta_end: float = 0.02,
                 num_sampling_steps: int = 100, lr: float = 1e-4, epoch: int = 0):
        """[summary]
            Use Monte Carlo Tree Search algorithm with DeepLearning.
        Args:
            encoder ([type]): [description]
            model ([type]): [description]
            device ([type]): [description]
            c (float, optional): [description]. Defaults to 5.0.
            simulations_per_move (int, optional): [description]. Defaults to 300.
            num_threads_per_round (int, optional): [description]. Defaults to 12.
            noise (bool, optional): [description]. Defaults to True.
            lr (float, optional): [description]. Defaults to 1e-3.
        """
        super().__init__()
        self.encoder = encoder
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.alpha = 0.2
        self.collector = None

        self.num_time_steps = 1000
        self.input_size = input_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_sampling_steps = num_sampling_steps

        self.epoch = epoch
        
        # vae
        self.vae = VAE(in_channels=2, latent_size=256)
        self.vae_lr = lr
        self.vae_optimizer = optim.AdamW(self.vae.parameters(), lr=self.vae_lr)
        self.vae_scheduler = optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=150, gamma=0.5)

        # diffusion
        self.betas = self.linear_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, device=device)
        self.lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.5)
        
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_time_steps, dtype=torch.float32)

    def _get_std(self, alpha_bar, alpha_bar_prev):
        """
        σ_t = sqrt((1 - α_t-1)/(1 - α_t)) * sqrt(1 - α_t/α_t-1)
        """
        return torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        
    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
                    
    def set_collector(self, collector: ExperienceCollector) -> None:
        self.collector = collector

    def add_num_simulated_games(self, num: int) -> None:
        self.num_simulated_games += num

    @classmethod
    def get_value(cls, winner: Player) -> float:
        # 무승부인 경우
        if winner == Player.both:
            return 0.
        else:
            return 1.
        
    def sample(self, game_state, n, num_sampling_steps, eta=0., clip_sample=False):
        self.model.eval()
        
        with torch.no_grad():
            # vae
            # encode state
            encoded_state = self.encoder.encode(game_state)
            # vae_input = torch.tensor(encoded_state, dtype=torch.float, device=self.device)
            vae_input = torch.tensor(encoded_state[0:2][np.newaxis], dtype=torch.float, device=self.device)
            z, mu, log_var = self.vae.encode(vae_input)
            
            # diffusion
            x = torch.randn((n, 1, self.input_size, self.input_size), device=self.device)
            player_val = 1 if game_state.player == Player.black else 0
            cur_player_input = torch.tensor([[player_val]] * n, dtype=torch.float, device=self.device)
            sampling_steps = list(reversed(range(0, self.num_time_steps, self.num_time_steps // self.num_sampling_steps)))
            for idx, t in enumerate(sampling_steps):
                predicted_noise = self.model(x, (torch.ones(n) * t).long().to(self.device), z, cur_player_input)
                alpha_bar = self.alphas_bar[t]
                
                # prev_t가 0보다 작아지는 경우 alpha_bar_prev = 1로 설정
                if idx + 1 < len(sampling_steps):
                    prev_t = sampling_steps[idx + 1]
                    alpha_bar_prev = self.alphas_bar[prev_t]
                else:
                    alpha_bar_prev = self.final_alpha_cumprod
                    
                std = eta * self._get_std(alpha_bar, alpha_bar_prev)
                predicted_x0 = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                direction_pointing_to_xt = torch.sqrt(1 - alpha_bar_prev - std ** 2) * predicted_noise
                
                if clip_sample:
                    predicted_x0 = torch.clamp(predicted_x0, -1, 1)
                    
                x = torch.sqrt(alpha_bar_prev) * predicted_x0 + direction_pointing_to_xt

                # Add noise
                if eta > 0:
                    if t > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x += std * noise
                    
        x = F.softmax(x, dim=2)
        # x = (x.clamp(-1, 1) + 1) / 2
        
        return x

    def select_move(self, game_state: AbstractGameState) -> Tuple[Move, Optional[np.ndarray]]:
        
        prior = self.sample(game_state,
                            n=1,
                            num_sampling_steps=10).cpu()
        
        move_priors = np.where(game_state.get_board().get_grid() == 0, prior, 0.0)
        
        move_priors = rearrange(move_priors, 'b n h w -> b n (h w)')
        
        if self.collector is not None:
            self.collector.record_decision(self.encoder.encode(game_state), move_priors)
        
        selected_move_idx = np.argmax(move_priors)
        return self.encoder.decode_move_index(selected_move_idx), move_priors

    def train(self, dataset: ExperienceDataset, batch_size: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                shuffle=False, pin_memory=False)
        
        # train vae
        self.vae.train()
        
        def vae_kl_loss(mu, log_var):
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)
            return torch.mean(kl_loss)
        
        r_criterion = nn.MSELoss()
        r_loss_factor = 10000
        
        vae_loss = 0.0
        r_loss = 0.0
        kl_loss = 0.0
        
        for states, _, _ in tqdm(dataloader):
            states = states[:, 0:2].to(self.device).float()
            
            with torch.set_grad_enabled(True):
                outputs, mu, log_var = self.vae(states)
                print(outputs.shape)
                print(states.shape)
                r_loss = r_loss_factor * r_criterion(outputs, states)
                kl_loss = vae_kl_loss(mu, log_var)
                vae_loss = r_loss + kl_loss
                vae_loss.backward()
                self.vae_optimizer.step()
                
        self.vae.eval()
            
        # train diffusion
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                shuffle=False, pin_memory=False)
        policy_loss_sum = 0
        value_loss_sum = 0
        epoch_loss_sum = 0
        dataset_size = len(dataset)

        for state, p, reward in tqdm(dataloader):
            state = state.to(self.device).float()
            p = p.to(self.device).float()
            reward = reward.to(self.device).float()
            reward = reward.unsqueeze(1)

            with torch.set_grad_enabled(True):
                output = self.model(state)
                # categorical crossentropy
                # output의 결과는 softmax
                policy_loss = -1 * torch.mean(torch.sum(p * output[0].log(), dim=1))
                value_loss = F.mse_loss(output[1], reward)

                epoch_loss = policy_loss + value_loss

                policy_loss_sum += policy_loss.item() * state.size(0)
                value_loss_sum += value_loss.item() * state.size(0)
                epoch_loss_sum += epoch_loss.item() * state.size(0)
                
                self.optimizer.zero_grad()
                epoch_loss.backward()
                self.optimizer.step()
                
            # del state
            # del p
            # del reward

        self.scheduler.step()

        policy_loss = policy_loss_sum / dataset_size
        value_loss = value_loss_sum / dataset_size
        loss = policy_loss + value_loss

        self.model.eval()
        
        del dataloader

        return loss, policy_loss, value_loss

    def save_agent(self, pthfile: str) -> None:
        state = {
            'encoder': self.encoder,
            'model': self.model,
            'num_simulated_games': self.num_simulated_games,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        torch.save(state, pthfile)

    @staticmethod
    def load_agent(pthfilename: str, device: str, simulations_per_move: int = 500,
                   num_threads_per_round: int = 1, noise: bool = False) -> SelfZeroAgent:
        loaded_file = torch.load(pthfilename, map_location=device)
        encoder = loaded_file['encoder']
        model = loaded_file['model']
        num_simulated_games = loaded_file['num_simulated_games']
        epoch = loaded_file['epoch']
        optimizer_state_dict = loaded_file['optimizer']
        scheduler_state_dict = loaded_file['scheduler']

        loaded_agent = DiffusionAgent(encoder, model, device,
                                      simulations_per_move=simulations_per_move,
                                      num_threads_per_round=num_threads_per_round,
                                      noise=noise, epoch=epoch, num_simulated_games=num_simulated_games)
        
        loaded_agent.optimizer.load_state_dict(optimizer_state_dict)
        loaded_agent.scheduler.load_state_dict(scheduler_state_dict)
        print("simulated %d games." % num_simulated_games)
        return loaded_agent
