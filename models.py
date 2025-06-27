import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class LSTM_QNet(nn.Module):
    def __init__(
        self,
        window: int = 16,
        feature_dim: int = 5,
        lstm_hidden: int = 32,
        fc_hidden: int = 64,
        action_dim: int = 3,
    ):
        super().__init__()
        self.window = window
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # LSTM over (window × feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=lstm_hidden, batch_first=True
        )
        # small FC on top of LSTM
        self.fc_after_lstm = nn.Linear(lstm_hidden, lstm_hidden)
        # two FC layers after concatenating prev-pos one-hot
        self.fc1 = nn.Linear(lstm_hidden + action_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        # final head: Q-values for each action
        self.q_head = nn.Linear(fc_hidden, action_dim)
        self.reset_parameters()

    def get_q_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, window*feature_dim + 1)
        returns Q(s,·): (B, action_dim)
        """
        B = state.size(0)
        hist = state[:, : self.window * self.feature_dim]
        prev = state[:, -1].long()  # δ ∈ {-1,0,1}
        seq = hist.view(B, self.window, self.feature_dim)

        lstm_out, _ = self.lstm(seq)  # (B, window, lstm_hidden)
        h_T = lstm_out[:, -1, :]  # (B, lstm_hidden)
        h = F.relu(self.fc_after_lstm(h_T))

        oh = F.one_hot(prev + 1, self.action_dim).float()  # (B,3)
        z = torch.cat([h, oh], dim=1)  # (B, lstm+3)
        z1 = F.relu(self.fc1(z))
        z2 = F.relu(self.fc2(z1))
        q = self.q_head(z2)  # (B,3)
        return q

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.get_q_value(state)

    def init_q_head(self, head: nn.Linear):
        nn.init.uniform_(head.weight, -1e-3, 1e-3)
        nn.init.zeros_(head.bias)

    def init_linear_relu(self, layer: nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

    def init_lstm(self, lstm: nn.LSTM):
        for name, param in lstm.named_parameters():
            if "weight_ih" in name:  # input → hidden
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:  # hidden → hidden
                nn.init.orthogonal_(param)
            elif "bias" in name:
                param.data.fill_(0.0)
                # PyTorch concatenates [i, f, g, o] in that order
                n = param.size(0)
                start, end = n // 4, n // 2  # f-gate slice
                param.data[start:end] = 1.0  # forget-gate bias = +1

    def reset_parameters(self):
        self.init_lstm(self.lstm)
        self.init_linear_relu(self.fc_after_lstm)
        self.init_linear_relu(self.fc1)
        self.init_linear_relu(self.fc2)
        self.init_q_head(self.q_head)

    @torch.no_grad()
    def act_greedy(self, state: torch.Tensor) -> torch.Tensor:
        q = self(state)
        return q.argmax(dim=1, keepdim=True)


class PPOActorCritic(nn.Module):
    def __init__(
        self,
        window: int = 16,
        feature_dim: int = 5,
        lstm_hidden: int = 128,
        fc_hidden: int = 64,
    ):
        super().__init__()
        self.window = window
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=lstm_hidden, batch_first=True
        )
        self.fc_after_lstm = nn.Linear(lstm_hidden, lstm_hidden)
        self.fc1 = nn.Linear(lstm_hidden + 3, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)

        self.trail_head = nn.Linear(fc_hidden, 3)
        self.trade_head = nn.Linear(fc_hidden, 3)
        self.value_head = nn.Linear(fc_hidden, 1)
        self.reset_parameters()

    def forward(self, state: torch.Tensor):
        B = state.size(0)
        hist = state[:, : self.window * self.feature_dim]
        prev = state[:, -1].long()
        seq = hist.view(B, self.window, self.feature_dim)

        lstm_out, _ = self.lstm(seq)
        h_T = lstm_out[:, -1, :]
        h = F.relu(self.fc_after_lstm(h_T))

        oh = F.one_hot(prev + 1, num_classes=3).float()

        z = torch.cat([h, oh], dim=1)
        z1 = F.relu(self.fc1(z))
        z2 = F.relu(self.fc2(z1))

        trail_logits = self.trail_head(z2)
        trade_logits = self.trade_head(z2)
        value = self.value_head(z2).squeeze(-1)
        return trail_logits, trade_logits, value

    def policy(self, state: torch.Tensor):
        trail_logits, trade_logits, _ = self.forward(state)
        return trail_logits, trade_logits

    def value(self, state: torch.Tensor):
        _, _, v = self.forward(state)
        return v

    def act(self, state: torch.Tensor):
        trail_logits, trade_logits, value = self.forward(state)
        d1 = Categorical(logits=trail_logits)
        d2 = Categorical(logits=trade_logits)
        a1 = d1.sample()
        a2 = d2.sample()
        logp = d1.log_prob(a1) + d2.log_prob(a2)
        return a1, a2, logp, value

    def act_greedy(self, state: torch.Tensor):
        trail_logits, _, _ = self.forward(state)
        return torch.argmax(trail_logits, dim=-1)

    def shared_parameters(self):
        return (
            list(self.lstm.parameters())
            + list(self.fc_after_lstm.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
        )

    def policy_parameters(self):
        return (
            self.shared_parameters()
            + list(self.trail_head.parameters())
            + list(self.trade_head.parameters())
        )

    def value_parameters(self):
        return self.shared_parameters() + list(self.value_head.parameters())

    def init_lstm(self, lstm: nn.LSTM):
        for name, p in lstm.named_parameters():
            if "weight_ih" in name:  # input → hidden
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:  # hidden → hidden
                nn.init.orthogonal_(p)
            elif "bias" in name:
                p.data.fill_(0.0)
                # forget gate is second quarter in PyTorch ordering [i, f, g, o]
                n = p.size(0)
                p.data[n // 4 : n // 2] = 1.0

    def init_linear_relu(self, layer: nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)

    def init_head_small(self, layer: nn.Linear):
        nn.init.uniform_(layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(layer.bias)

    def reset_parameters(self):
        self.init_lstm(self.lstm)
        self.init_linear_relu(self.fc_after_lstm)
        self.init_linear_relu(self.fc1)
        self.init_linear_relu(self.fc2)
        self.init_head_small(self.trail_head)
        self.init_head_small(self.trade_head)
        self.init_head_small(self.value_head)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s2),
            torch.FloatTensor(1 - d),
        )

    def __len__(self):
        return len(self.buffer)


class PPOBuffer:
    def __init__(self, gamma: float = 0.995, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.clear()

    def clear(self):
        self.obs, self.trail_a, self.trade_a = [], [], []
        self.logp, self.rews, self.vals, self.dones = [], [], [], []

    def store(self, obs, a1, a2, logp, rew, val, done):
        self.obs.append(obs)
        self.trail_a.append(a1)
        self.trade_a.append(a2)
        self.logp.append(logp)
        self.rews.append(rew)
        self.vals.append(val)
        self.dones.append(done)

    def finish_path(self, last_val: float = 0.0):
        obs = torch.stack(self.obs)
        trail_a = torch.stack(self.trail_a)
        trade_a = torch.stack(self.trade_a)
        logp_old = torch.stack(self.logp)
        vals = np.array(self.vals + [last_val])

        adv = []
        gae = 0.0
        for t in reversed(range(len(self.rews))):
            non_terminal = 1.0 - self.dones[t]
            delta = self.rews[t] + self.gamma * vals[t + 1] * non_terminal - vals[t]
            gae = delta + self.gamma * self.lam * gae * non_terminal
            adv.insert(0, gae)

        adv = torch.tensor(adv, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + torch.tensor(self.vals, dtype=torch.float32)
        self.clear()
        return obs, trail_a, trade_a, logp_old, ret, adv
