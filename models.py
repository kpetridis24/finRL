import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
import torch.nn.functional as F


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
    
    @torch.no_grad()
    def act_greedy(self, state: torch.Tensor) -> torch.Tensor:
        q = self(state)
        return q.argmax(dim=1, keepdim=True)


# ─── 3) REPLAY BUFFER ───────────────────────────────────────────────────────────


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
