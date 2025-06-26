from datetime import datetime, date
import pandas as pd
import itertools
from enum import StrEnum, auto
import numpy as np
from torch import optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import ray
from ray import train
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from ray.train import Checkpoint
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_forex_price_trailing import ForexPriceTrailingEnv
import torch as th
import torch
import torch.nn as nn
from collections import deque
import random
import os
import torch.nn.functional as F
import torch.nn as nns
from copy import deepcopy
from models import LSTM_QNet, ReplayBuffer

dash_writer = SummaryWriter("runs/price_trail_experiment/ddqn")
torch.cuda.set_device(0)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ngpus = torch.cuda.device_count()
print(f"Running on {device}.")

class Currency(StrEnum):
    USD = auto()
    EUR = auto()
    JPY = auto()
    GBP = auto()
    AUD = auto()
    CAD = auto()
    CHF = auto()
    NZD = auto()
    CNY = auto()

    
def max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    return drawdowns.min()


def add_fx_features_for_tick(g: pd.DataFrame) -> pd.DataFrame:
    g["close_prev"] = g.close.shift(1)
    g["high_prev"] = g.high.shift(1)
    g["low_prev"] = g.low.shift(1)

    g["x1"] = (g.close - g.close_prev) / g.close_prev
    g["x2"] = (g.high - g.high_prev) / g.high_prev
    g["x3"] = (g.low - g.low_prev) / g.low_prev
    g["x4"] = (g.high - g.close) / g.close
    g["x5"] = (g.close - g.low) / g.close
    return g


def add_fx_features(df: pd.DataFrame, tic_col: str = "tic") -> pd.DataFrame:
    df_with_features = (
        df
        .groupby(tic_col, group_keys=False)
        .apply(add_fx_features_for_tick)
        .drop(columns=["close_prev", "high_prev", "low_prev"])
        .fillna(0)
    )
    return df_with_features


def evaluate_greedy(
    net: nn.Module, 
    env: ForexPriceTrailingEnv, 
    repeats: int = 5,
    global_step: int = 0,
    device: torch.device = torch.device("cpu")
) -> tuple[float, float, float]:
    pnls = []
    # all_step_returns = []
    # last_pnl_series = None

    for _ in range(repeats):
        env.reset()
        env.set_fixed_window(env.start_idx, env.EP_LEN)
        done = False
        trunc = False
        obs, _ = env.reset()
        pnl_series = []

        while not (done or trunc):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = net.act_greedy(obs_tensor).item()
            obs, _, done, trunc, info = env.step(int(action))
            pnl_series.append(info.get("pnl", 0.0))
        pnls.append(np.sum(pnl_series))
        # last_pnl_series = pnl_series  

    returns = np.array(pnls)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24)
    mdd = max_drawdown(returns)
    mean_ret = returns.mean()

    dash_writer.add_scalar("eval/Sharpe", sharpe, global_step)
    dash_writer.add_scalar("eval/MeanReturn", mean_ret, global_step)
    dash_writer.add_scalar("eval/MaxDrawdown", mdd, global_step)
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # # a) Cumulative PnL for last episode
    # cum = np.cumsum(last_pnl_series)
    # axes[0,0].plot(cum)
    # axes[0,0].set_title("Last Episode: Cumulative PnL")
    # axes[0,0].set_xlabel("Step")
    # axes[0,0].set_ylabel("Cum. PnL")

    # # b) Drawdown curve
    # peak = np.maximum.accumulate(cum)
    # dd = (cum - peak) / (peak + 1e-9)
    # axes[0,1].plot(dd)
    # axes[0,1].set_title("Last Episode: Drawdown")
    # axes[0,1].set_xlabel("Step")
    # axes[0,1].set_ylabel("Drawdown")

    # # c) Histogram of total PnLs
    # axes[1,0].hist(returns, bins=20, edgecolor='k')
    # axes[1,0].set_title("Distribution of Total PnL (repeats)")
    # axes[1,0].set_xlabel("Total PnL")
    # axes[1,0].set_ylabel("Frequency")

    # # d) Boxplot of all step-returns
    # axes[1,1].boxplot(all_step_returns, vert=False)
    # axes[1,1].set_title("Boxplot of Step Returns")
    # axes[1,1].set_xlabel("Step Return")

    # plt.tight_layout()

    # # 3) Push that grid into TB
    # # dash_writer.add_figure(f"eval_step_{global_step}/EvaluationGrid", fig)
    # plt.close(fig)
    return sharpe, returns.mean(), mdd


def train_ddqn(
    env_train: ForexPriceTrailingEnv,
    env_eval: ForexPriceTrailingEnv,
    q_net: nn.Module,
    target_q: nn.Module,
    buffer,
    episodes: int = 5000,
    eval_every: int = 50,
    patience: int = 8,
    min_delta: float = 0.01,
    batch_size: int = 512,
    gamma: float = 0.995,
    lr: float = 1e-4,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 25000,
    target_update: int = 10,
    device: torch.device = torch.device("cpu")
):
    optim_q = optim.Adam(q_net.parameters(), lr=lr)
    steps_done = 0
    best_sharpe = -np.inf
    wait = 0
    best_state = deepcopy(q_net.state_dict())

    target_q.load_state_dict(q_net.state_dict())   # ← add this line
    target_q.eval()   

    for ep in range(1, episodes + 1):
        obs, _ = env_train.reset()
        ep_reward = 0.0

        for t in range(env_train.EP_LEN):
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
            dash_writer.add_scalar("epsilon", eps, (ep - 1) * env_train.EP_LEN + t)
            steps_done += 1

            if random.random() > eps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    qs = q_net(obs_tensor)
                    action = qs.argmax(dim=1, keepdim=True).item()
            else:
                action = env_train.action_space.sample()
            
            next_obs, reward, done, trunc, info = env_train.step(int(action))
            ep_reward += reward
            buffer.push(obs, action, reward, next_obs, done or trunc)
            obs = next_obs

            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, not_done = buffer.sample(batch_size)
                s_batch, a_batch, r_batch, s2_batch, not_done = (
                    s_batch.to(device),
                    a_batch.to(device),
                    r_batch.to(device),
                    s2_batch.to(device),
                    not_done.to(device),
                )

                q_vals = q_net(s_batch)
                q_a = q_vals.gather(1, a_batch.unsqueeze(1))

                with torch.no_grad():
                    next_actions = q_net(s2_batch).argmax(dim=1, keepdim=True)
                    q2 = target_q(s2_batch)
                    q2_a = q2.gather(1, next_actions)
                    q_target = r_batch.unsqueeze(1) + gamma * not_done.unsqueeze(1) * q2_a

                loss = F.smooth_l1_loss(q_a, q_target)
                optim_q.zero_grad()
                loss.backward()
                optim_q.step()

            if done or trunc:
                break

        if ep % target_update == 0:
            target_q.load_state_dict(q_net.state_dict())
        
        if ep % eval_every == 0:
            sharpe, avg_pnl, mdd = evaluate_greedy(q_net, env_eval, 5, steps_done, device=device)
            print(f"Eval @ Ep {ep}: Sharpe={sharpe:.3f}, PnL={avg_pnl:.2f}, MDD={mdd:.2f}")
            if sharpe > best_sharpe + min_delta:
                best_sharpe = sharpe
                wait = 0
                best_state = deepcopy(q_net.state_dict())
            else:
                wait += 1
            if wait >= patience:
                print(f"Early stopping at ep {ep}, best Sharpe={best_sharpe:.3f}")
                q_net.load_state_dict(best_state)
                break
    return q_net


majors = [
    "EURUSD=X","USDJPY=X","GBPUSD=X",
    "AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X"
]

# 2) Top non-USD crosses
crosses = [
    "EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X",
    "CADJPY=X","EURAUD=X","EURCAD=X","EURCHF=X",
    "GBPCHF=X","AUDCAD=X","NZDJPY=X","NZDCAD=X"
]

# 3) Key CNY pairs
cny_pairs = [
    "USDCNY=X","EURCNY=X","JPY CNY=X".replace(" ",""),  # -> "JPYCNY=X"
    "GBPCNY=X","AUDCNY=X","CADCNY=X","CHFCNY=X","NZDCNY=X"
]

# 4) Stitch together, then take the first 20 unique
all_tickers = majors + crosses + cny_pairs
# remove any duplicates and slice to 20
seen = set()
forex_ticks: tuple[str, ...] = tuple(
    t for t in all_tickers
    if not (t in seen or seen.add(t))
)

start_train = date(2000,1,1)
end_train = date(2017,1,1)
eval_span = (date(2016,7,1), date(2017,1,1))  # last 6 months
test_span = (date(2017,1,1), date(2018,6,1))

yfd = YahooDownloader(start_date=str(start_train), end_date=str(end_train), ticker_list=forex_ticks)
df_train = add_fx_features(yfd.fetch_data())

yfd2 = YahooDownloader(start_date=str(test_span[0]), end_date=str(test_span[1]), ticker_list=forex_ticks)
df_test = add_fx_features(yfd2.fetch_data())

sample_df = df_train[df_train.tic == forex_ticks[0]].reset_index(drop=True)
val_start_idx = sample_df[pd.to_datetime(sample_df.date) == pd.Timestamp(eval_span[0])].index[0]
val_length = int((eval_span[1] - eval_span[0]).days)


window = 16
env_context = {
    "tic_col": "tic",
    "window": 16,
    "margin": 0.02,
    "step_frac": 0.1,
    "fee": 2e-4,
    "alpha_trail": 0.97,
    "alpha_pnl": 0.85,
    "alpha_fee": 1.0,
    "episode_len": 600,
    "pick_new_pair_every": 1,
}
train_env = ForexPriceTrailingEnv(df_train, **env_context)
eval_env_context = env_context
eval_env_context["episode_len"] = 150
eval_env   = ForexPriceTrailingEnv(df_train, **eval_env_context)
eval_env.set_fixed_window(val_start_idx, val_length)
test_env = ForexPriceTrailingEnv(df_test, window=window)

q_net = LSTM_QNet(
    window=window,
    feature_dim=5,
    lstm_hidden=128,
    fc_hidden=64,
    action_dim=3
)
target_q = deepcopy(q_net)
buffer = ReplayBuffer(capacity=200_000)

q_net = q_net.to(device)
target_q = target_q.to(device)

# if ngpus > 1:                        
#     q_net = torch.nn.DataParallel(q_net)     
#     target_q = torch.nn.DataParallel(target_q)

best_model = train_ddqn(
    train_env,
    eval_env,
    q_net,
    target_q,
    buffer,
    episodes=2000,
    eval_every=50,
    patience=8,
    batch_size=512,
    eps_decay=25000,
    device=device
)

torch.save(best_model.state_dict(), "ddqn_forex_trained_model.pth")