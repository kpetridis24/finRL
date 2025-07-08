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

# ddqn_writer = SummaryWriter("runs/price_trail_experiment/ddqn")
# torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on {device}.")
checkpoint_dir = "checkpoints/ddqn"


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
        df.groupby(tic_col, group_keys=False)
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
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, float]:
    pnls = []
    for _ in range(repeats):
        env.reset()
        env.set_fixed_window(env.start_idx, env.EP_LEN)
        done = False
        trunc = False
        obs, _ = env.reset()
        pnl_series = []

        while not (done or trunc):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = net.act_greedy(obs_tensor).item()
            obs, _, done, trunc, info = env.step(int(action))
            pnl_series.append(info.get("pnl", 0.0))
        pnls.append(np.sum(pnl_series))

    returns = np.array(pnls)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24)
    mdd = max_drawdown(returns)
    mean_ret = returns.mean()

    # dash_writer.add_scalar("eval/Sharpe", sharpe, global_step)
    # dash_writer.add_scalar("eval/MeanReturn", mean_ret, global_step)
    # dash_writer.add_scalar("eval/MaxDrawdown", mdd, global_step)
    return sharpe, returns.mean(), mdd


def train_ddqn(
    env_train: ForexPriceTrailingEnv,
    env_eval: ForexPriceTrailingEnv,
    q_net: nn.Module,
    target_q: nn.Module,
    buffer,
    episodes: int = 5000,
    eval_every: int = 50,
    checkpoint_every: int = 100,
    patience: int = 8,
    min_delta: float = 0.01,
    batch_size: int = 512,
    gamma: float = 0.995,
    lr: float = 1e-4,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 25000,
    target_update: int = 10,
    device: torch.device = torch.device("cpu"),
    writer: SummaryWriter | None = None,
):
    optim_q = optim.Adam(q_net.parameters(), lr=lr)
    steps_done = 0
    best_sharpe = -np.inf
    wait = 0
    best_state = deepcopy(q_net.state_dict())

    target_q.load_state_dict(q_net.state_dict())  # ← add this line
    target_q.eval()

    for ep in range(1, episodes + 1):
        obs, _ = env_train.reset()
        ep_reward = 0.0

        for t in range(env_train.EP_LEN):
            eps = eps_end + (eps_start - eps_end) * np.exp(
                -1.0 * steps_done / eps_decay
            )
            writer.add_scalar("epsilon", eps, (ep - 1) * env_train.EP_LEN + t)
            steps_done += 1

            if random.random() > eps:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
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
                s_batch, a_batch, r_batch, s2_batch, not_done = buffer.sample(
                    batch_size
                )
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
                    q_target = (
                        r_batch.unsqueeze(1) + gamma * not_done.unsqueeze(1) * q2_a
                    )

                loss = F.smooth_l1_loss(q_a, q_target)
                optim_q.zero_grad()
                loss.backward()
                optim_q.step()

            if done or trunc:
                break

        if ep % target_update == 0:
            target_q.load_state_dict(q_net.state_dict())

        if ep % checkpoint_every == 0:
            print("Checkpointing...")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = (
                f"ddqn_ep{ep}_steps{steps_done}_sharpe{sharpe:.3f}_"
                f"lr{lr}_bs{batch_size}_gamma{gamma}_epsdecay{eps_decay}.pth"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(
                {
                    "model_state_dict": q_net.state_dict(),
                    "episode": ep,
                    "steps_done": steps_done,
                    "sharpe": sharpe,
                    "avg_pnl": avg_pnl,
                    "mdd": mdd,
                    "lr": lr,
                    "batch_size": batch_size,
                    "gamma": gamma,
                    "eps_decay": eps_decay,
                },
                checkpoint_path,
            )

        if ep % eval_every == 0:
            sharpe, avg_pnl, mdd = evaluate_greedy(
                q_net, env_eval, 5, steps_done, device=device
            )
            print(
                f"Eval @ Ep {ep}: Sharpe={sharpe:.3f}, PnL={avg_pnl:.2f}, MDD={mdd:.2f}"
            )
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


def train_ddqn_vectorized(
    env_train,  # Sync/AsyncVectorEnv with N envs
    env_eval,  # another vector env (can be N_eval)
    q_net: nn.Module,
    target_q: nn.Module,
    buffer,  # ReplayBuffer
    *,  # force keywords after this
    episodes: int = 5000,
    eval_every: int = 50,
    checkpoint_every: int = 100,
    patience: int = 8,
    min_delta: float = 0.01,
    batch_size: int = 512,
    gamma: float = 0.995,
    lr: float = 1e-4,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 25_000,
    target_update: int = 10,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints/ddqn_vec",
    writer: SummaryWriter | None = None,
):
    q_net = q_net.to(device)
    target_q = target_q.to(device)
    optim_q = optim.Adam(q_net.parameters(), lr=lr)
    next_checkpoint = checkpoint_every

    eps = eps_start
    target_q.load_state_dict(q_net.state_dict())
    target_q.eval()

    n_envs = env_train.num_envs
    total_episodes = 0  # across *all* envs
    steps_done = 0
    steps_done_delayed = 0
    # best_sharpe = -np.inf
    # wait = 0
    best_state = q_net.state_dict()
    # decay_frames = 10_000

    obs, _ = env_train.reset()  # (N, obs_dim)
    # warmup_action_probs = np.array([0.35, 0.3, 0.35])
    # fixed_random_until = 100 * n_envs
    # warmup_steps = fixed_random_until + 1000 * n_envs
    # effective_decay = eps_decay / n_envs
    # slow_factor = 350.0
    # decay_const = effective_decay * slow_factor

    # f1, f2, f3 = True, True, True
    start_decreasing = False
    MIN_BUFFER = 4 * batch_size

    while total_episodes < episodes:
        frames_per_env = steps_done // n_envs
        eps = eps_end + (eps_start - eps_end) * np.exp(-frames_per_env / eps_decay)
        writer.add_scalar("epsilon", eps, steps_done)

        if steps_done % (300 * n_envs) == 0:
            print(f"Steps done: {steps_done}")

        state_t = torch.tensor(obs, dtype=torch.float32, device=device)

        if len(buffer) < MIN_BUFFER:
            a_space = env_train.single_action_space
            actions = np.random.randint(0, a_space.n, size=n_envs)
        elif np.random.rand() > eps:
            with torch.no_grad():
                actions = q_net(state_t).argmax(dim=1).cpu().numpy()
        else:
            a_space = env_train.single_action_space
            actions = np.random.randint(0, a_space.n, size=n_envs)

        # ── environment step ───────────────────────────────────────────────────
        next_obs, rewards, dones, truncs, infos = env_train.step(actions)
        done_flags = np.logical_or(dones, truncs)

        # store transitions in the *shared* replay-buffer
        for i in range(n_envs):
            buffer.push(
                obs[i],
                int(actions[i]),
                float(rewards[i]),
                next_obs[i],
                bool(done_flags[i]),
            )

        steps_done += n_envs
        if start_decreasing:
            steps_done_delayed += n_envs
        obs = next_obs

        # count completed episodes so far
        total_episodes += done_flags.sum()

        if len(buffer) >= batch_size:
            s, a, r, s2, not_done = buffer.sample(batch_size)
            s, a, r, s2, not_done = [t.to(device) for t in (s, a, r, s2, not_done)]

            q = q_net(s).gather(1, a.unsqueeze(1))
            with torch.no_grad():
                next_a = q_net(s2).argmax(dim=1, keepdim=True)
                q2 = target_q(s2).gather(1, next_a)
                target = r.unsqueeze(1) + gamma * not_done.unsqueeze(1) * q2

            loss = F.smooth_l1_loss(q, target)
            optim_q.zero_grad()
            loss.backward()
            optim_q.step()

        # update target network
        if frames_per_env % target_update == 0:  # target_update = 10
            target_q.load_state_dict(q_net.state_dict())
        # if steps_done % (target_update * n_envs) == 0:
        #     target_q.load_state_dict(q_net.state_dict())

        # # ── evaluation / early-stopping / checkpoint ──────────────────────────
        # if total_episodes % eval_every == 0:
        #     sharpe, *_ = evaluate_greedy(
        #         q_net, env_eval.envs[0], repeats=5, device=device
        #     )
        #     print(f"[DDQN] eps={total_episodes:>5}  Sharpe={sharpe:6.3f}")

        # if sharpe > best_sharpe + min_delta:
        #     best_sharpe, wait = sharpe, 0
        #     best_state = q_net.state_dict()
        # else:
        #     wait += 1
        # if wait >= patience:
        #     print("Early-stopping triggered.")
        #     break

        if steps_done >= next_checkpoint:
            print("Checkpointing...")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                q_net.state_dict(),
                os.path.join(checkpoint_dir, f"qnet_step{steps_done}.pth"),
            )
            next_checkpoint += checkpoint_every

    print(f"Total training steps: {steps_done}.")
    q_net.load_state_dict(best_state)
    return q_net
