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
from torch.distributions import Categorical
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
from models import PPOBuffer, PPOActorCritic


# ppo_writer = SummaryWriter("runs/price_trail_experiment/ppo")
# torch.cuda.set_device(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on {device}.")
checkpoint_dir = "checkpoints/ppo"


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


def eval_agent(
    env: ForexPriceTrailingEnv,
    model: nn.Module,
    device: torch.device = torch.device("cpu"),
):
    sharpes = []
    for _ in range(5):
        ov, _ = env.reset()
        env.set_fixed_window(env.start_idx, env.EP_LEN)
        done = trunc = False
        pnls = []
        while not (done or trunc):
            with torch.no_grad():
                a = model.act_greedy(
                    torch.tensor(ov, dtype=torch.float32, device=device).unsqueeze(0)
                )
            ov, _, done, trunc, info = env.step(int(a.cpu()))
            pnls.append(info.get("pnl", 0.0))
        arr = np.array(pnls)
        sharpes.append(arr.mean() / (arr.std() + 1e-9) * np.sqrt(365))
    return np.mean(sharpes)


def train_ppo(
    env_train: ForexPriceTrailingEnv,
    env_eval: ForexPriceTrailingEnv,
    model: nn.Module,
    episodes: int = 500,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_pi_iters: int = 80,
    train_v_iters: int = 80,
    max_grad_norm: float = 0.5,
    eval_every: int = 50,
    checkpoint_every: int = 100,
    patience: int = 8,
    device: torch.device = torch.device("cpu"),
):
    model = model.to(device)
    buffer = PPOBuffer(gamma, lam)
    pi_params = list(model.policy_parameters())
    vf_params = list(model.value_parameters())

    pi_optimizer = optim.Adam(pi_params, lr=pi_lr)
    vf_optimizer = optim.Adam(vf_params, lr=vf_lr)

    best_sharpe = -np.inf
    wait = 0
    best_state = deepcopy(model.state_dict())

    o, _ = env_train.reset()
    ep_ret, ep_len = 0.0, 0
    steps_done = 0

    for ep in range(1, episodes + 1):
        for t in range(env_train.EP_LEN):
            steps_done += 1
            state = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_trail, a_trade, logp, val = model.act(state)

            a_trail_cpu, a_trade_cpu, logp_cpu, val_cpu = (
                a_trail.cpu(),
                a_trade.cpu(),
                logp.cpu(),
                val.cpu(),
            )

            nxt, r, done, trunc, info = env_train.step(int(a_trail_cpu))
            ep_ret += r
            ep_len += 1

            buffer.store(
                torch.tensor(o, dtype=torch.float32),
                a_trail_cpu,
                a_trade_cpu,
                logp_cpu,
                float(r),
                val_cpu.item(),
                done or trunc,
            )

            o = nxt
            terminal = done or trunc

            if terminal or (t == env_train.EP_LEN - 1):
                with torch.no_grad():
                    if terminal:
                        last_val = 0.0
                    else:
                        last_val = (
                            model.value(
                                torch.tensor(o, dtype=torch.float32, device=device)
                            )
                            .unsqueeze(0)
                            .item()
                        )

                obs_b, a1_b, a2_b, logp_b, ret_b, adv_b = buffer.finish_path(last_val)
                o, _ = env_train.reset()
                ep_ret, ep_len = 0.0, 0

                obs_b, logp_b, ret_b, adv_b = (
                    obs_b.to(device),
                    logp_b.to(device),
                    ret_b.to(device),
                    adv_b.to(device),
                )
                a1_b, a2_b = a1_b.to(device), a2_b.to(device)

                for _ in range(train_pi_iters):
                    pi_optimizer.zero_grad()
                    trail_logits, trade_logits = model.policy(obs_b)
                    dist1 = Categorical(logits=trail_logits)
                    dist2 = Categorical(logits=trade_logits)
                    logp = dist1.log_prob(a1_b) + dist2.log_prob(a2_b)
                    ratio = torch.exp(logp - logp_b)
                    clip_adv = (
                        torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
                    )
                    loss_pi = -(torch.min(ratio * adv_b, clip_adv)).mean()
                    loss_pi.backward()
                    torch.nn.utils.clip_grad_norm_(pi_params, max_grad_norm)
                    pi_optimizer.step()

                for _ in range(train_v_iters):
                    vf_optimizer.zero_grad()
                    v_pred = model.value(obs_b)
                    loss_v = F.mse_loss(v_pred, ret_b)
                    loss_v.backward()
                    torch.nn.utils.clip_grad_norm_(vf_params, max_grad_norm)
                    vf_optimizer.step()

                break

        if ep % checkpoint_every == 0:
            print("Checkpointing...")
            os.makedirs(checkpoint_dir, exist_ok=True)
            curr_sharpe = best_sharpe if 'sharpe' not in locals() else sharpe
            safe = curr_sharpe if np.isfinite(curr_sharpe) else -999.0
            checkpoint_name = (
                f"ppo_ep{ep}_steps{steps_done}_sharpe{safe:.3f}_"
                f"pilr{pi_lr}_vflr{vf_lr}_gamma{gamma}_lam{lam}.pth"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                    'vf_optimizer_state_dict': vf_optimizer.state_dict(),
                    'best_sharpe': best_sharpe,
                    'episode': ep,
                },
                checkpoint_path,
            )

        if ep % eval_every == 0:
            sharpe = eval_agent(env_eval, model, device)
            print(f"PPO Eval @ Ep {ep}: Sharpe={sharpe:.3f}")
            if sharpe > best_sharpe:
                best_sharpe, wait = sharpe, 0
                best_state = deepcopy(model.state_dict())
            else:
                wait += 1
            if wait >= patience:
                print(f"Early stopping at ep {ep}, best Sharpe={best_sharpe:.3f}")
                model.load_state_dict(best_state)
                break

    model.load_state_dict(best_state)
    return model


def train_ppo_vectorized(
    env_train,  # Sync/AsyncVectorEnv
    env_eval,  # single env (your original) OR vector – both fine
    model: nn.Module,
    episodes: int = 500,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_pi_iters: int = 80,
    train_v_iters: int = 80,
    max_grad_norm: float = 0.5,
    eval_every: int = 50,
    checkpoint_every: int = 100,
    patience: int = 8,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints/ppo_vec",  # ← new folder so it won’t clash
):
    """
    Vectorised PPO that preserves ALL the hyper-parameters of the original
    implementation.  Each “episode” here means one optimisation epoch that
    collects `T = env_train.envs[0].EP_LEN` steps *from every parallel env*,
    then performs TRPO-style updates exactly as before.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    next_checkpoint = checkpoint_every
    model = model.to(device)
    opt = optim.Adam(
        [
            {"params": model.shared_parameters(), "lr": pi_lr},
            {"params": model.trail_head.parameters(), "lr": pi_lr},
            {"params": model.trade_head.parameters(), "lr": pi_lr},
            {"params": model.value_head.parameters(), "lr": vf_lr},
        ]
    )

    # --- helpers -------------------------------------------------------------
    def _stack(lst):
        return torch.stack(lst)  # just for brevity

    def _flat(t):
        return t.reshape(-1) if t.ndim == 2 else t.reshape(-1, *t.shape[2:])

    # training env meta-data
    n_envs = env_train.num_envs
    steps_per_cycle = env_train.envs[0].EP_LEN  # every env was created with this len

    # reset vector env     (obs: (N, obs_dim))
    obs, _ = env_train.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    best_sharpe, wait = -np.inf, 0
    best_state = model.state_dict()
    total_steps = 0

    for ep in range(1, episodes + 1):
        # ───────────────────  rollout phase  ────────────────────────────────
        o_buf, a1_buf, a2_buf, lp_buf, r_buf, v_buf, d_buf = [], [], [], [], [], [], []
        term_last = torch.zeros(n_envs, device=device)

        for t in range(steps_per_cycle):
            with torch.no_grad():
                a1, a2, logp, val = model.act(obs)

            # env.step needs numpy actions
            next_obs, rew, done, trunc, _ = env_train.step(a1.cpu().numpy())
            term = torch.as_tensor(done | trunc, dtype=torch.float32, device=device)

            # store
            o_buf.append(obs)
            a1_buf.append(a1)
            a2_buf.append(a2)
            lp_buf.append(logp)
            r_buf.append(torch.as_tensor(rew, device=device, dtype=torch.float32))
            v_buf.append(val)
            d_buf.append(term)

            term_last = term

            if term.any():
                done_mask = term.cpu().numpy().astype(bool)  # shape (n_envs,)

                # --- preferred (Gymnasium ≥0.29) -------------------
                if hasattr(env_train, "reset_done"):
                    env_train.reset_done(done_mask)
                # --- fallback for older vector API -----------------
                else:
                    for i, d in enumerate(done_mask):
                        if d:
                            single_obs, _ = env_train.envs[i].reset()
                            obs[i] = torch.as_tensor(
                                single_obs, dtype=torch.float32, device=device
                            )

            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            total_steps += n_envs

        # bootstrap value
        with torch.no_grad():
            last_val = model.value(obs) * (1.0 - term_last)

        # ───────────────────  advantage / return  ───────────────────────────
        # tensors with shape (T, N, …)
        o_t = _stack(o_buf)
        a1_t = _stack(a1_buf)
        a2_t = _stack(a2_buf)
        lp_t = _stack(lp_buf)
        r_t = _stack(r_buf)
        v_t = _stack(v_buf)
        d_t = _stack(d_buf)

        adv = torch.zeros_like(r_t)
        gae = torch.zeros(n_envs, device=device)
        v_next = torch.cat([v_t, last_val.unsqueeze(0)], dim=0)  # (T+1, N)

        for t in reversed(range(steps_per_cycle)):
            mask = 1.0 - d_t[t]
            delta = r_t[t] + gamma * v_next[t + 1] * mask - v_next[t]
            gae = delta + gamma * lam * gae * mask
            adv[t] = gae

        ret_t = adv + v_t

        # flatten   (T*N, …)
        o_f = o_t.reshape(-1, o_t.shape[-1])
        a1_f = _flat(a1_t)
        a2_f = _flat(a2_t)
        lp_f = _flat(lp_t)
        adv_f = _flat(adv)
        ret_f = _flat(ret_t)

        # normalise advantage
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        for _ in range(train_pi_iters):
            # 1) zero all grads
            opt.zero_grad()

            # 2) forward policy
            trail_logits, trade_logits = model.policy(o_f)   # shape (T*N, 3)
            d1 = Categorical(logits=trail_logits)
            d2 = Categorical(logits=trade_logits)
            logp = d1.log_prob(a1_f) + d2.log_prob(a2_f)

            # 3) PPO clipped surrogate
            ratio    = torch.exp(logp - lp_f)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_f
            loss_pi  = -(torch.min(ratio * adv_f, clip_adv)).mean()

            # 4) backward + step
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()


        for _ in range(train_v_iters):
            opt.zero_grad()
            v_pred = model.value(o_f)                           # shape (T*N,)

            # MSE against *flattened* returns
            loss_v = F.mse_loss(v_pred, ret_f)

            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

        # # ───────────────────  evaluation  / early-stop  ─────────────────────
        # if ep % eval_every == 0:
        #     sharpe = eval_agent(env_eval.envs[0], model, device)
        #     print(f"[PPO] ep={ep:4d}  Sharpe={sharpe:6.3f}")

        # if sharpe > best_sharpe:
        #     best_sharpe, wait = sharpe, 0
        #     best_state = model.state_dict()
        # else:
        #     wait += 1
        # if wait >= patience:
        #     print(f"Early stopping at ep {ep}, best Sharpe={best_sharpe:.3f}")
        #     break

        if total_steps >= next_checkpoint:  # crossed a boundary?
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_sharpe": best_sharpe,
                    "episode": ep,
                },
                f"{checkpoint_dir}/ppo_step{total_steps}_ep{ep}.pth",
            )
            next_checkpoint += checkpoint_every

    model.load_state_dict(best_state)
    return model
