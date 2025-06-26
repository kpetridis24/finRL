import os, random, datetime
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dash_writer import WRITER


class ForexPriceTrailingEnv(gym.Env):
    """
    Multi‐pair “price trailing” env.  Each episode picks a (possibly new) tic
    and runs for EP_LEN steps; when the tic changes we dump last episode’s
    data to CSV and start fresh.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        tic_col: str = "tic",
        window: int = 16,
        margin: float = 0.02,
        step_frac: float = 0.01,
        fee: float = 1e-4,
        alpha_trail: float = 0.3,
        alpha_pnl: float = 1.0,
        alpha_fee: float = 1.0,
        episode_len: int = 1000,
        pick_new_pair_every: int = 1,
    ):
        # master data & tickers
        self.full_df = full_df
        self.tic_col = tic_col
        self.pair_list = list(full_df[tic_col].unique())
        self.pick_new_pair_every = pick_new_pair_every
        self._reset_calls = 0

        # params
        self.W, self.M, self.U = window, margin, step_frac
        self.fee = fee
        self.alpha_trail, self.alpha_pnl, self.alpha_fee = (
            alpha_trail,
            alpha_pnl,
            alpha_fee,
        )
        self.EP_LEN = episode_len
        self.action_space = spaces.Discrete(3)

        # these get set in reset()
        self.df = None
        self.ptr = None
        self.start_idx = None
        self.position = 0
        self.agent_price = 0.0

        self.total_steps = 0
        self.episode_num = 0
        self.cum_pnl = 0.0

    def _pick_new_pair(self):
        tic = random.choice(self.pair_list)
        print(f"Picking new FX pair: {tic} (episode #{self._reset_calls})")
        sub = self.full_df[self.full_df[self.tic_col] == tic].reset_index(drop=True)
        self.df = sub
        # recompute normalization
        rets = self.df["x1"].values
        self.mu_abs = np.abs(rets).mean()
        self.sig = rets.std() + 1e-8

    def reset(self, *, seed=None, options=None):
        self.total_steps = 0
        self.episode_num += 1
        self.cum_pnl = 0.0
        self._reset_calls += 1

        if (self._reset_calls - 1) % self.pick_new_pair_every == 0:
            self._pick_new_pair()

        # pick random start so that we have W history + EP_LEN forward
        self.start_idx = random.randint(self.W, len(self.df) - self.EP_LEN - 1)
        self.ptr = self.start_idx
        self.position = 0
        self.agent_price = float(self.df.loc[self.ptr, "close"])

        pair = self.df[self.tic_col].iloc[0]
        self.tb_run_name = f"episode_{self.episode_num}_{pair}"

        return self._window_obs(self.ptr), {
            "forex_pair": self.df[self.tic_col].iloc[0] if self.df is not None else None
        }

    def set_fixed_window(self, start_idx: int, length: int):
        self.start_idx = start_idx
        self.ptr = start_idx
        self.EP_LEN = length

    def _window_obs(self, idx):
        sub = self.df.iloc[idx - self.W + 1 : idx + 1][['x1', 'x2', 'x3', 'x4', 'x5']]
        feat = sub.values.flatten()
        return np.concatenate([feat, [self.position]]).astype(np.float32)

    def _trailing_reward(self, idx):
        close = float(self.df.loc[idx, "close"])
        upper = close * (1 + self.M)
        lower = close * (1 - self.M)
        if lower <= self.agent_price <= upper:
            return 1 - abs(self.agent_price - close) / (self.M * close)
        else:
            deviation = (
                self.agent_price - upper
                if self.agent_price > upper
                else lower - self.agent_price
            )
            arg = min(deviation / (self.M * close), 10.0)
            return -np.exp(arg)

    def step(self, action: int):
        prev_pos = self.position
        self.position = [-1, 0, 1][action]

        close = float(self.df.loc[self.ptr, "close"])
        upper = close * (1 + self.M)
        lower = close * (1 - self.M)

        WRITER.add_scalars(
            f"Live Agent Performance",
            tag_scalar_dict={
                "upper": upper,
                "close": close,
                "lower": lower,
                "agent": self.agent_price,
            },
            global_step=self.total_steps,
        )

        WRITER.add_scalars(
            f"{self.tb_run_name}/live_performance",
            tag_scalar_dict={"upper": upper, "lower": lower, "agent": self.agent_price},
            global_step=self.total_steps,
        )

        # pnl + fee
        ret = float(self.df.loc[self.ptr + 1, "x1"])
        r_pnl = self.position * ret
        self.cum_pnl += r_pnl
        r_fee = -self.fee * abs(self.position - prev_pos)

        WRITER.add_scalar(
            f"{self.tb_run_name}/cumulative_pnl",
            self.cum_pnl,
            global_step=self.total_steps,
        )

        WRITER.add_scalar(
            f"Live Cumulative PnL", self.cum_pnl, global_step=self.total_steps
        )

        self.total_steps += 1

        # move the “agent_price”
        if self.position == 1:
            self.agent_price += self.U * abs(self.agent_price - upper)
        elif self.position == -1:
            self.agent_price -= self.U * abs(self.agent_price - lower)

        r_trail = self._trailing_reward(self.ptr)
        r_total = (
            self.alpha_trail * r_trail * self.mu_abs / self.sig
            + self.alpha_pnl * r_pnl / self.sig
            + self.alpha_fee * r_fee / self.sig
        )

        self.ptr += 1
        obs = self._window_obs(self.ptr)
        done = False
        truncated = (self.ptr - self.start_idx) >= self.EP_LEN

        return (
            obs,
            r_total,
            done,
            truncated,
            {"pnl": r_pnl, "trail": r_trail, "prev_pos": self.position},
        )

    def save_asset_memory(self):
        # at any point you can still get the last episode’s PnL series
        return pd.DataFrame(
            {"date": self._hist_dates, "account_value": np.cumsum(self.pnl_memory)}
        )

    def save_action_memory(self):
        return pd.DataFrame({"date": self._hist_dates, "action": self.actions_memory})
