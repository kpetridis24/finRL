import multiprocessing as mp
from datetime import date
import pandas as pd
from gymnasium.vector import SyncVectorEnv
import torch
from torch.utils.tensorboard import SummaryWriter

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_forex_price_trailing import ForexPriceTrailingEnv

from models import LSTM_QNet, ReplayBuffer, PPOActorCritic
from train_ddqn import train_ddqn, train_ddqn_vectorized
from train_ppo import train_ppo, train_ppo_vectorized

_NUM_TRAIN_ENVS: int = 32
_NUM_EVAL_ENVS: int = 16


def make_env(
    full_df,
    writer,
    *,
    tic_col,
    window,
    margin,
    step_frac,
    fee,
    alpha_trail,
    alpha_pnl,
    alpha_fee,
    episode_len,
    pick_new_pair_every,
    name,
):
    def _thunk():
        return ForexPriceTrailingEnv(
            full_df,
            tic_col=tic_col,
            window=window,
            margin=margin,
            step_frac=step_frac,
            fee=fee,
            alpha_trail=alpha_trail,
            alpha_pnl=alpha_pnl,
            alpha_fee=alpha_fee,
            episode_len=episode_len,
            pick_new_pair_every=pick_new_pair_every,
            writer=writer,
            name=name,
        )

    return _thunk


def get_vectorized_train_env(df, writer):
    train_env = SyncVectorEnv(
        [
            make_env(
                df,
                tic_col="tic",
                window=16,
                margin=0.02,
                step_frac=0.1,
                fee=2e-4,
                alpha_trail=0.97,
                alpha_pnl=0.85,
                alpha_fee=1.0,
                episode_len=600,
                pick_new_pair_every=1,
                writer=writer,
                name=f"Env_{i}",
            )
            for i in range(_NUM_TRAIN_ENVS)
        ]
    )
    return train_env


def get_vectorized_eval_env(df, val_idx, val_len, writer):
    eval_env = SyncVectorEnv(
        [
            make_env(
                df,
                tic_col="tic",
                window=16,
                margin=0.02,
                step_frac=0.1,
                fee=2e-4,
                alpha_trail=0.97,
                alpha_pnl=0.85,
                alpha_fee=1.0,
                episode_len=150,
                pick_new_pair_every=1,
                writer=writer,
                name=f"Env_{i}",
            )
            for i in range(_NUM_EVAL_ENVS)
        ]
    )
    for e in eval_env.envs:
        e.set_fixed_window(val_idx, val_len)
    return eval_env


def load_and_preprocess():
    # 1) tickers
    majors = [
        "EURUSD=X",
        "USDJPY=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "NZDUSD=X",
    ]
    crosses = [
        "EURGBP=X",
        "EURJPY=X",
        "GBPJPY=X",
        "AUDJPY=X",
        "CADJPY=X",
        "EURAUD=X",
        "EURCAD=X",
        "EURCHF=X",
        "GBPCHF=X",
        "AUDCAD=X",
        "NZDJPY=X",
        "NZDCAD=X",
    ]
    cny = [
        "USDCNY=X",
        "EURCNY=X",
        "JPYCNY=X",
        "GBPCNY=X",
        "AUDCNY=X",
        "CADCNY=X",
        "CHFCNY=X",
        "NZDCNY=X",
    ]
    all_tickers = majors + crosses + cny
    seen = set()
    forex_ticks = [t for t in all_tickers if not (t in seen or seen.add(t))]

    # 2) download
    start_train = date(2000, 1, 1)
    end_train = date(2017, 1, 1)
    eval_span = (date(2016, 7, 1), date(2017, 1, 1))  # last 6 months
    test_span = (date(2017, 1, 1), date(2018, 6, 1))

    yfd = YahooDownloader(
        start_date=str(start_train), end_date=str(end_train), ticker_list=forex_ticks
    )
    df_train = add_fx_features(yfd.fetch_data())

    yfd2 = YahooDownloader(
        start_date=str(test_span[0]),
        end_date=str(test_span[1]),
        ticker_list=forex_ticks,
    )
    df_test = add_fx_features(yfd2.fetch_data())

    sample_df = df_train[df_train.tic == forex_ticks[0]].reset_index(drop=True)
    val_start_idx = sample_df[
        pd.to_datetime(sample_df.date) == pd.Timestamp(eval_span[0])
    ].index[0]
    val_length = int((eval_span[1] - eval_span[0]).days)

    return df_train, df_test, forex_ticks, val_start_idx, val_length


def add_fx_features_for_tick(g):
    g["close_prev"] = g.close.shift(1)
    g["high_prev"] = g.high.shift(1)
    g["low_prev"] = g.low.shift(1)
    g["x1"] = (g.close - g.close_prev) / g.close_prev
    g["x2"] = (g.high - g.high_prev) / g.high_prev
    g["x3"] = (g.low - g.low_prev) / g.low_prev
    g["x4"] = (g.high - g.close) / g.close
    g["x5"] = (g.close - g.low) / g.close
    return g


def add_fx_features(df, tic_col="tic"):
    return (
        df.groupby(tic_col, group_keys=False)
        .apply(add_fx_features_for_tick)
        .drop(columns=["close_prev", "high_prev", "low_prev"])
        .fillna(0)
    )


def run_ddqn(df_train, val_idx, val_len):
    ddqn_writer = SummaryWriter("runs/price_trail_experiment/ddqn")
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_env = ForexPriceTrailingEnv(
        df_train,
        tic_col="tic",
        window=16,
        margin=0.02,
        step_frac=0.1,
        fee=2e-4,
        alpha_trail=0.97,
        alpha_pnl=0.85,
        alpha_fee=1.0,
        episode_len=600,
        pick_new_pair_every=1,
        writer=ddqn_writer,
    )
    #get_vectorized_train_env(df_train, writer=ddqn_writer)
    eval_env = ForexPriceTrailingEnv(
        df_train,
        tic_col="tic",
        window=16,
        margin=0.02,
        step_frac=0.1,
        fee=2e-4,
        alpha_trail=0.97,
        alpha_pnl=0.85,
        alpha_fee=1.0,
        episode_len=600,
        pick_new_pair_every=1,
        writer=ddqn_writer,
    )#get_vectorized_eval_env(df_train, val_idx, val_len, writer=ddqn_writer)

    eval_env.set_fixed_window(val_idx, val_len)
    # model + buffer
    q_net = LSTM_QNet(
        window=16, feature_dim=5, lstm_hidden=128, fc_hidden=64, action_dim=3
    ).to(device)
    target_q = LSTM_QNet(
        window=16, feature_dim=5, lstm_hidden=128, fc_hidden=64, action_dim=3
    ).to(device)
    target_q.load_state_dict(q_net.state_dict())
    buffer = ReplayBuffer(capacity=1_000_000)

    # train
    best = train_ddqn(
        train_env,
        eval_env,
        q_net,
        target_q,
        ReplayBuffer(capacity=200_000),
        episodes=5000,
        checkpoint_every=100,
        eval_every=50,
        patience=8,
        batch_size=512,
        eps_start=1.0,
        eps_decay=25000,
        target_update=10,
        device=device,
        writer=ddqn_writer,
    )
    # best = train_ddqn_vectorized(
    #     train_env,
    #     eval_env,
    #     q_net,
    #     target_q,
    #     buffer,
    #     episodes=50_000,
    #     checkpoint_every=100_000,
    #     eval_every=50,
    #     patience=8,
    #     batch_size=512,
    #     eps_start=1.0,
    #     eps_decay=25000,
    #     target_update=10,
    #     device=device,
    #     writer=ddqn_writer,
    # )
    # save final
    torch.save(best.state_dict(), "ddqn_forex_trained_model.pth")


def run_ppo(df_train, val_idx, val_len):
    ppo_writer = SummaryWriter("runs/price_trail_experiment/ppo")
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_env = ForexPriceTrailingEnv(
        df_train,
        tic_col="tic",
        window=16,
        margin=0.02,
        step_frac=0.1,
        fee=2e-4,
        alpha_trail=0.97,
        alpha_pnl=0.85,
        alpha_fee=1.0,
        episode_len=600,
        pick_new_pair_every=1,
        writer=ppo_writer,
    )
    eval_env = ForexPriceTrailingEnv(
        df_train,
        tic_col="tic",
        window=16,
        margin=0.02,
        step_frac=0.1,
        fee=2e-4,
        alpha_trail=0.97,
        alpha_pnl=0.85,
        alpha_fee=1.0,
        episode_len=600,
        pick_new_pair_every=1,
        writer=ppo_writer,
    )
    # train_env = get_vectorized_train_env(df_train, writer=ppo_writer)
    # eval_env = get_vectorized_eval_env(df_train, val_idx, val_len, writer=ppo_writer)
    eval_env.set_fixed_window(val_idx, val_len)

    # model
    model = PPOActorCritic(window=16, feature_dim=5, lstm_hidden=128, fc_hidden=64).to(
        device
    )

    # train
    best = train_ppo(
        train_env,
        eval_env,
        model,
        episodes=500,
        gamma=0.995,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        max_grad_norm=0.5,
        eval_every=50,
        checkpoint_every=100,
        patience=8,
        device=device,
    )
    # best = train_ppo_vectorized(
    #     train_env,
    #     eval_env,
    #     model,
    #     episodes=500,
    #     gamma=0.995,
    #     lam=0.95,
    #     clip_ratio=0.2,
    #     pi_lr=3e-4,
    #     vf_lr=1e-3,
    #     train_pi_iters=80,
    #     train_v_iters=80,
    #     max_grad_norm=0.5,
    #     eval_every=50,
    #     checkpoint_every=100_000,
    #     patience=8,
    #     device=device,
    # )
    torch.save(best.state_dict(), "ppo_forex_trained_model.pth")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    df_train, df_test, ticks, val_idx, val_len = load_and_preprocess()

    p1 = mp.Process(target=run_ddqn, args=(df_train.copy(), val_idx, val_len))
    p2 = mp.Process(target=run_ppo, args=(df_train.copy(), val_idx, val_len))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print("Both DDQN and PPO training complete.")
