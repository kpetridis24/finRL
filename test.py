import multiprocessing as mp
from datetime import date
import pandas as pd
from gymnasium.vector import SyncVectorEnv
import torch
from torch.utils.tensorboard import SummaryWriter

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_forex_price_trailing import ForexPriceTrailingEnv

from models import LSTM_QNet, ReplayBuffer, PPOActorCritic
from train_all import load_and_preprocess


_, df_test, _, _, _ = load_and_preprocess()
ddqn_writer = SummaryWriter("test/price_trail_experiment/ddqn")

env = ForexPriceTrailingEnv(
    df_test,
    tic_col="tic",
    window=16,
    margin=0.02,
    step_frac=0.1,
    fee=2e-4,
    alpha_trail=0.97,
    alpha_pnl=0.85,
    alpha_fee=1.0,
    episode_len=200,
    pick_new_pair_every=1,
    writer=ddqn_writer,
)

obs, _ = env.reset()
history = {"date": [], "close": [], "agent": [], "upper": [], "lower": [], "reward": []}

trained_model_path = "checkpoints/ddqn_vec/qnet_step800000.pth"
trained_actor = LSTM_QNet(
    window=16, feature_dim=5, lstm_hidden=128, fc_hidden=64, action_dim=3
)
trained_actor.load_state_dict(torch.load(trained_model_path))
trained_actor.eval()

for t in range(200):
    # record pre-step values
    date = env.df.loc[env.ptr, "date"]
    close = float(env.df.loc[env.ptr, "close"])
    upper = close * (1 + env.M)
    lower = close * (1 - env.M)

    history["date"].append(pd.to_datetime(date))
    history["close"].append(close)
    history["agent"].append(env.agent_price)
    history["upper"].append(upper)
    history["lower"].append(lower)

    # assume ElegantRL style: agent.act returns a tensor shape (1,1)
    st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = trained_actor.act_greedy(st).cpu().item()

    # step
    obs, reward, done, truncated, info = env.step(int(action))
    history["reward"].append(reward)
    if done or truncated:
        break

pass
