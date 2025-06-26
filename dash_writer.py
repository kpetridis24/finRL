from torch.utils.tensorboard import SummaryWriter

DASH_WRITER_LOG_DIR = "runs/price_trail_experiment/ddqn"
WRITER = SummaryWriter(DASH_WRITER_LOG_DIR)
