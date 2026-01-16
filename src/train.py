import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import argparse

from load_data import load_df
from dataset import PortfolioDataset
from model import POptModel
from loss import SharpeLoss, WeightPenalty

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, required=True,
    help="Path to the returns dataframe"
)

parser.add_argument(
    "--portfolio_dim", type=int, required=True,
    help='Number of asset composing a portfolio'
)
parser.add_argument(
    "--lr", type=float, required=True,
)
parser.add_argument(
    "--batch_size", type=int, required=True,
)
parser.add_argument(
    "--lambda_w", type=float, required=True
)
args = parser.parse_args()


DATA_PATH = args.data_path
prices_df, returns_df, norm_returns_df = load_df(DATA_PATH)

num_timesteps, _ = returns_df.shape
train_lim = int(0.8 * num_timesteps)

train_pri_df = prices_df[:train_lim]
train_ret_df = returns_df[:train_lim]
train_inpts_df = norm_returns_df[:train_lim]

test_pri_df = prices_df[train_lim:]
test_ret_df = returns_df[train_lim:]
test_inpts_df = norm_returns_df[train_lim:]

train_dataset = PortfolioDataset(train_pri_df, train_ret_df, train_inpts_df, n_asset=args.portfolio_dim)
test_dataset = PortfolioDataset(test_pri_df, test_ret_df, test_inpts_df, n_asset=args.portfolio_dim)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f'using device: {device}')

model = POptModel(
    n_asset = train_dataset.k
).to(device)

sharpe_crit = SharpeLoss().to(device)
weight_crit = WeightPenalty(
    param=args.lambda_w
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

from datetime import datetime
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

writer = SummaryWriter(
    log_dir=f"runs/deep_portfolio_optimization/{run_id}"
)

writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={}
)

n_epoch = 10_000
for epoch in range(n_epoch):
    model.train()
    train_loss = 0.0
    train_sharpe_loss = 0.0
    train_weight_loss = 0.0

    print(
        f"Running epoch {epoch+1:03d} ..."
    )

    for batch in tqdm(train_dataloader, desc="Train"):
        x = batch['input'].to(device)      # (B, L, 2 * K) (price+returns)
        r = batch['returns'].to(device)

        optimizer.zero_grad()

        w = model(x)
        r = r[:, model.decision_step+1:, :]

        sharpe_loss = sharpe_crit(w, r)
        weight_loss = weight_crit(w)
        loss = sharpe_loss + weight_crit.param * weight_loss
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()
        train_sharpe_loss += sharpe_loss.item()
        train_weight_loss += weight_loss.item()

    train_loss /= len(train_dataloader)
    train_sharpe_loss /= len(train_dataloader)
    train_weight_loss /= len(train_dataloader)

    # --------------------
    # Evaluation
    # --------------------
    model.eval()
    test_loss = 0.0
    test_sharpe_loss = 0.0
    test_weight_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test"):
            x = batch['input'].to(device)      # (B, L, 2 * K) (price+returns)
            r = batch['returns'].to(device)

            w = model(x)
            r = r[:, model.decision_step+1:, :]
            sharpe_loss = sharpe_crit(w, r)
            weight_loss = weight_crit(w)
            loss = sharpe_loss + weight_crit.param * weight_loss

            test_loss += loss.item()
            test_sharpe_loss += sharpe_loss.item()
            test_weight_loss += weight_loss.item()

    test_loss /= len(test_dataloader)
    test_sharpe_loss /= len(test_dataloader)
    test_weight_loss /= len(test_dataloader)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)

    # You minimize negative Sharpe → log positive Sharpe
    writer.add_scalar("Sharpe/train", -train_sharpe_loss, epoch)
    writer.add_scalar("Sharpe/test", -test_sharpe_loss, epoch)

    writer.add_scalar("WeightPenalty/train", train_weight_loss, epoch)
    writer.add_scalar("WeightPenalty/test", test_weight_loss, epoch)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss       : {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f}             \n"

        f"          | "
        f"Train Sharpe.    : {-train_sharpe_loss:.4f}  | "
        f"Test Sharpe: {-test_sharpe_loss:.4f}    \n"

        f"          | "
        f"Train Weight Pen.: {train_weight_loss:.4f}  | "
        f"Test Weight Pen.: {test_weight_loss:.4f} \n"
    )