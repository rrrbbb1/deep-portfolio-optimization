import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import argparse

from src.load_data import load_df, get_asset_list
from src.init_dataset import init_dataset
from src.model import POptModel
from src.loss import SharpeLoss, WeightPenalty

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, required=True
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
df_map = load_df(DATA_PATH)
asset_list = get_asset_list(DATA_PATH)

feature_mod = 'signal'
train_dataloader, test_dataloader = init_dataset(df_map, feature_mod, asset_list, split_ratio=0.8, batch_size=args.batch_size)


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f'using device: {device}')

model = POptModel(
    n_asset = train_dataloader.dataset.k,
    ts_dim = train_dataloader.dataset.ts_dim
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