import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import argparse

from load_data import load_df
from dataset import PortfolioDataset
from model import POptModel
from loss import SharpeLoss, WeightPenalty

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to the returns dataframe"
)
args = parser.parse_args()


DATA_PATH = args.data_path
returns_df = load_df(DATA_PATH)


num_timesteps, _ = returns_df.shape
train_lim = int(0.8 * num_timesteps)

train_df = returns_df[:train_lim]
test_df = returns_df[train_lim:]

train_dataset = PortfolioDataset(train_df)
test_dataset = PortfolioDataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=256)


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f'using device: {device}')

model = POptModel(n_asset = train_dataset.k).to(device)

sharpe_crit = SharpeLoss().to(device)
weight_crit = WeightPenalty(param=0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

n_epoch = 100
for epoch in range(n_epoch):
    model.train()
    train_loss = 0.0
    train_sharpe_loss = 0.0
    train_weight_loss = 0.0

    print(
        f"Running epoch {epoch+1:03d} ..."
    )
    
    for batch in tqdm(train_dataloader, desc="Train"):
        x = batch['input_r'].to(device)      # (B, L, K)

        optimizer.zero_grad()

        w, next_r = model(x)
        sharpe_loss = sharpe_crit(w, next_r)
        weight_loss = weight_crit(w)
        loss = sharpe_loss + weight_loss
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
        for batch in tqdm(test_dataloader, desc="Eval"):
            x = batch['input_r'].to(device)

            w, next_r = model(x)
            sharpe_loss = sharpe_crit(w, next_r)
            weight_loss = weight_crit(w)
            loss = sharpe_loss + weight_loss

            test_loss += loss.item()
            test_sharpe_loss += sharpe_loss.item()
            test_weight_loss += weight_loss.item()

    test_loss /= len(test_dataloader)
    test_sharpe_loss /= len(test_dataloader)
    test_weight_loss /= len(test_dataloader)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss       : {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f}               | \n"

        f"          | "
        f"Train Sharpe.    : {-train_sharpe_loss:.4f}  | "
        f"Test Sharpe: {-test_sharpe_loss:.4f}     | \n"

        f"          | "
        f"Train Weight Pen.: {train_weight_loss:.4f}  | "
        f"Test Weight Pen.: {test_weight_loss:.4f} | \n"
    )