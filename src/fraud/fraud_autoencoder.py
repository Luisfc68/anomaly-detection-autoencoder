import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fraud.config import DEVICE


class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 1e-3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),  # no activation — output is unbounded
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Training history, populated by .fit()
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def _train_one_epoch(self, loader) -> float:
        """One full pass over the training data. Returns mean loss."""
        self.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            loss = self.criterion(self(batch), batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item() * batch.size(0)

        return total / len(loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader) -> float:
        """Compute mean loss over a loader without updating weights."""
        self.eval()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            total += self.criterion(self(batch), batch).item() * batch.size(0)
        return total / len(loader.dataset)

    def fit(self, train_loader, test_loader, epochs: int = 20, verbose: bool = True):
        """
        Train the autoencoder and keep the best weights by validation loss.
        """
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate(test_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                marker = " ← best"
            else:
                marker = ""

            if verbose and (epoch % 2 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:2d}/{epochs}  "
                    f"train={train_loss:.5f}  val={val_loss:.5f}{marker}"
                )

        # Restore the best checkpoint before returning
        self.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model (val_loss = {best_val_loss:.5f})")

    @torch.no_grad()
    def reconstruction_errors(self, X_np: np.ndarray) -> np.ndarray:
        """
        Compute per-sample MSE reconstruction error for a numpy array.

        WHY a dedicated method for this?
        Reconstruction error is the model's "fraud score" — we use it both
        to pick a threshold and to score new transactions at inference time.
        Centralising the logic avoids duplicating tensor conversion everywhere.
        """
        self.eval()
        tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        recon = self(tensor)
        return ((tensor - recon) ** 2).mean(dim=1).cpu().numpy()

    def predict(self, X_np: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary fraud predictions (1 = fraud) given a threshold."""
        return (self.reconstruction_errors(X_np) > threshold).astype(int)
