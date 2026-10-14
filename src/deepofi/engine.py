from pathlib import Path

import torch
from torch import nn


def train(model, train_loader, val_loader, learning_rate, weights_path, epochs=50, patience=5, device="cuda", save_weights=True):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.numel()
            train_count += y.numel()

        train_loss = train_loss_sum / train_count
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                predictions = model(X)
                loss = criterion(predictions, y)
                val_loss_sum += loss.item() * y.numel()
                val_count += y.numel()

        val_loss = val_loss_sum / val_count
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"Epoch {epoch}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: {patience} epochs without improvement")
            break

    
    model.load_state_dict(best_state)
    model.eval()

    if save_weights:
        project_root = Path(__file__).resolve().parents[2]
        weights_path = project_root / "weights" / weights_path
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, weights_path)
        print(f"Weights saved: {weights_path}")

    return model, history


def load_weights(model, weights_path, device="cpu"):
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "weights" / weights_path
    model = model.to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model