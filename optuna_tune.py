import optuna
import torch
import torch.nn as nn
import pickle
import pandas as pd
from dataset import get_dataloaders
from model import CodyConvNet
from utils import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_loader, valid_loader, _ = get_dataloaders(batch_size=64)
print("data fetched")

def objective(trial, train_loader, valid_loader, device):
    n_epochs = 40
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2,  log=True)
    conv_neuron_rate = trial.suggest_categorical("cnr", [0.25,0.5,1,2,4])
    dense_neuron_rate = trial.suggest_categorical("dnr", [0.25,0.5,1,2,4])
    dropout = trial.suggest_float("dropout", 0.0, 0.8)
    dropout2d = trial.suggest_float("dropout2d", 0.0, 0.6)


    model = CodyConvNet(
        image_dim=96, 
        n_classes=10, 
        conv_neuron_rate=conv_neuron_rate, 
        dense_neuron_rate=dense_neuron_rate,
        dropout=dropout, 
        dropout2d=dropout2d
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # -- TRENING --
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                epoch_val_loss += loss.item()
                
        avg_val_loss = epoch_val_loss / len(valid_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned() # type: ignore
    return best_val_loss


def save_best_trial_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE: # type: ignore
        if study.best_trial.number == trial.number:
            try:
                with open("best_params.pkl", "wb") as f:
                    pickle.dump(study.best_params, f)
                print(f"Dict saved. Loss: {trial.value:.4f}")
            except Exception as e:
                print(f"Save failed: {e}")


if __name__ == "__main__":
    print(f"Starting optimization on: {device}")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )

    study = optuna.create_study(
        direction="minimize", 
        study_name="Cody_STL10",
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective(trial, train_loader, valid_loader, device), 
        n_trials=45,
        callbacks=[save_best_trial_callback]
    )

    print(f"Lowest Loss: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    study.trials_dataframe().to_csv("study_results.csv")