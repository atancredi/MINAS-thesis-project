import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split 

from dataloader import load_reflection_spectra_data
from mlp_pytorch import ForwardMLP, ReflectivityDataset


DATASET_PATH = "__data/pisa_data_2025_12_02.h5"

def validate_model(model, val_loader, criterion):
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad(): # No gradient needed for validation
        for inputs, targets in val_loader:
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss

if __name__ == '__main__':

    lr = 0.001
    epochs = 20
    batch_size = 32
    
    torch.manual_seed(42)
    np.random.seed(42)

    x_full_train, x_test, y_full_train, y_test, wavelength = load_reflection_spectra_data(DATASET_PATH, test_fraction=0.05)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_full_train, y_full_train, test_size=0.1, random_state=42
    )

    print(f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
    
    train_dataset = ReflectivityDataset(x_train, y_train)
    val_dataset   = ReflectivityDataset(x_val, y_val)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # setup model
    mlp = ForwardMLP()
    loss_function = nn.MSELoss() 
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    arch_version = "01"
    model_dir = "__models"
    model_name = f"{model_dir}/mlp_model_{arch_version}.pth"
    os.makedirs(model_dir, exist_ok=True)

    # plots
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    # training loop
    if not os.path.exists(model_name):
        print("Starting training...")
        
        for epoch in range(epochs):
            mlp.train() # model in training mode
            running_train_loss = 0.0

            for i, (inputs, targets) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # average training loss
            avg_train_loss = running_train_loss / len(trainloader)
            train_loss_history.append(avg_train_loss)

            # validation loss
            avg_val_loss = validate_model(mlp, valloader, loss_function)
            val_loss_history.append(avg_val_loss)

            print(f'epoch {epoch+1}/{epochs} | train loss: {avg_train_loss:.5f} | val loss: {avg_val_loss:.5f}')

            # early stopping: save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(mlp.state_dict(), model_name)
                print(f"saved best model with val_loss {best_val_loss}")

        print('training finished')

        # plot train and val loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_dir}/mlp_pytorch_loss_curve.png")
        

    else:
        print(f"loading existing model from {model_name}")
        mlp.load_state_dict(torch.load(model_name))

    # eval model on test set
    mlp.eval()
    
    x_test_tensor = torch.from_numpy(x_test).float().view(x_test.shape[0], -1)
    
    with torch.no_grad():
        y_pred = mlp(x_test_tensor)

    y_pred = y_pred.numpy()

    # metrics
    y_test = y_test.squeeze()
    y_pred = y_pred.squeeze()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"test - MSE: {mse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f}")
