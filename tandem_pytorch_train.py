import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copy2
from dotenv import load_dotenv
load_dotenv("tandem_params.env")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlp_pytorch import ForwardMLP
from dataloader import load_reflection_spectra_dataloaders
from loss import ResonancePeaksLoss
from training_utils import validate_tandem_model, evaluate_peak_shift

DATASET_PATH = os.getenv("DATASET_PATH")
MODEL_DIR = os.getenv("MODEL_DIR")

FORWARD_MODEL_NAME = os.getenv("FORWARD_MODEL_NAME")


class TandemModel(nn.Module):
    def __init__(self, forward_model, inverse_model):
        super(TandemModel, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        
        # freeze forward model
        self.forward_model.eval()
        for param in self.forward_model.parameters():
            param.requires_grad = False

    def forward(self, forward_y):
        # forward_y is the target of the forward model (R)
        # predicted_inverse_y is the target\
        #   predicted by the inverse model (geo) using the forward model's\
        #   prediction as input
        predicted_inverse_y = self.inverse_model(forward_y)
        
        # reconstruct the target of the forward model (R)\
        #   by passing the predicted target of the inverse model (geo\
        #   in the frozen forward model
        reconstructed_forward_y = self.forward_model(predicted_inverse_y)
        
        return predicted_inverse_y, reconstructed_forward_y
    

if __name__ == '__main__':

    lr = float(os.getenv("LR"))
    epochs = int(os.getenv("EPOCHS"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    patience = int(os.getenv("PATIENCE"))
    print("lr:", lr, "epochs:", epochs, "batch size:", batch_size, "patience:", patience)
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelength = load_reflection_spectra_dataloaders(
        DATASET_PATH,
        test_fraction=0.25
    )

    # define forward and inverse models
    forward_model = ForwardMLP(activation_name="GELU") 
    inverse_model = ForwardMLP(input_dim=81, output_dim=4, activation_name="GELU") 

    # load last trained forward model    
    forward_model.load_state_dict(torch.load(FORWARD_MODEL_NAME))

    # define tandem model
    tandem_model = TandemModel(inverse_model=inverse_model, forward_model=forward_model)
    
    # optimizer on inverse model's parameters
    optimizer = torch.optim.Adam(tandem_model.inverse_model.parameters(), lr=lr)


    w_amp = float(os.getenv("W_AMP"))
    w_fd = float(os.getenv("W_FD"))
    w_sd = float(os.getenv("W_SD"))
    w_wass = float(os.getenv("W_WASS"))
    print("loss weights", w_amp, w_fd, w_sd, w_wass)
    loss_function = ResonancePeaksLoss(w_amp,w_fd,w_sd,w_wass)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    arch_version = os.getenv("ARCH_VERSION")
    model_dir = os.getenv("MODEL_DIR")
    model_name = f"tandem_{arch_version}_{epochs}"
    model_path = f"{model_dir}/{model_name}.pth"
    print("final model path", model_path)
    os.makedirs(model_dir, exist_ok=True)


    # plots
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_val_epoch = 0

    # training loop
    if not os.path.exists(model_path):
        print()
        print("Starting training...")
        
        tq = tqdm(range(epochs))
        for epoch in tq:
            tandem_model.train() # model in training mode
            running_train_loss = 0.0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                optimizer.zero_grad()
                geo_prediction, spectra_reconstructed = tandem_model(targets)
                loss = loss_function(spectra_reconstructed, targets.squeeze())
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # average training loss
            avg_train_loss = running_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # validation loss
            avg_val_loss = validate_tandem_model(tandem_model, val_loader, loss_function)
            val_loss_history.append(avg_val_loss)

            # Step the scheduler
            scheduler.step(avg_val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            tq.set_description_str(f'Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {current_lr:.6f} | Best Val: {best_val_loss:.5f} ({best_val_epoch})')
            # TODO log epochs

            # early stopping: save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(tandem_model.state_dict(), model_path)
                best_val_epoch = epoch

        print('training finished')

        # save parameters file to model folder
        model_params_path = f"{model_dir}/{model_name}_params.env"
        copy2("tandem_params.env", model_params_path)

        # plot train and val loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_dir}/loss_curve_{model_name}.png")
        

    else:
        print(f"loading existing model from {model_path}")
        tandem_model.load_state_dict(torch.load(model_path))

    # eval model on test set
    tandem_model.eval()
    
    test_spectra = torch.from_numpy(y_test).float().view(y_test.shape[0], -1)
    
    with torch.no_grad():
        geo_prediction, spectra_reconstructed = tandem_model(test_spectra)

    pred_spectra = spectra_reconstructed.numpy()

    # metrics
    y_test = y_test.squeeze()
    pred_spectra = pred_spectra.squeeze()
    mse = mean_squared_error(y_test, pred_spectra)
    mae = mean_absolute_error(y_test, pred_spectra)
    r2 = r2_score(y_test, pred_spectra)
    peak_shift = evaluate_peak_shift(y_test, pred_spectra, wavelength)

    print(f"test - MSE: {mse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f} | PS: {peak_shift:.6f}")
