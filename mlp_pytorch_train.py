import torch
import json
import os
import numpy as np
from tqdm import tqdm
from shutil import copy2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv("params.env")

from dataloader import load_reflection_spectra_dataloaders
from mlp_pytorch import ForwardMLP
from loss import ResonancePeaksLoss
from test_model import test_model
from training_utils import validate_model, parity_plot, MathEncoder
from metrics import evaluate_resonance_metrics

DATASET_PATH = os.getenv("DATASET_PATH")



if __name__ == '__main__':

    layers = [1024,512,256,128]
    lr = float(os.getenv("LR"))
    epochs = int(os.getenv("EPOCHS"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    patience = int(os.getenv("PATIENCE"))
    print("lr:", lr, "epochs:", epochs, "batch size:", batch_size, "patience:", patience)

    normalize_spectra = False if int(os.getenv("NORMALIZE_SPECTRA")) == 0 else True
    print("normalize spectra:", normalize_spectra)
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelength = load_reflection_spectra_dataloaders(
        DATASET_PATH,
        test_fraction=0.25,
        normalize_spectra=normalize_spectra
    )

    # setup model
    mlp = ForwardMLP(
        hidden_layers=layers,
        activation_name="GELU"
    )

    w_amp = float(os.getenv("W_AMP"))
    w_fd = float(os.getenv("W_FD"))
    w_wass = float(os.getenv("W_WASS"))
    w_sam = float(os.getenv("W_SAM"))
    print("loss weights", w_amp, w_fd, w_wass, w_sam)
    loss_function = ResonancePeaksLoss(w_amp,w_fd,w_wass,w_sam)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1.5e-05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    arch_version = os.getenv("ARCH_VERSION")
    model_dir = os.getenv("MODEL_DIR")
    model_name = f"mlp_{arch_version}_{epochs}"
    model_path = f"{model_dir}/{model_name}.pth"
    print("final model path", model_path)
    os.makedirs(model_dir, exist_ok=True)

    # plots
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    best_val_loss = float('inf')
    best_val_epoch = 0

    # training loop
    if not os.path.exists(model_path):
        print()
        print("Starting training...")
        
        tq = tqdm(range(epochs))
        for epoch in tq:
            mlp.train() # model in training mode
            running_train_loss = 0.0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # average training loss
            avg_train_loss = running_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # validation loss
            avg_val_loss = validate_model(mlp, val_loader, loss_function)
            val_loss_history.append(avg_val_loss)

            # Step the scheduler
            scheduler.step(avg_val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            tq.set_description_str(f'Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {current_lr:.6f} | Best Val: {best_val_loss:.5f} ({best_val_epoch})')
            lr_history.append(current_lr)

            # early stopping: save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(mlp.state_dict(), model_path)
                best_val_epoch = epoch

        print('training finished')

        # save parameters file to model folder
        model_params_path = f"{model_dir}/{model_name}_params.env"
        copy2("params.env", model_params_path)

        # plot train and val loss
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # lr drops
        lr = np.array(lr_history)
        drop_epochs = np.where(lr[1:] < lr[:-1])[0] + 1  # +1 to align epoch index

        axes[0].plot(train_loss_history, label='Training Loss')
        axes[0].plot(val_loss_history, label='Validation Loss', linestyle='--')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for e in drop_epochs:
            axes[0].axvline(e, color='red', linestyle='--', alpha=0.6)
            axes[0].text(
                e, axes[0].get_ylim()[1],
                f"LR → {lr[e]:.1e}",
                color='red',
                fontsize=9,
                rotation=90,
                verticalalignment='top',
                horizontalalignment='right'
            )

        axes[1].plot(train_loss_history, label='Training Loss', alpha=0.8)
        axes[1].plot(val_loss_history, label='Validation Loss', linestyle='--', alpha=0.8)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss (Log Scale)')
        axes[1].grid(True, which="both", alpha=0.3)

        for e in drop_epochs:
            axes[1].axvline(e, color='red', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}_loss_curve.png", dpi=300)
        plt.close()
        

    else:
        print(f"loading existing model from {model_path}")
        mlp.load_state_dict(torch.load(model_path))

    # eval model on test set
    mlp.eval()

    # parity plot
    parity_plot_path = f"{model_dir}/{model_name}_parity_plot.png"
    parity_plot(val_loader, parity_plot_path, mlp)
    print("generated parity plot")
    
    x_test_tensor = torch.from_numpy(x_test).float().view(x_test.shape[0], -1)
    
    with torch.no_grad():
        y_pred = mlp(x_test_tensor)
    y_pred = y_pred.numpy()

    y_test = y_test.squeeze()
    y_pred = y_pred.squeeze()
    
    # tests
    results = evaluate_resonance_metrics(y_test, y_pred, wavelength)
    print(f"  MSE:          {results['mse']:.6f}")
    print(f"  MAE:          {results['mae']:.6f}")
    print(f"  R2:           {results['r2']:.6f}")
    print(f"  Avg Shift:    {results['avg_peak_shift']:.6f} [nm]")
    print(f"  Max Shift:    {results['max_peak_shift']:.6f} [nm]")
    print(f"  Depth Error:  {results['avg_depth_error']:.6f}")
    print(f"  SAM:          {results['avg_sam']:.6f}")
    print(f"  Dip-only MSE: {results['roi_mse']:.6f}")

    # dump results to file
    json.dump(results, open(f"{model_dir}/{model_name}_results.json", "w+"), cls=MathEncoder, indent=4)

    # model test
    test_model(mlp, loss_function, x_test, y_test, 2, f"{model_dir}/{model_name}_test.png")
