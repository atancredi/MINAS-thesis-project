from fire import Fire
import torch
import matplotlib.pyplot as plt
from random import sample
import os
from dotenv import load_dotenv
load_dotenv("params.env")

from loss import ResonancePeaksLoss 
from mlp_pytorch import ForwardMLP
from mlp_pytorch_train import load_reflection_spectra_dataloaders

DATASET_PATH = os.getenv("DATASET_PATH")

train_loader, val_loader, x_test, y_test, wavelength = load_reflection_spectra_dataloaders(
    DATASET_PATH,
    test_fraction=0.25
)


def run_test(model_path, n=1):

    if n > 3:
        raise ValueError("Too many graphs to plot...")

    print("sample prediction with model", model_path)

    # load model and weights
    model = ForwardMLP(hidden_layers=[1024, 512, 256, 128], activation_name="GELU")
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

    # set model to eval mode
    model.eval()

    w_amp = float(os.getenv("W_AMP"))
    w_fd = float(os.getenv("W_FD"))
    w_sd = float(os.getenv("W_SD"))
    w_wass = float(os.getenv("W_WASS"))
    print(w_amp, w_fd, w_sd, w_wass)
    criterion = ResonancePeaksLoss(w_amp,w_fd,w_sd,w_wass)

    total_samples = len(x_test)
    indices = sample(range(total_samples), k=n)
    print(f"indices", indices)

    fig, ax = plt.subplots(ncols=n,nrows=1,figsize=(int(5*n), 3))
    if n==1:
        ax = [ax]
    
    for i, idx in enumerate(indices):

        x_sample = x_test[idx] 
        y_sample = y_test[idx]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            prediction = model(x_tensor)
        y_pred = prediction.cpu()

        y_true = torch.as_tensor(y_sample, dtype=torch.float32)
        y_true = y_true.T

        total = criterion(y_pred, y_true)

        print(f"loss {total.item():.5f}")

        y_true_np = y_true.detach().numpy().flatten()
        y_pred_np = y_pred.detach().numpy().flatten()

        ax[i].plot(y_true_np, label='Ground Truth', color='black', linewidth=2, linestyle='--')
        ax[i].plot(y_pred_np, label='MLP Prediction', color='#d62728', linewidth=2)
        ax[i].fill_between(range(len(y_true_np)), y_true_np, y_pred_np, color='gray', alpha=0.2, label='Error')
        ax[i].set_title(f"Loss Analysis (Idx {idx}): Total={total.item():.4f}")
        ax[i].grid(True, alpha=0.3)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Fire(run_test)