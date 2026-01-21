import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from metrics import evaluate_resonance_metrics
from dataloader import load_reflection_spectra_data
from loss import ResonancePeaksLoss

def reconstruct_and_evaluate(model, input_spectra, wavelength_grid):
    
    input_spectra = np.array(input_spectra)
    spectra_tensor = torch.from_numpy(input_spectra).float()

    spectra_tensor = spectra_tensor.unsqueeze(0)
    
    device = next(model.parameters()).device
    spectra_tensor = spectra_tensor.to(device)

    model.eval()
    with torch.no_grad():
        predicted_geo, reconstructed_spectra = model(spectra_tensor)

    original_numpy = spectra_tensor.cpu().view(1,-1).numpy()
    reconstructed_numpy = reconstructed_spectra.cpu().view(1,-1).numpy()
    predicted_geo_numpy = predicted_geo.cpu().numpy()

    metrics = evaluate_resonance_metrics(original_numpy, reconstructed_numpy, wavelength_grid)

    return metrics, reconstructed_numpy, predicted_geo_numpy



def main(model_path, forward_model_path, dataset_path, output_path):

    from dotenv import load_dotenv
    load_dotenv("tandem_params.env")

    w_amp = float(os.getenv("W_AMP"))
    w_fd = float(os.getenv("W_FD"))
    w_wass = float(os.getenv("W_WASS"))
    w_sam = float(os.getenv("W_SAM"))
    print("loss weights", w_amp, w_fd, w_wass, w_sam)
    criterion = ResonancePeaksLoss(w_amp,w_fd,w_wass,w_sam)

    from generate_spectra import generate_spectrum
    spectrum_test, params = generate_spectrum(num_points=81, num_peaks=1, peak_type='gaussian', noise_level=0.001)
    print(params)

    from tandem_pytorch_train import TandemModel
    from mlp_pytorch import ForwardMLP
    from cnn_pytorch_inverse import InverseMetricsCNN
    # define forward and inverse models
    forward_model = ForwardMLP(activation_name="GELU") 
    inverse_model = ForwardMLP(input_dim=81, output_dim=4, activation_name="GELU") 

    # load last trained forward model    
    forward_model.load_state_dict(torch.load(forward_model_path))

    # define tandem model
    tandem_model = TandemModel(inverse_model=inverse_model, forward_model=forward_model)

    print(f"loading existing model from {model_path}")
    tandem_model.load_state_dict(torch.load(model_path))


    # reconstruct and evaluate spectra
    _,_,_,_,wavelengths = load_reflection_spectra_data(dataset_path)
    metrics, reconstructed_numpy, predicted_geo_numpy = reconstruct_and_evaluate(tandem_model, [spectrum_test], wavelengths)
    print(metrics, predicted_geo_numpy)


    y_pred = reconstructed_numpy
    y_sample = spectrum_test

    # XXX
    y_true = torch.as_tensor(y_sample, dtype=torch.float32).view(1, -1)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32).view(1, -1)

    total = criterion(y_pred, y_true)

    print(f"loss {total.item():.5f}")

    y_true_np = y_true.detach().numpy().flatten()
    y_pred_np = y_pred.detach().numpy().flatten()

    plt.plot(y_true_np, label='Ground Truth', color='black', linewidth=2, linestyle='--')
    plt.plot(y_pred_np, label='MLP Prediction', color='#d62728', linewidth=2)
    plt.fill_between(range(len(y_true_np)), y_true_np, y_pred_np, color='gray', alpha=0.2, label='Error')
    plt.title(f"Loss: {total.item():.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.savefig(output_path)    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
