import torch
import numpy as np

import matplotlib.pyplot as plt

from json import JSONEncoder
class MathEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float32) or isinstance(o, np.float64):
            return float(o)
        return o.__dict__


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

def validate_tandem_model(tandem_model, val_loader, criterion):
    tandem_model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad(): # No gradient needed for validation
        for inputs, targets in val_loader:
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            
            geo_prediction, spectrum_reconstructed = tandem_model(targets)
            loss = criterion(spectrum_reconstructed, targets.squeeze())
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss

def evaluate_peak_shift(y_true, y_pred, wavelengths):
    true_peaks_nm = wavelengths[np.argmin(y_true, axis=1)]
    pred_peaks_nm = wavelengths[np.argmin(y_pred, axis=1)]
    
    shifts = np.abs(true_peaks_nm - pred_peaks_nm)
    avg_shift = np.mean(shifts)
    print("avg shift", avg_shift)

    return avg_shift

def parity_plot(val_loader, parity_plot_path, model, k=400):
    true_peaks = []
    pred_peaks = []

    # get statistics from k samples of validation set
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i > k: break

            inputs = inputs.squeeze()
            targets = targets.squeeze()
            
            preds = model(inputs)
            
            # argmin - index of minimum value (dip)
            batch_true_peaks = torch.argmin(targets, dim=1).cpu().numpy()
            batch_pred_peaks = torch.argmin(preds, dim=1).cpu().numpy()
            
            true_peaks.extend(batch_true_peaks)
            pred_peaks.extend(batch_pred_peaks)
    
    # parity
    plt.figure(figsize=(8, 8))
    plt.scatter(true_peaks, pred_peaks, alpha=0.5, s=15, color='#1f77b4', edgecolor='k', linewidth=0.5)
    
    # diagonal line - perfect alignment
    limit_min = min(min(true_peaks), min(pred_peaks))
    limit_max = max(max(true_peaks), max(pred_peaks))
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', label='Perfect Match', linewidth=2)
    
    plt.xlabel("True Peak Index")
    plt.ylabel("Predicted Peak Index")
    plt.title("Parity Plot\n(Points on the line = Perfect Alignment)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(parity_plot_path)
    plt.close()
    

############################################################################################################
# DATA DIMENSIONALITY CHECKS

def unsqueeze_2d_tensor(x):
    is_2d = x.dim() == 2
    if is_2d:
        x = x.unsqueeze(1) # [Batch, Length] -> [Batch, 1, Length]
    return x

def squeeze_3d_tensor(x):
    is_3d = x.dim() == 2
    if is_3d and x.shape[1] == 1:
        x = x.squeeze(1)  # [Batch, 1, Length] -> [Batch, Length]
    return x