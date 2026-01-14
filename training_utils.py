import torch
import numpy as np

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
            
            x_prediction, y_reconstructed = tandem_model(targets)
            loss = criterion(y_reconstructed, targets.squeeze())
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