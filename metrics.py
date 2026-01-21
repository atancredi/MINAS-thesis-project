import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_resonance_metrics(y_true, y_pred, wavelength_grid):
    """
    y_true, y_pred: (N, 81)
    wavelength_grid: (81,)
    """
    
    n_samples = y_true.shape[0]
    
    peak_shifts = []
    depth_errors = []
    sams = [] # spectral angle mapper
    dips_mses = [] # mse inside dips
    
    for i in range(n_samples):
        true_spec = y_true[i]
        pred_spec = y_pred[i]
        
        # horizontal shift of the global minimum (biggest peak)
        idx_true = np.argmin(true_spec)
        idx_pred = np.argmin(pred_spec)
        
        val_true = wavelength_grid[idx_true]
        val_pred = wavelength_grid[idx_pred]
        peak_shifts.append(np.abs(val_true - val_pred))
        
        # vertical shift of the global minimum (biggest peak)
        depth_true = np.min(true_spec)
        depth_pred = np.min(pred_spec)
        depth_errors.append(np.abs(depth_true - depth_pred))
        
        # sam - spectral angle mapper
        #   angle between spectra-vectors in 81-dim space
        #   encodes the shape of the spectra
        #   SAM = arccos( (A . B) / (|A| * |B|) )
        dot_product = np.dot(true_spec, pred_spec)
        norm_true = np.linalg.norm(true_spec)
        norm_pred = np.linalg.norm(pred_spec)
        cosine_sim = np.clip(dot_product / (norm_true * norm_pred + 1e-8), -1.0, 1.0) # clip
        sam = np.arccos(cosine_sim)
        sams.append(sam)
        
        # MSE where the true signal is < 0.9 (inside dips)
        mask = true_spec < 0.90 
        if np.sum(mask) > 0:
            dips_mse = np.mean((true_spec[mask] - pred_spec[mask])**2)
        else:
            # if no peaks: standard MSE
            dips_mse = np.mean((true_spec - pred_spec)**2)
        dips_mses.append(dips_mse)

    # Aggregate Results
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "avg_peak_shift": np.mean(peak_shifts),
        "max_peak_shift": np.max(peak_shifts),
        "avg_depth_error": np.mean(depth_errors),
        "avg_sam": np.mean(sams),
        "roi_mse": np.mean(dips_mses)
    }
    
    return metrics

