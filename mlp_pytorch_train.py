import torch
import json
import os
import numpy as np

from utils.config import TrainingConfig
from utils.training_utils import validate_model, parity_plot, MathEncoder
from utils.dataloader import load_reflection_spectra_dataloaders
from mlp_pytorch import ForwardMLP
from loss import ResonancePeaksLoss
from test_model import test_model
from metrics import evaluate_resonance_metrics

from model_trainer import ModelTrainer

if __name__ == '__main__':

    # layers = [1024,512,256,128]
    layers = [512,256,128]
    config = TrainingConfig("mlp", "params.env")
    config.print()

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelength = load_reflection_spectra_dataloaders(
        config.dataset_path,
        test_fraction=0.25,
    )

    # setup model
    mlp = ForwardMLP(
        hidden_layers=layers,
        activation_name="GELU"
    )

    loss_function = ResonancePeaksLoss(config.w_amp,config.w_fd,config.w_wass,config.w_sam,peaks_importance=False)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=config.learning_rate, weight_decay=1.5e-05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.patience
    )

    print("final model path", config.model_path)
    os.makedirs(config.model_dir, exist_ok=True)

    # training loop
    if not os.path.exists(config.model_path):
        print()
        print("Starting training...")
        
        trainer = ModelTrainer(
            train_loader,
            val_loader,
            loss_function,
            optimizer,
            scheduler
        )

        trainer.training_loop(mlp, validate_model, config, inverse=False)
        print('training finished')

        trainer.training_stats(config)        

    else:
        print(f"loading existing model from {config.model_path}")
        mlp.load_state_dict(torch.load(config.model_path))




    # eval model on test set
    mlp.eval()

    # parity plot
    parity_plot_path = f"{config.model_dir}/{config.model_name}_parity_plot.png"
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
    json.dump(results, open(f"{config.model_dir}/{config.model_name}_results.json", "w+"), cls=MathEncoder, indent=4)

    # model test
    test_model(mlp, loss_function, x_test, y_test, 2, f"{config.model_dir}/{config.model_name}_test.png")
