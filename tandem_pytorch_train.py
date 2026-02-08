import torch
import json
import os
import numpy as np

from mlp_pytorch import ForwardMLP
from cnn_pytorch_inverse import InverseCNN
from utils.dataloader import load_reflection_spectra_dataloaders
from loss import ResonancePeaksLoss
from utils.training_utils import validate_tandem_model, MathEncoder
from metrics import evaluate_resonance_metrics
from test_model import test_tandem_model

from utils.augment_data import RandomGaussianBlur1D


from utils.config import TrainingConfig

from tandem_model import TandemModel


if __name__ == '__main__':

    use_augmenter = False
    augmenter = None
    
    config = TrainingConfig("tandem","tandem_params.env")
    config.print()
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelength = load_reflection_spectra_dataloaders(
        config.dataset_path,
        test_fraction=0.25
    )

    # define forward and inverse models
    # layers = [1024,512,256,128]
    layers = [512,256,128]
    forward_model = ForwardMLP(hidden_layers=layers, activation_name="GELU") 
    # inverse_model = ForwardMLP(input_dim=81, output_dim=4, activation_name="GELU") 
    inverse_model = InverseCNN(output_geom_dim=4)

    # load last trained forward model    
    forward_model.load_state_dict(torch.load(config.forward_model_name))

    # define tandem model
    tandem_model = TandemModel(inverse_model=inverse_model, forward_model=forward_model)
    
    # optimizer on inverse model's parameters
    optimizer = torch.optim.Adam(tandem_model.inverse_model.parameters(), lr=config.learning_rate)


    loss_function = ResonancePeaksLoss(config.w_amp,config.w_fd,config.w_wass,config.w_sam,peaks_importance=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.patience
    )

    print("final model path", config.model_path)
    os.makedirs(config.model_dir, exist_ok=True)

    if use_augmenter:
        augmenter = RandomGaussianBlur1D(kernel_size=5, sigma_range=(1.0, 2.0), p=0.5)
    

    # training loop
    if not os.path.exists(config.model_path):
        print()
        print("Starting training...")
        
        
        from model_trainer import ModelTrainer
        trainer = ModelTrainer(
            train_loader,
            val_loader,
            loss_function,
            optimizer,
            scheduler,
            augmenter
        )

        trainer.training_loop(tandem_model, validate_tandem_model, config, inverse=True)

        print('training finished')

        trainer.training_stats(config)
        

    else:
        print(f"loading existing model from {config.model_path}")
        tandem_model.load_state_dict(torch.load(config.model_path))

    # eval model on test set
    tandem_model.eval()
    
    test_spectra = torch.from_numpy(y_test).float().view(y_test.shape[0], -1)
    
    with torch.no_grad():
        geo_prediction, spectra_reconstructed = tandem_model(test_spectra)

    pred_spectra = spectra_reconstructed.numpy()
    test_spectra = test_spectra.numpy()

    # tests
    results = evaluate_resonance_metrics(test_spectra, pred_spectra, wavelength)
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
    test_tandem_model(loss_function, test_spectra, pred_spectra, 2, f"{config.model_dir}/{config.model_name}_test.png")
