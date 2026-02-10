import torch
import json
import os
import numpy as np


from utils.augment_data import RandomGaussianBlur1D

from utils.config import TrainingConfig

from tandem_model import TandemModel


from utils.config import TrainingConfig
from utils.training_utils import validate_model, parity_plot, MathEncoder, validate_tandem_model
from utils.dataloader import load_reflection_spectra_dataloaders, load_reflection_spectra_data
from mlp_pytorch import ForwardMLP
from cnn_pytorch_inverse import InverseCNN
from loss import ResonancePeaksLoss, LossScheduler
from test_model import test_model, test_tandem_model
from metrics import evaluate_resonance_metrics

from model_trainer import ModelTrainer

ranges = {
    'w_amp':  (10.0, 50.0),
    'w_grad': (50.0, 5.0),
    'w_wass': (1.0, 20.0),
    'w_sam':  (1.0, 10.0)
}

tandem_ranges = {
    'w_amp':  (10.0, 50.0),
    'w_grad': (10.0, 50.0),
    'w_wass': (20.0, 1.0),
    'w_sam':  (1.0, 1.0)
}


from generated_spectra_test import reconstruct_and_evaluate
def reconstruct_evaluate_spectrum(tandem_model, wavelengths, spectrum_test, output_path = "res.png"):
    # reconstruct and evaluate spectra
    metrics, reconstructed_numpy, predicted_geo_numpy = reconstruct_and_evaluate(tandem_model, [spectrum_test], wavelengths)

    print(scaler_geo_tandem.inverse_transform(predicted_geo_numpy))    
    
    print(metrics, predicted_geo_numpy)

    y_pred = reconstructed_numpy
    y_sample = spectrum_test

    # XXX
    y_true = torch.as_tensor(y_sample, dtype=torch.float32).view(1, -1)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32).view(1, -1)

    total = loss_function(y_pred, y_true)

    print(f"loss {total.item():.5f}")

    y_true_np = y_true.detach().numpy().flatten()
    y_pred_np = y_pred.detach().numpy().flatten()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.plot(y_true_np, label='Ground Truth', color='black', linewidth=2, linestyle='--')
    ax.plot(y_pred_np, label='MLP Prediction', color='#d62728', linewidth=2)
    ax.fill_between(range(len(y_true_np)), y_true_np, y_pred_np, color='gray', alpha=0.2, label='Error')
    plt.title(f"Loss: {total.item():.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.legend()

    plt.savefig(output_path) 
    plt.close()


if __name__ == '__main__':

    layers = [1024,512,256,128]
    # layers = [512,256,128]
    training_config = TrainingConfig("mlp", "params.env")
    training_config.print()

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelength, scaler_geo = load_reflection_spectra_dataloaders(
        training_config.dataset_path,
        test_fraction=0.25,
        augment_spectra=False,
        p_aug=0.0
    )

    # setup model
    mlp = ForwardMLP(
        hidden_layers=layers,
        activation_name="GELU"
    )

    loss_function = ResonancePeaksLoss(
        w_amp=ranges['w_amp'][0], 
        w_grad=ranges['w_grad'][0], 
        w_wass=ranges['w_wass'][0], 
        w_sam=ranges['w_sam'][0],
        peaks_importance=True
    )

    loss_scheduler = LossScheduler(
        loss_function, 
        total_epochs=training_config.epochs,
        num_cycles=1,         # 1 cycle = linear ramp; >1 = warm restarts
        w_amp_range=ranges['w_amp'],
        w_grad_range=ranges['w_grad'],
        w_wass_range=ranges['w_wass'],
        w_sam_range=ranges['w_sam']
    )

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=training_config.learning_rate, weight_decay=1.5e-05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=training_config.patience
    )

    print("final model path", training_config.model_path)
    os.makedirs(training_config.model_dir, exist_ok=True)

    # training loop
    if not os.path.exists(training_config.model_path):
        print()
        print("Starting training...")
        
        trainer = ModelTrainer(
            train_loader,
            val_loader,
            loss_function,
            optimizer,
            scheduler,
            loss_scheduler=loss_scheduler
        )

        trainer.training_loop(mlp, validate_model, training_config, inverse=False)
        print('training finished')

        trainer.training_stats(training_config)        

    else:
        print(f"loading existing model from {training_config.model_path}")
        checkpoint = torch.load(training_config.model_path, weights_only=False)
        loss_function = ResonancePeaksLoss(
            w_amp=checkpoint['loss_config']['w_amp'], 
            w_grad=checkpoint['loss_config']['w_grad'], 
            w_wass=checkpoint['loss_config']['w_wass'], 
            w_sam=checkpoint['loss_config']['w_sam']
        )
        mlp.load_state_dict(checkpoint["model_state_dict"])

    # eval model on test set
    mlp.eval()

    # parity plot
    parity_plot_path = f"{training_config.model_dir}/{training_config.model_name}_parity_plot.png"
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

    results["weights_ranges"] = ranges

    # dump results to file
    json.dump(results, open(f"{training_config.model_dir}/{training_config.model_name}_results.json", "w+"), cls=MathEncoder, indent=4)

    # model test
    test_model(mlp, loss_function, x_test, y_test, 2, f"{training_config.model_dir}/{training_config.model_name}_test.png")


    print()
    print()
    print("# TANDEM MODEL ######################################################################################")
    print()
    print()

    
    use_augmenter = False
    augmenter = None
    
    config = TrainingConfig("tandem","tandem_params.env", forward_model_name=training_config.model_path)
    config.print()
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, x_test, y_test, wavelengths, scaler_geo_tandem = load_reflection_spectra_dataloaders(
        config.dataset_path,
        test_fraction=0.25,
        augment_spectra=False,
        p_aug=0.0
    )

    # # define forward and inverse models
    # # layers = [1024,512,256,128]
    # layers = [512,256,128]
    forward_model = ForwardMLP(hidden_layers=layers, activation_name="GELU") 
    # inverse_model = ForwardMLP(input_dim=81, output_dim=4, activation_name="GELU") 
    inverse_model = InverseCNN(output_geom_dim=4)

    # load last trained forward model

    checkpoint = torch.load(config.forward_model_name, weights_only=False)
    forward_model.load_state_dict(checkpoint["model_state_dict"])
    # forward_model = mlp

    # define tandem model
    tandem_model = TandemModel(inverse_model=inverse_model, forward_model=forward_model)
    
    # optimizer on inverse model's parameters
    optimizer = torch.optim.Adam(tandem_model.inverse_model.parameters(), lr=config.learning_rate)

    loss_function = ResonancePeaksLoss(
        w_amp=tandem_ranges['w_amp'][0], 
        w_grad=tandem_ranges['w_grad'][0], 
        w_wass=tandem_ranges['w_wass'][0], 
        w_sam=tandem_ranges['w_sam'][0],
        peaks_importance=True
    )
    loss_scheduler = LossScheduler(
        loss_function, 
        total_epochs=training_config.epochs,
        num_cycles=1,         # 1 cycle = linear ramp; >1 = warm restarts
        w_amp_range=tandem_ranges['w_amp'],
        w_grad_range=tandem_ranges['w_grad'],
        w_wass_range=tandem_ranges['w_wass'],
        w_sam_range=tandem_ranges['w_sam']
    )

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
            augmenter,
            loss_scheduler=loss_scheduler
        )

        trainer.training_loop(tandem_model, validate_tandem_model, config, inverse=True)

        print('training finished')

        trainer.training_stats(config)
        

    else:
        print(f"loading existing model from {config.model_path}")
        checkpoint = torch.load(config.model_path, weights_only=False)
        loss_function = ResonancePeaksLoss(
            w_amp=checkpoint['loss_config']['w_amp'], 
            w_grad=checkpoint['loss_config']['w_grad'], 
            w_wass=checkpoint['loss_config']['w_wass'], 
            w_sam=checkpoint['loss_config']['w_sam']
        )
        tandem_model.load_state_dict(checkpoint["model_state_dict"])

    # eval model on test set
    tandem_model.eval()
    
    test_spectra = torch.from_numpy(y_test).float().view(y_test.shape[0], -1)
    
    with torch.no_grad():
        geo_prediction, spectra_reconstructed = tandem_model(test_spectra)

    pred_spectra = spectra_reconstructed.numpy()
    test_spectra = test_spectra.numpy()


    print("test geometries")
    designs_predict_physical = scaler_geo.inverse_transform(geo_prediction)
    designs_predict_physical = designs_predict_physical[:9]
    sep = np.expand_dims(designs_predict_physical[:,3] - np.maximum(designs_predict_physical[:,1], designs_predict_physical[:,2]), axis=1)
    print(sep.shape)
    designs_predict_physical = np.hstack((np.expand_dims(designs_predict_physical[:,0], axis=1), sep, designs_predict_physical[:,[1,2,3]]))
    print('h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'Period Probe')
    print(designs_predict_physical)
    print()
    np.savetxt(os.path.join('geo_params.csv'), designs_predict_physical[:, :4], delimiter=',', header='h_pill, sep, d_pill, w_pill', comments='') # used for re simulation

    print("saving related reonstructed spectra")
    test_pred_spectra = pred_spectra[:9]
    np.savetxt(os.path.join('pred_spectra.csv'), test_pred_spectra[:, :81], delimiter=',')

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

    results["weights_ranges"] = tandem_ranges

    # dump results to file
    json.dump(results, open(f"{config.model_dir}/{config.model_name}_results.json", "w+"), cls=MathEncoder, indent=4)

    # model test
    test_tandem_model(loss_function, test_spectra, pred_spectra, 2, f"{config.model_dir}/{config.model_name}_test.png")


    print()
    print()
    print("# GENERATED SPECTRA TEST ###############################################################################")
    print()
    print()

    from utils.generate_spectra import generate_spectrum
    spectrum_test, params = generate_spectrum(
        num_points=81,
        num_peaks=1,
        # peak_type='lorentzian',
        peak_type='gaussian',
        noise_level=0.001,
        peak_spread=0.2
    )
    print(params)

    reconstruct_evaluate_spectrum(tandem_model, wavelengths, spectrum_test, output_path = "res_generated.png")

    # instead of generating spectrum, use an augmented test sample
    rng = np.random.default_rng()
    test_i = int(len(y_test) * rng.random())
    test_spectrum = torch.from_numpy(y_test[test_i]).float().view(y_test[test_i].shape[0], -1)
    from utils.dataloader import apply_smooth_flattening, compute_peak_bounds
    bounds = compute_peak_bounds(test_spectrum.numpy().squeeze(), 3)
    aug_signal = apply_smooth_flattening(test_spectrum.numpy().squeeze(), bounds)
    spectrum_test = aug_signal


    reconstruct_evaluate_spectrum(tandem_model, wavelengths, spectrum_test, output_path = "res_augmented_sample.png")


