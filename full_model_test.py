import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from mlp_pytorch import ForwardMLP
from cnn_pytorch_inverse import InverseCNN
from loss import ResonancePeaksLoss
from tandem_model import TandemModel
from test_model import test_tandem_model

from utils.generate_spectra import generate_spectrum
from utils.dataloader import apply_smooth_flattening, compute_peak_bounds, load_reflection_spectra_dataloaders
from utils.config import TrainingConfig
from generated_spectra_test import reconstruct_and_evaluate


def reconstruct_evaluate_spectrum(tandem_model, loss_function, wavelengths, spectrum_test, scaler, title_str, output_path = "res.png"):

    
    # reconstruct and evaluate spectra
    metrics, reconstructed_numpy, predicted_geo_numpy = reconstruct_and_evaluate(tandem_model, [spectrum_test], wavelengths)
    
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

    predicted_geo_numpy = scaler.inverse_transform(predicted_geo_numpy)

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.plot(wavelengths, y_true_np, label = f"Target", linestyle='--', color='red')
    # pred_geo_str = get_geo_str(predicted_geo_numpy)
    pred_geo_str = " ".join([f"{i:.3f}" for i in predicted_geo_numpy[0]])
    ax.plot(wavelengths, y_pred_np, label = f"Predicted {pred_geo_str}")
    
    ax.legend(title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ all $\mu m$")

    fig.suptitle(title_str, fontsize=16)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Reflectance")

    plt.tight_layout()
    plt.savefig(output_path) 
    

    return predicted_geo_numpy, y_pred, y_true, total.item()


def test_tandem_model(test_type: str, use_mse=False):


    training_config = TrainingConfig("mlp", "params.env")
    # training_config.print()

    tests_folder = ""

    if test_type == "t":
        print("Testing augmented test samples")
        test_aug = True
        # tests_folder = "tandem_tests/v3_ls_04_4layers/test_peaks/"
        title_str = "Prediction of test dataset sample"
    elif test_type == "g":
        print("Testing generated samples")
        test_aug = False
        # tests_folder = "tandem_tests/v3_ls_04_4layers/generated_peaks/"
        title_str = "Prediction of generated sample"
    else:
        raise ValueError("test_type must be t or g")
    
    
    if use_mse:
        print("using MSE loss")
        forward_model_path = "__models/official/version_mse/mlp_mse_20.pth"
        tandem_model_path = "__models/official/version_mse/tandem_cnn_mse_20.pth"
        title_str = title_str + " (MSE Loss)"
        # tests_folder = os.path.join(tests_folder, "mse/")
    else:
        print("using PIL loss")
        forward_model_path = "__models/official/version001/mlp_v3_ls_04_4layers_80.pth"
        tandem_model_path = "__models/official/version001/tandem_cnn_v3_ls_04_4layers_50.pth"
        title_str = title_str + " (Physics-Informed Loss)"
        # tests_folder = os.path.join(tests_folder, "pil/")

    print(title_str)
    print(tests_folder)
    print(forward_model_path)
    print(tandem_model_path)

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

    # define forward and inverse models
    layers = [1024,512,256,128]
    forward_model = ForwardMLP(hidden_layers=layers, activation_name="GELU") 
    inverse_model = InverseCNN(output_geom_dim=4)


    # load last trained forward model
    print(f"loading last trained forward model from {forward_model_path}")
    checkpoint = torch.load(forward_model_path, weights_only=False)
    forward_model.load_state_dict(checkpoint["model_state_dict"])


    # define tandem model
    tandem_model = TandemModel(inverse_model=inverse_model, forward_model=forward_model)
    
    print(f"loading existing tandem model from {tandem_model_path}")
    checkpoint = torch.load(tandem_model_path, weights_only=False)
    if use_mse:
        loss_function = torch.nn.MSELoss()
    else:
        loss_function = ResonancePeaksLoss(
            w_amp=checkpoint['loss_config']['w_amp'], 
            w_grad=checkpoint['loss_config']['w_grad'], 
            w_wass=checkpoint['loss_config']['w_wass'], 
            w_sam=checkpoint['loss_config']['w_sam'],
            v2=False
        )
    tandem_model.load_state_dict(checkpoint["model_state_dict"])

    
    # eval model on test set
    tandem_model.eval()
    

    # os.makedirs(tests_folder, exist_ok=True)
    if test_aug:
        # instead of generating spectrum, use an augmented test sample
        rng = np.random.default_rng()
        test_i = int(len(y_test) * rng.random())
        test_spectrum = torch.from_numpy(y_test[test_i]).float().view(y_test[test_i].shape[0], -1)
        bounds = compute_peak_bounds(test_spectrum.numpy().squeeze(), 3)
        aug_signal = apply_smooth_flattening(test_spectrum.numpy().squeeze(), bounds)
        out_file = os.path.join(tests_folder, "res_augmented_sample.png")
        reconstruct_evaluate_spectrum(tandem_model, loss_function, wavelengths, aug_signal, scaler=scaler_geo_tandem, title_str=title_str, output_path = out_file)

    else:
        # generate a spectrum
        spectrum_test, _ = generate_spectrum(
            num_points=81,
            num_peaks=1,
            peak_type='lorentzian',
            # peak_type='gaussian',
            noise_level=0.001,
            peak_spread=0.15
        )

        # i don't know the 'true' geometries because the spectrum is arbitrarily generated
        # geometries are already rescaled
        predicted_geo_numpy, y_pred, y_true, _ = reconstruct_evaluate_spectrum(tandem_model, loss_function, wavelengths, spectrum_test, scaler=scaler_geo_tandem, title_str=title_str, output_path = os.path.join(tests_folder, f"res_generated.png"))

        print("test geometries")
        designs_predict_physical = predicted_geo_numpy
        sep = np.expand_dims(designs_predict_physical[:,3] - np.maximum(designs_predict_physical[:,1], designs_predict_physical[:,2]), axis=1)
        designs_predict_physical = np.hstack((np.expand_dims(designs_predict_physical[:,0], axis=1), sep, designs_predict_physical[:,[1,2,3]]))
        # print('h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'Period Probe')
        # print(designs_predict_physical)
        # print()

        # np.savetxt(os.path.join(tests_folder, 'geo_params.csv'), designs_predict_physical[:, :4][0][np.newaxis, :], delimiter=',', header='h_pill, sep, d_pill, w_pill', comments='') # used for re simulation

        # print("saving related reonstructed spectra")
        # np.savetxt(os.path.join(tests_folder, 'pred_spectra.csv'), y_pred[:, :81][0][np.newaxis, :], delimiter=',')

        # print("saving test spectra and geometries samples")
        # np.savetxt(os.path.join(tests_folder, 'test_spectra.csv'), y_true[:, :81][0][np.newaxis, :], delimiter=',')

    print("Done")

if __name__ == '__main__':

    from fire import Fire
    Fire(test_tandem_model)
