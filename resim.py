import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.helper import plot_comsol, read_comsol_csv
from utils.dataloader import load_reflection_spectra_data


if __name__ == "__main__":

    _, _, _, _, wavelength, scaler_geo = load_reflection_spectra_data("__data/pisa_data_2025_12_02.h5")


    df_resim = read_comsol_csv("__resimulations/2026_02_10__Alessandro_resim.csv")
    columns = ['h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'freq (THz)', 'Reflectance (1)']

    def get_rows(n=1, start=0):
        rows = []
        for i in range(start,n):
            rows.append([df_resim.get(x)[i] for x in columns])
        return rows
    

    # extract spectra
    spectra = []
    n_spectra = 9
    for i in range(n_spectra):
        spectra.append(get_rows(n=81*(i+1), start=81*i))


    # get target spectra
    df_target = pd.read_csv("__models/official/version001/generated/test_spectra_001_v3_04_80.csv", header=None)
    df_geo_target = pd.read_csv("__models/official/version001/generated/test_geometries.csv", header=None)

    # get predicted spectra
    df_pred = pd.read_csv("__models/official/version001/generated/pred_spectra_001_v3_04_80_test.csv", header=None)



    spectra_to_plot = [1]

    fig, axes = plt.subplots(1, len(spectra_to_plot), figsize=(12, 6), sharey=True)
    if len(spectra_to_plot) == 1:
        axes = [axes]
    
    target_np = df_target.to_numpy()
    geo_target_np = df_geo_target.to_numpy()
    pred_np = df_pred.to_numpy()
    for j, spectrum_index in enumerate(spectra_to_plot):
        
        df_spectrum = pd.DataFrame(spectra[spectrum_index], columns=columns)
        target = target_np[spectrum_index]
        pred = pred_np[spectrum_index]


        target_geo = geo_target_np[spectrum_index]
        target_geo_transformed = scaler_geo.inverse_transform(target_geo.reshape(1, -1))[0]
        target_geo_str = " ".join([f"{i:.3f}" for i in target_geo_transformed])
        axes[j].plot(wavelength, target, label = f"Target {target_geo_str}", linestyle='--', color='red')

        axes[j].plot(wavelength, pred, label = "Predicted", linestyle='--', color='green')

        plot_comsol(
            df_spectrum,
            axes[j],
            ["Reflectance (1)"], # ["abs(emw.S11)^2 (1)"], #
            xdata_indx=4,
            resim=True
        )

        axes[j].set_xlabel("Frequency (THz)")
        if j == 0:
            axes[j].set_ylabel("Reflectance")

    # axes.set_title("Re-simulation of suggested designs run " + ID)
    fig.suptitle("Re-simulation of test dataset sample", fontsize=16)

    axes[0].legend(title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ all $\mu m$")
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', 
        #    title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ [$\mu m]$")

    # plt.tight_layout(rect=[0, 0.03, 0.8, 0.95])
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/resim_plotsv2_001_test_id{spectra_to_plot[0]}.png")
        