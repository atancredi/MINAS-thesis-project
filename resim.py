import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.helper import plot_comsol, read_comsol_csv
# from utils.dataloader import load_reflection_spectra_data


def get_spectra_from_comsol_file(file_path, columns, n_spectra=9):
    # get resimulated spectra from COMSOL file
    df_resim = read_comsol_csv(file_path)
    
    def get_rows(n=1, start=0):
        rows = []
        for i in range(start,n):
            rows.append([df_resim.get(x)[i] for x in columns])
        return rows
    

    # extract spectra
    resimulated_spectra = []
    for i in range(n_spectra):
        resimulated_spectra.append(get_rows(n=81*(i+1), start=81*i))
    
    return resimulated_spectra


test_resimulated = {
    "resimulated_spectra": "__resimulations/2026_02_10__Alessandro_resim.csv",
    "target_spectra": "__models/official/version001/generated/test_spectra_001_v3_04_80.csv",
    "predicted_spectra": "__models/official/version001/generated/pred_spectra_001_v3_04_80_test.csv",
    # "predicted_geometries": "__models/official/version001/generated/geo_params_001_v3_04_80_test.csv"
}

generated_resimulated = {
    "resimulated_spectra": "__resimulations/2026_02_12__Alessandro_resim_custom_peaks.csv",
    "target_spectra": "__models/official/version001/generated_custompeaks/test_spectra_001_v3_04_80_custompeaks.csv",
    "predicted_spectra": "__models/official/version001/generated_custompeaks/pred_spectra_001_v3_04_80_custompeaks.csv",
    # "predicted_geometries": "__models/official/version001/generated_custompeaks/geo_params_001_v3_04_80_custompeaks.csv"
}



def plot_resimulated_spectrum(source: str, spectrum_index: int, out_file = "resim.png"):
    
    if source == "g": #generated
        params = generated_resimulated
        type_str = "suggested"
    elif source == "t": # test
        params = test_resimulated
        type_str = "test dataset"
    else:
        raise ValueError("source must be g,t")
    
    # _, _, _, _, wavelength, scaler_geo = load_reflection_spectra_data("__data/pisa_data_2025_12_02.h5")

    columns = ['h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'freq (THz)', 'Reflectance (1)']
    resimulated_spectra = get_spectra_from_comsol_file(params["resimulated_spectra"], columns, n_spectra=9)


    # get target spectra and geometries
    df_target = pd.read_csv(params["target_spectra"], header=None)

    # get predicted spectra
    df_pred = pd.read_csv(params["predicted_spectra"], header=None)

    # get predicted geometries
    #   this is the file sent for the resimulation, so it has the header
    # df_geo_pred = pd.read_csv(params["predicted_geometries"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharey=True)
    
    target_np = df_target.to_numpy() # target spectra
    pred_np = df_pred.to_numpy() # predicted spectra
    # geo_pred_np = df_geo_pred.to_numpy() # predicted geometries

    # resim spectrum
    df_spectrum = pd.DataFrame(resimulated_spectra[spectrum_index], columns=columns)
    wavelength = df_spectrum.get(df_spectrum.columns[4])

    target = target_np[spectrum_index]
    pred = pred_np[spectrum_index]
    # pred_geo = geo_pred_np[spectrum_index]

    ax.plot(wavelength, target, label = f"Target", linestyle='--', color='red')

    ax.plot(wavelength, pred, label = f"Predicted", linestyle='--', color='green')

    # plot resimulated spectrum
    plot_comsol(
        df_spectrum,
        ax,
        ["Reflectance (1)"], # ["abs(emw.S11)^2 (1)"], #
        xdata_indx=4,
        resim=True
    )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Reflectance")

    fig.suptitle(f"Re-simulation of {type_str} design", fontsize=16)

    ax.legend(title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ all $\mu m$")
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', 
        #    title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ [$\mu m]$")

    # plt.tight_layout(rect=[0, 0.03, 0.8, 0.95])
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_file)


if __name__ == "__main__":
    from fire import Fire
    Fire(plot_resimulated_spectrum)