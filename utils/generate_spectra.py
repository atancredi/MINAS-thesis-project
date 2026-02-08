import numpy as np

def generate_spectrum(num_points=81, num_peaks=3, peak_type='lorentzian', noise_level=0.02):
    x = np.arange(num_points)
    spectrum = np.zeros(num_points)
    ground_truth_params = []

    spectrum += np.random.uniform(0.01, 0.03)

    for _ in range(num_peaks):
        A = np.random.uniform(0.3, 0.9) 
        mu = np.random.uniform(num_points * 0.1, num_points * 0.9)
        width = np.random.uniform(1.0, 3.0) 

        if peak_type == 'gaussian':
            peak = A * np.exp(-((x - mu)**2) / (2 * width**2))
            params = {'type': 'gaussian', 'A': A, 'mu': mu, 'sigma': width}
            
        elif peak_type == 'lorentzian':
            peak = A * (width**2 / ((x - mu)**2 + width**2))
            params = {'type': 'lorentzian', 'A': A, 'mu': mu, 'gamma': width}
            
        spectrum += peak
        ground_truth_params.append(params)

    white_noise = np.random.normal(0, noise_level, num_points)
    spectrum += white_noise
    
    spectrum = np.clip(spectrum, 0.0, 1.05) # clip spectrum

    # return dips
    spectrum = 1 - spectrum

    # XXX maybe unsqueeze spectrum here to preserve batch dimension

    return spectrum.astype(np.float32), ground_truth_params
