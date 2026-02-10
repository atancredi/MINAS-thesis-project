import numpy as np

def generate_spectrum(num_points=81, num_peaks=3, min_height=0.6, peak_type='lorentzian', noise_level=0.02, peak_spread=0.25):
    rng = np.random.default_rng() 
    
    x = np.arange(num_points)
    spectrum = np.zeros(num_points)
    ground_truth_params = []

    spectrum += rng.uniform(0.01, 0.03)

    margin = num_points * 0.1
    cluster_width_points = num_points * peak_spread
    max_start_index = num_points - margin - cluster_width_points
    min_start_index = margin
    if max_start_index > min_start_index:
        cluster_start = rng.uniform(min_start_index, max_start_index)
    else:
        cluster_start = margin
        cluster_width_points = num_points - (2 * margin)

    for _ in range(num_peaks):
        A = rng.uniform(min_height, 0.9)
        mu = rng.uniform(cluster_start, cluster_start + cluster_width_points)
        width = rng.uniform(2.0, 5.0) 

        if peak_type == 'gaussian':
            peak = A * np.exp(-((x - mu)**2) / (2 * width**2))
            params = {'type': 'gaussian', 'A': A, 'mu': mu, 'sigma': width}
            
        elif peak_type == 'lorentzian':
            peak = A * (width**2 / ((x - mu)**2 + width**2))
            params = {'type': 'lorentzian', 'A': A, 'mu': mu, 'gamma': width}
            
        spectrum += peak
        ground_truth_params.append(params)

    white_noise = rng.normal(0, noise_level, num_points)
    spectrum += white_noise
    
    spectrum = np.clip(spectrum, 0.0, 1.05) 

    spectrum = 1 - spectrum

    # spectrum = np.expand_dims(spectrum, axis=0) 
    return spectrum.astype(np.float32), ground_truth_params