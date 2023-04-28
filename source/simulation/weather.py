import numpy as np

def calculate_wind_gradient(coordinates: np.ndarray, ref_wind_speed: float, ref_height: float, alpha: float):
    """
    Calculate the wind direction and speed for a terrain represented by a numpy array using a power law formula.

    :param coordinates: A numpy array representing the terrain surface.
    :param ref_wind_speed: The reference wind speed at the reference height.
    :param ref_height: The reference height at which the reference wind speed is measured.
    :param alpha: The exponent of the power law formula.
    :return: Two numpy arrays representing the calculated wind direction and speed for the terrain.
    """
    # wind_dir = np.zeros(coordinates.shape[:2])
    wind_spd = np.zeros(coordinates.shape[:2])
    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            height = coordinates[i, j, 2]
            wind_spd[i, j] = ref_wind_speed * (height / ref_height) ** alpha
    return wind_spd