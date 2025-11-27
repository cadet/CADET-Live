# this scipt implements a methods to convert optical density (OD) reading to concentration

def lambert_beer_law(od_measurement , extinction_coefficient, path_length):
    """
    Convert optical density (OD) measurement to concentration using the Lambert-Beer law.
    Parameters:
    od_measurement (float): The optical density measurement.
    extinction_coefficient (float): The molar extinction coefficient
    path_length (float): The path length of the cuvette.
    Returns:
    float: The concentration.

    """
    
    return od_measurement  / ((extinction_coefficient * path_length))
