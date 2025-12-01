# this scipt implements methods to convert optical density (OD) reading to concentration

def lambert_beer_law(od_measurement , extinction_coefficient, wall_length = 1.0):
    """
    Convert optical density (OD) measurement to concentration using the Lambert-Beer law.
    Parameters:
    od_measurement (float): The optical density measurement.
    extinction_coefficient (float): The molar extinction coefficient
    wall_length (float): The wall length of the cuvette.
    Returns:
    float: The concentration.

    """
    
    return od_measurement  / ((extinction_coefficient * wall_length))
