import numpy as np
import pandas as pd


class Mesuremtents:
    
    def __init__(self,names: list):
        self.names = names
        self.values = {name: [] for name in names}
    
    def addMeasurement(self, name: str, time: float, value: np.ndarray):
        if name not in self.names:
            raise ValueError(f"Measurement name {name} not recognized.")
        self.values[name].append((time, value))
        self.values[name].sort(key=lambda x: x[0])  # keep sorted by time


class MeasurementProvider:
    
    def __init__(self,
                 name : str,
                 measurements: Mesuremtents,
                 noise : np.ndarray = np.array([[0.0]]),
                 transformation: callable = lambda x: x):
        
        self.name = name
        self.noise = noise
        self.transformation = transformation
        
        self.raw_measurents = measurements.values[name]

    @property
    def nMeasurements(self) -> int:
        return len(self.raw_measurents)
    
    @property
    def values(self) -> np.ndarray:
        return np.array(list(self.raw_measurents.values()))
    
    @property
    def times(self) -> np.ndarray:
        return np.array(sorted(self.raw_measurents.keys()))
    
    def transformeMeasures(self) -> np.ndarray:
        """Apply transformation to all raw measurements."""
        transformed = {
            t: self.transformation(y)
            for t, y in self.raw_measurents.items()
        }
        return transformed
    
    def getRawMeasurement(self, time: float) -> np.ndarray:
        """Get the raw measurement at a specific time."""
        if time in self.raw_measurents:
            return self.raw_measurents[time]
        else:
            raise ValueError(f"No measurement available at time {time}.")
    
    def getTransformedMeasurement(self, time: float) -> np.ndarray:
        """Get the transformed measurement at a specific time."""
        raw = self.getRawMeasurement(time)
        return self.transformation(raw)
    

class DFProvider(MeasurementProvider):
    """
    Measurement provider based on a pandas DataFrame. Mostly for testing purposes.
    Each column contains a list of (time, value) tuples.
    """
    def __init__(self,
                DataFrame: pd.DataFrame,
                y_columns: list,
                noise: np.ndarray = np.array([[0.0]]),
                transformation: callable = lambda x: x
                ):
        
        super().__init__()
        
        self.df = DataFrame
        self.value_columns = y_columns
        self.noise = noise

        # Build a dict of {time: [values]} by merging all time points from all columns
        time_value_map = {}
        for col in self.value_columns:
            for time, value in self.df[col].iloc[0]:  # assuming each cell contains list of tuples
                if time not in time_value_map:
                    time_value_map[time] = {}
                time_value_map[col][time] = value
        
        # Convert to {time: array([val1, val2, ...])}
        self.raw_measure = {}
        for time in sorted(time_value_map.keys()):
            values = [time_value_map.get(col, {}).get(time, np.nan) for col in self.value_columns]
            self.raw_measure[time] = np.array(values)

        self.transformation = transformation
        self.state_measure = {
            t: self.transformation(y)
            for t, y in self.raw_measure.items()
        }

if __name__ == "__main__":

    # Example usage of DFProvider
    df = pd.DataFrame({
        "X": [[(0.0, 0.5), (1.0, 1.2), (2.0, 1.5)]],
        "S": [[(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)]]
    })
    provider = DFProvider(df, y_columns=["X", "S"])
    print(provider.raw_measure)  # → {0.0: array([0.5, 0.1]), 1.0: array([1.2, 0.2]), 2.0: array([1.5, 0.3])}
