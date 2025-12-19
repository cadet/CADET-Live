import numpy as np
import pandas as pd



class MeasureProvider:
    
    def __init__(self):
        
        self.noise = np.array([[0.0]])
        self.transformation = lambda x: x
        self.raw_measure = dict()
        self.state_measure = dict()
        self.state_name = "undefined"


    @property
    def nMeasurements(self) -> int:
        return len(self.raw_measure)
    
    @property
    def values(self) -> np.ndarray:
        return np.array(list(self.raw_measure.values()))
    
    @property
    def times(self) -> np.ndarray:
        return np.array(sorted(self.raw_measure.keys()))
    
    @property
    def nObservedStates(self) -> int:
        first_value = next(iter(self.raw_measure.values()))
        return first_value.shape[0] if hasattr(first_value, 'shape') else 1
    
    def getRawMeasurement(self, time: float) -> np.ndarray:
        """Gibt den Messwert für einen bestimmten Zeitpunkt zurück."""
        orig_time = time
        if time not in self.raw_measure:
            time = min(self.raw_measure.keys(), key=lambda t: abs(t - time)) # todo find other more stable solution
            
            if abs(orig_time - time) > 1e-8:
                print(f"Warning: No exact measurement at time {orig_time}, using closest available time {time} instead.")
        
        return self.raw_measure[time]
    
    def getStateMeasurement(self, time: float) -> np.ndarray:
        """Gibt den transformierten Messwert für einen bestimmten Zeitpunkt zurück."""
        orig_time = time
        if time not in self.state_measure:
            time = min(self.state_measure.keys(), key=lambda t: abs(t - time)) # todo find other more stable solution
            
            if abs(orig_time - time) > 1e-8:
                print(f"Warning: No exact measurement at time {orig_time}, using closest available time {time} instead.")
        
        return self.state_measure[time]
    
    def __repr__(self) -> str:
        return f"Measurement({self.state_name}, n={self.nMeasurements}, times={sorted(self.raw_measure.keys())})"
    

class DFProvider(MeasureProvider):
    """
    Measurement provider based on a pandas DataFrame. Mostly for testing purposes.
    """
    def __init__(self, 
                DataFrame: pd.DataFrame,
                x_column: str,
                y_columns: list,
                noise: np.ndarray = np.array([[0.0]]),
                transformation: callable = lambda x: x
                ):
        
        super().__init__()
        
        self.df = DataFrame
        self.time_column = x_column
        self.value_columns = y_columns
        self.noise = noise

        self.raw_measure = {
            row[x_column]: row[y_columns].values.astype(float)
            for _, row in DataFrame.iterrows()
        }

        self.transformation = transformation
        self.state_measure = {
            t: self.transformation(y)
            for t, y in self.raw_measure.items()
        }

if __name__ == "__main__":

    # Example usage of DFProvider
    df = pd.DataFrame({
        "t": [0.0, 1.0, 2.0],
        "X": [0.5, 1.2, 1.5],
        "S": [0.1, 0.2, 0.3]
    })
    provider = DFProvider(df, x_column="t", y_columns=["X", "S"])
    print(provider.getStateMeasurement(0.51))  # → array([1.2, 0.2])
