import numpy as np
import pandas as pd

class TimeDependentData:
        
    def __init__(self,names: list):
        self.names = names
        self.entry = {name: [] for name in names}
    
    def addData(self, name: str, time: float, value: np.ndarray):
        if name not in self.names:
            raise ValueError(f"Name {name} not recognized.")
        self.entry[name].append((time, value))
        self.entry[name].sort(key=lambda x: x[0])  # keep sorted by time


class Provider:
        
    def __init__(self,
                name : str,
                data: TimeDependentData = None,
                noise : np.ndarray = np.array([[0.0]])):
    
        self.name = name
        self.noise = noise
        self.raw_data = data #raw data

        if data is None:
            self.raw_data = TimeDependentData([name])

    def values(self) -> np.ndarray:
        return np.array([y for t, y in self.raw_data.entry[self.name]])
    
    def times(self) -> np.ndarray:
        return np.array([item[0] for item in self.raw_data.entry[self.name]])
    
    def nData(self) -> int:
        return len(self.trans_data.entry[self.name])
    
        
    def getRawData(self) -> np.ndarray:
        return self.raw_data

    

class Control(TimeDependentData):
    
    def __init__(self,names: list):
        super().__init__(names)
    

class ControlProvider(Provider):
    
    def __init__(self,
                 name : str,
                 controls: Control = None,
                 noise : np.ndarray = np.array([[0.0]])):
        
        super().__init__(name, controls, noise)

    @property
    def nControls(self) -> int:
        return super().nData()
    
    @property
    def times(self) -> np.ndarray:
        return super().times()
    
    @property
    def controls(self) -> np.ndarray:
        return super().values()
    
    def addControl(self, name: str, time: float, value: np.ndarray):
        self.raw_data.addData(name, time, value)

class Mesuremtents(TimeDependentData):
    
    def __init__(self,names: list):
        super().__init__(names)


class MeasurementProvider(Provider):
    
    def __init__(self,
                 name : str,
                 measurements: Mesuremtents = None,
                 noise : np.ndarray = np.array([[0.0]])):
        
        super().__init__(name, measurements, noise)

    @property
    def numMeas(self) -> int:
        super().nData()

    @property
    def times(self) -> np.ndarray:
        return super().times()
    
    @property
    def measurements(self) -> np.ndarray:
        return super().values()
    
    def transformMeasures(self) -> np.ndarray:
        return super().transform()
    
    def getMeasurement(self) -> np.ndarray:
        return super().getRawData()
    
    def getMeasurement(self, time: float) -> np.ndarray:
        return super().getRawData(time)
    
    
    def addMeasurement(self, name: str, time: float, value: np.ndarray):
        self.raw_data.addData(name, time, value)
    

class DFProvider(MeasurementProvider):
    """
    Measurement provider based on a pandas DataFrame. Mostly for testing purposes.
    Each column contains a list of (time, value) tuples.
    """
    def __init__(self,
                name: str,
                DataFrame: pd.DataFrame,
                y_columns: list,
                noise: np.ndarray = np.array([[0.0]])
                ):
        
        super().__init__(name=name,
                         measurements=None,
                         noise=noise)
        
        self.df = DataFrame
        self.value_columns = y_columns
        self.noise = noise
        
        # Parse DataFrame and build raw_measure dictionary
        self._build_raw_measure()
    
    def _build_raw_measure(self):
        """Build a dictionary mapping time -> measurement array from DataFrame."""
        self.raw_measure = {}
        
        # Get all time points from the first column
        for col in self.value_columns:
            time_value_pairs = self.df[col].iloc[0]
            for time, value in time_value_pairs:
                if time not in self.raw_measure:
                    self.raw_measure[time] = []
                self.raw_measure[time].append(value)
        
        # Convert lists to numpy arrays
        for time in self.raw_measure:
            self.raw_measure[time] = np.array(self.raw_measure[time])
    
    def getMeasurement(self, time: float) -> np.ndarray:
        """Get measurement at specific time point."""
        if time not in self.raw_measure:
            raise ValueError(f"No data available at time {time}.")
        return self.raw_measure[time]
    
    @property
    def times(self) -> np.ndarray:
        """Get all available time points."""
        return np.array(sorted(self.raw_measure.keys()))



if __name__ == "__main__":

    # Example usage of DFProvider
    df = pd.DataFrame({
        "X": [[(0.0, 0.5), (1.0, 1.2), (2.0, 1.5)]],
        "S": [[(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)]]
    })
    provider = DFProvider("test",df, y_columns=["X", "S"])
    print(provider.raw_measure)  # → {0.0: array([0.5, 0.1]), 1.0: array([1.2, 0.2]), 2.0: array([1.5, 0.3])}

    
