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

    _transformed = False
        
    def __init__(self,
                name : str,
                data: TimeDependentData = None,
                noise : np.ndarray = np.array([[0.0]]),
                transformation: callable = lambda x: x):
    
        self.name = name
        self.noise = noise
        self.transformation = transformation
        self.rawData = data #raw data
        self.data = data # the data that can be transformed

        if data is None:
            self.rawData = TimeDependentData([name])
            self.data = TimeDependentData([name])

    def values(self) -> np.ndarray:
        if self._transformed:
            return np.array([self.data[t] for t in self.times])
        else:
            return np.array([y for t, y in self.rawData.entry[self.name]])
    
    def times(self) -> np.ndarray:
        if self._transformed:
            return np.array([item[0] for item in self.data.entry[self.name]])
        else:
            return np.array([item[0] for item in self.rawData.entry[self.name]])
    
    def nData(self) -> int:
        return len(self.data.entry[self.name])
    
    def transform(self):
       self._transformed = True
       values =  self.values
       self.data = {t: self.transformation(y) for t, y in zip(self.times, values)}

    def hasTransformed(self) -> bool:
        return self._transformed
        
    def getRawData(self) -> np.ndarray:
        return self.rawData
    
    def getRawData(self, time: float) -> np.ndarray:
        for t, value in self.rawData.entry[self.name]:
            if t == time:
                return value
        raise ValueError(f"No data available at time {time}.")
    
    def getData(self, time: float) -> np.ndarray:
        return self.data[time]
    
    def getData(self) -> np.ndarray:
        return self.data
    

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
        self.rawData.addData(name, time, value)
        
        if not self._transformed:
            self.data.addData(name, time, value)
        else:
            transformed_value = self.transformation(value)
            self.data.addData(name, time, transformed_value)



class Mesuremtents(TimeDependentData):
    
    def __init__(self,names: list):
        super().__init__(names)


class MeasurementProvider(Provider):
    
    def __init__(self,
                 name : str,
                 measurements: Mesuremtents = None,
                 noise : np.ndarray = np.array([[0.0]]),
                 transformation: callable = lambda x: x):
        
        super().__init__(name, measurements, noise, transformation)

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
    
    def getRawMeasurement(self) -> np.ndarray:
        return super().getRawData()
    
    def getRawMeasurement(self, time: float) -> np.ndarray:
        return super().getRawData(time)
    
    def getMeasurement(self, time: float) -> np.ndarray:
        return super().getData(time)
    
    def addMeasurement(self, name: str, time: float, value: np.ndarray):
        self.rawData.addData(name, time, value)
        if not self._transformed:
            self.data.addData(name, time, value)
        else:
            transformed_value = self.transformation(value)
            self.data.addData(name, time, transformed_value)
    

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

        time_value_map = {}
        for col in self.value_columns:
            for time, value in self.df[col].iloc[0]:  # assuming each cell contains list of tuples
                if time not in time_value_map:
                    time_value_map[time] = {}
                time_value_map[col][time] = value
        
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

    
