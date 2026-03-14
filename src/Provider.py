from typing import Dict, List, Optional, Tuple, Union

import threading

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

# TODO: add cadet specific info like column names, units, etc. to metadata or separate attribute


class TimeDependentData:
    """
    Stores time-series data for a SINGLE variable with associated metadata.

    This class represents one time-dependent variable.
    It stores both the data (time-value pairs) and metadata about the measurement
    (noise, state indices, etc.).
    """

    def __init__(
        self,
        name: str,
        noise: Optional[np.ndarray] = None,
        state_index: Optional[int] = None,  # noqa: E501
    ) -> None:
        self.name = name
        self._data: List[Tuple[float, np.ndarray]] = []  # [(time, value), ...]
        self._lock = threading.Lock()

        # Core metadata
        self.noise = noise if noise is not None else np.array([[0.0]])
        self.state_index = state_index

    def add_data(self, time: float, value: Union[float, np.ndarray]) -> None:
        """Add a data point and maintain time-sorted order. Thread-safe."""
        value_array = np.atleast_1d(value)
        with self._lock:
            self._data.append((time, value_array))
            self._data.sort(key=lambda x: x[0])

    def get_data(
        self, time: float, method: str = "exact"
    ) -> Optional[np.ndarray]:
        """Retrieve data at a specific time point.

        Parameters
        ----------
        time : float
            The time point to retrieve data for.
        method : str
            Method for retrieval: 'exact', 'nearest', or 'interpolate'.

        Returns
        -------
        Optional[np.ndarray]
            The data value at the specified time, or None if not found.
        """
        if not self._data:
            return None

        if method == "exact":
            for t, v in self._data:
                if np.isclose(t, time):
                    return v.copy()
            return None

        elif method == "nearest":
            times = np.array([t for t, _ in self._data])
            idx = np.argmin(np.abs(times - time))
            return self._data[idx][1].copy()

        elif method == "interpolate":
            # Linear interpolation between two nearest points
            times = np.array([t for t, _ in self._data])
            if time < times[0] or time > times[-1]:
                return None  # Outside range

            # Find bracketing indices
            idx_after = np.searchsorted(times, time)
            if idx_after == 0:
                return self._data[0][1].copy()

            idx_before = idx_after - 1
            t_before, v_before = self._data[idx_before]
            t_after, v_after = self._data[idx_after]

            # Linear interpolation
            alpha = (time - t_before) / (t_after - t_before)
            return v_before + alpha * (v_after - v_before)

        else:
            raise ValueError(f"Unknown method: {method}")

    @property
    def times(self) -> np.ndarray:
        """Get all time points."""
        return np.array([t for t, _ in self._data])

    @property
    def values(self) -> np.ndarray:
        """Get all values as array."""
        if not self._data:
            return np.array([])
        return np.array([v for _, v in self._data])

    @property
    def data_points(self) -> List[Tuple[float, np.ndarray]]:
        """Get raw data as list of (time, value) tuples."""
        return self._data.copy()

    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self._data)


class Provider:
    """Time-dependent data provider for multiple variables.

    Manages a collection of TimeDependentData objects representing
    multiple variables measured over time.
    """

    def __init__(
        self,
        name: str,
        variable_names: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self._data: Dict[str, TimeDependentData] = {}

        if variable_names is not None:
            for var_name in variable_names:
                self._data[var_name] = TimeDependentData(name=var_name)

    def add_variable(self,
                     name: str,
                     noise: Optional[np.ndarray] = None,
                     state_index: Optional[int] = None) -> TimeDependentData:
        """Add a new variable to this provider."""
        tdd = TimeDependentData(
            name=name,
            noise=noise,
            state_index=state_index
        )
        self._data[name] = tdd
        return tdd

    def get_variable(self, name: str) -> Optional[TimeDependentData]:
        """Get TimeDependentData for a specific variable."""
        return self._data.get(name)

    def add_data(self,
                 variable_name: str,
                 time: float,
                 value: Union[float, np.ndarray]) -> None:
        """Add data point to a specific variable."""
        if variable_name not in self._data:
            self._data[variable_name] = TimeDependentData(name=variable_name)
        self._data[variable_name].add_data(time, value)

    def get_data(self,
                 variable_name: str,
                 time: float,
                 method: str = 'exact') -> Optional[np.ndarray]:
        """Get data for a specific variable at a time point."""
        if variable_name not in self._data:
            return None
        return self._data[variable_name].get_data(time, method)

    def get_all_data_at_time(
        self, time: float, method: str = "exact"
    ) -> Dict[str, np.ndarray]:
        """Retrieve all variable data at a specific time point.

        Parameters
        ----------
        time : float
            The time to retrieve data for.
        method : str
            Method for retrieval: 'exact', 'nearest', or 'interpolate'.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping variable names to their data values.
        """
        result = {}
        for var_name, tdd in self._data.items():
            value = tdd.get_data(time, method)
            if value is not None:
                result[var_name] = value
        return result

    def get_data_vector(
        self,
        time: float,
        variable_order: Optional[List[str]] = None,
        method: str = "exact",
    ) -> Optional[np.ndarray]:
        """Retrieve data vector for all variables at a time point.

        Parameters
        ----------
        time : float
            The time point to retrieve data for.
        variable_order : Optional[List[str]]
            Order of variables in output. If None, uses sorted order.
        method : str
            Method for retrieval: 'exact', 'nearest', or 'interpolate'.

        Returns
        -------
        Optional[np.ndarray]
            Concatenated data vector for all variables, or None if incomplete.
        """
        if variable_order is None:
            variable_order = sorted(self._data.keys())

        measurements = []
        for var_name in variable_order:
            value = self.get_data(var_name, time, method)
            if value is None:
                return None  # Missing data for this variable
            measurements.append(np.atleast_1d(value))

        if not measurements:
            return None
        return np.concatenate(measurements)

    @property
    def all_times(self) -> np.ndarray:
        """Get sorted array of all unique time points across all variables."""
        all_times = set()
        for tdd in self._data.values():
            all_times.update(tdd.times)
        return np.array(sorted(all_times))

    @property
    def variable_names(self) -> List[str]:
        """Get list of all variable names in this provider."""
        return list(self._data.keys())

    @property
    def noise_matrix(self) -> np.ndarray:
        """
        Get combined noise covariance matrix for all variables.

        Returns block-diagonal matrix of individual variable noises.
        """
        noise_blocks = []
        for var_name in sorted(self._data.keys()):
            noise_blocks.append(self._data[var_name].noise)

        if not noise_blocks:
            return np.array([[0.0]])

        # Create block diagonal matrix
        return block_diag(*noise_blocks)

    def __repr__(self) -> str:
        """Return string representation of Provider."""
        return f"Provider(name='{self.name}', variables={self.variable_names})"


class MeasurementProvider(Provider):
    """Provider for measurement data with noise characteristics.

    Specialized provider that handles measurement-specific functionality
    including noise covariance matrices.
    """

    def __init__(
        self,
        name: str,
        variable_names: Optional[List[str]] = None,
        noise: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, variable_names)

        # For convenience: if single noise provided for multiple variables
        if noise is not None and variable_names is not None:
            for var_name in variable_names:
                if var_name in self._data:
                    self._data[var_name].noise = noise

    def add_measurement(self,
                       variable_name: str,
                       time: float,
                       value: Union[float, np.ndarray]) -> None:
        """Add measurement."""
        self.add_data(variable_name, time, value)

    def get_measurement(
        self, time: float, method: str = "exact"
    ) -> Optional[np.ndarray]:
        """Retrieve measurement vector at a specific time point.

        Parameters
        ----------
        time : float
            The time point to retrieve measurement for.
        method : str
            Method for retrieval: 'exact', 'nearest', or 'interpolate'.

        Returns
        -------
        Optional[np.ndarray]
            Measurement vector or None if data unavailable.
        """
        return self.get_data_vector(time, method=method)

    @property
    def times(self) -> np.ndarray:
        """All measurement times."""
        return self.all_times

    @property
    def noise(self) -> np.ndarray:
        """Noise covariance matrix."""
        return self.noise_matrix

    @property
    def measurements(self) -> np.ndarray:
        """
        All measurements as 2D array.

        Returns array of shape (n_times, n_variables).
        """
        times = self.all_times
        var_names = sorted(self.variable_names)

        result = []
        for t in times:
            meas = self.get_data_vector(t, var_names, method='exact')
            if meas is not None:
                result.append(meas)

        return np.array(result) if result else np.array([])


class ControlProvider(Provider):
    """Specialized provider for control inputs."""

    def add_control(self,
                   variable_name: str,
                   time: float,
                   value: Union[float, np.ndarray]) -> None:
        """Add control input."""
        self.add_data(variable_name, time, value)

    @property
    def times(self) -> np.ndarray:
        """All control times."""
        return self.all_times

    @property
    def controls(self) -> np.ndarray:
        """All controls as 2D array."""
        times = self.all_times
        var_names = sorted(self.variable_names)

        result = []
        for t in times:
            ctrl = self.get_data_vector(t, var_names, method='exact')
            if ctrl is not None:
                result.append(ctrl)

        return np.array(result) if result else np.array([])


class DFProvider(MeasurementProvider):
    """Measurement provider initialized from a pandas DataFrame.

    Parses measurement data from a DataFrame and creates a provider
    with time-dependent data for specified columns.
    """

    def __init__(
        self,
        name: str,
        dataframe: pd.DataFrame,
        y_columns: List[str],
        noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> None:
        super().__init__(name, variable_names=y_columns)

        self.df = dataframe
        self.value_columns = y_columns

        # Parse DataFrame and populate TimeDependentData objects
        self._parse_dataframe(noise)

    def _parse_dataframe(
        self, noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    ) -> None:
        """Parse DataFrame and create TimeDependentData for each column."""
        for col_name in self.value_columns:
            if col_name not in self.df.columns:
                raise ValueError(f"Column '{col_name}' not found in DataFrame")

            # Extract time-value pairs from column
            time_value_pairs = self.df[col_name].iloc[0]

            # Determine noise for this variable
            if isinstance(noise, dict):
                var_noise = noise.get(col_name, np.array([[0.0]]))
            elif noise is not None:
                var_noise = noise
            else:
                var_noise = np.array([[0.0]])

            # Create TimeDependentData
            tdd = TimeDependentData(
                name=col_name,
                noise=var_noise
            )

            # Add all data points
            for time, value in time_value_pairs:
                tdd.add_data(time, value)

            self._data[col_name] = tdd


if __name__ == "__main__":

    tdd = TimeDependentData("X", noise=np.array([[0.05]]))
    tdd.add_data(0.0, 1.0)
    tdd.add_data(1.0, 1.5)
    tdd.add_data(2.0, 2.1)
    print(f"TimeDependentData: {tdd}")
    print(f"Times: {tdd.times}")
    print(f"Values: {tdd.values}")
    print(f"Value at t=1.0: {tdd.get_data(1.0, 'exact')}")
    print(f"Value at t=0.5 (interpolated): {tdd.get_data(0.5, 'interpolate')}")
    print()

    print("=== Example 2: MeasurementProvider with Multiple Variables ===")
    provider = MeasurementProvider(name="ProcessAnalyzer")
    provider.add_variable("X", noise=np.array([[0.05**2]]), state_index=0)
    provider.add_variable("S", noise=np.array([[0.03**2]]), state_index=1)

    # Add data points
    for t in [0.0, 1.0, 2.0]:
        provider.add_data("X", t, np.random.rand())
        provider.add_data("S", t, np.random.rand())

    print(f"Provider: {provider}")
    print(f"Variable names: {provider.variable_names}")
    print(f"All times: {provider.all_times}")
    print(f"Measurement vector at t=1.0: {provider.get_measurement(1.0)}")
    print(f"Noise matrix:\n{provider.noise}")

    # Access metadata
    x_data = provider.get_variable("X")
    print(f"X metadata - noise: {x_data.noise}, state_index: {x_data.state_index}")
    print()

    print("=== Example 3: DFProvider ===")
    df = pd.DataFrame({
        "X": [[(0.0, 0.5), (1.0, 1.2), (2.0, 1.5)]],
        "S": [[(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)]]
    })

    df_provider = DFProvider(
        name="LabData",
        dataframe=df,
        y_columns=["X", "S"],
        noise={"X": np.array([[0.05**2]]), "S": np.array([[0.03**2]])}
    )

    print(f"DFProvider: {df_provider}")
    print(f"Times: {df_provider.times}")
    print(f"Measurement at t=1.0: {df_provider.get_measurement(1.0)}")
    print(f"All measurements:\n{df_provider.measurements}")
