import numpy as np
from typing import Callable, Union, Optional
import sys
from pathlib import Path

from Provider import ControlProvider, MeasurementProvider


class PID:
    """A PID controller for process control.

    Attributes
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    setpoint : float
        The desired target value.
    integral : float
        The accumulated integral term.
    previous_error : float
        The error from the previous iteration.
    output_limits : tuple, optional
        Min and max output values (anti-windup).
    """

    def __init__(self,
                 kp: float,
                 ki: float,
                 kd: float,
                 setpoint: Union[float, Callable[[float], float]] = 0.0,
                 output_limits: tuple[Optional[float], Optional[float]] = (None, None)) -> None:

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._setpoint_fn = self._normalize_setpoint(setpoint)
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits

    @staticmethod
    def _normalize_setpoint(setpoint: Union[float, Callable[[float], float]]) -> Callable[[float], float]:
        """Ensure the setpoint is callable in time."""
        if callable(setpoint):
            return setpoint
        return lambda t: float(setpoint)

    def set_setpoint(self, setpoint: Union[float, Callable[[float], float]]) -> None:
        """Update setpoint; accepts constant or callable of time t."""
        self._setpoint_fn = self._normalize_setpoint(setpoint)

    def current_setpoint(self, t: float = 0.0) -> float:
        return self._setpoint_fn(t)
    
    def _clamp_output(self, output: float) -> float:
        """Apply output limits."""
        min_val, max_val = self.output_limits
        if min_val is not None and output < min_val:
            return min_val
        if max_val is not None and output > max_val:
            return max_val
        return output

    def update(self, measurement: float, dt: float, t: Optional[float] = None) -> tuple[float, float]:
        """Update the PID controller.

        Parameters
        ----------
        measurement : float
            The current measured value.
        dt : float
            The time interval since the last update.
        t : float, optional
            Current time for time-varying setpoints. If None, defaults to 0.0.

        Returns
        -------
        tuple[float, float]
            A tuple of (time, control_output) compatible with ControlProvider.
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        time_val = 0.0 if t is None else t
        setpoint_val = self._setpoint_fn(time_val)
        error = setpoint_val - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        
        # Apply output limits
        clamped_output = self._clamp_output(output)
        
        self.previous_error = error

        return (time_val, clamped_output)
    
    def update_as_array(self, measurement: float, dt: float, t: Optional[float] = None) -> tuple[float, np.ndarray]:
        """Update and return control as numpy array for ControlProvider."""
        time_val, control = self.update(measurement, dt, t)
        return (time_val, np.array([[control]]))

    def reset(self) -> None:
        """Reset the PID controller state."""
        self.integral = 0.0
        self.previous_error = 0.0

