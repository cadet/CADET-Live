import math
from typing import Optional


class PIDController:
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
        The accumulated error, corrected for output saturation.
    previous_error : float
        The error from the previous iteration.
    output_limits : tuple, optional
        Min and max output values.
    k_aw : float
        Back-calculation gain in inverse units of dt. The correction is scaled
        by the elapsed time.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        output_limits: tuple[Optional[float], Optional[float]] = (None, None),
        k_aw: float = 1.0,
    ) -> None:
        if not math.isfinite(k_aw) or k_aw < 0:
            raise ValueError(f"k_aw must be finite and non-negative, got {k_aw}")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.k_aw = k_aw
        self._setpoint = setpoint

        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits

    def _clamp_output(self, output: float) -> float:
        """Apply output limits."""
        min_val, max_val = self.output_limits

        if min_val is not None and output < min_val:
            return min_val
        if max_val is not None and output > max_val:
            return max_val

        return output

    def update(self, measurement: float, dt: float) -> float:
        """Update the PID controller."""
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

        setpoint_val = self._setpoint

        error = setpoint_val - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        # Apply output limits
        clamped_output = self._clamp_output(output)

        # Back-calculate the integral from the difference caused by saturation.
        if self.ki != 0:
            self.integral += self.k_aw * (clamped_output - output) * dt / self.ki

        self.previous_error = error

        return (clamped_output)

    def clear(self) -> None:
        """Clear the PID controller state."""
        self.integral = 0.0
        self.previous_error = 0.0

